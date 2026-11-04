"""Protocol checks for the in-Blender true-3D tire render script.

The ``bpy`` module only exists inside Blender, so these tests install a fake
module that records every operator call and object the script makes.  The
checks therefore prove the protocol the script speaks, not a rendered image.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "blender_tire_render.py"


class FakeVector:
    """A minimal stand-in for mathutils vectors on object transforms."""

    def __init__(self, values: tuple[float, float, float] = (0.0, 0.0, 0.0)):
        self.values = tuple(float(value) for value in values)

    def __getitem__(self, index: int) -> float:
        return self.values[index]

    def __iter__(self):
        return iter(self.values)

    def __len__(self) -> int:
        return 3


class FakeVertex:
    """One mesh vertex."""

    def __init__(self, co: tuple[float, float, float]):
        self.co = co


class FakePolygon:
    """One mesh polygon with a radial position for the material split."""

    def __init__(self, index: int, vertices: list[int]):
        self.index = index
        self.vertices = vertices
        self.material_index = 0
        self.use_smooth = False


class FakeMesh:
    """A minimal mesh with vertices, polygons, edges, and material slots."""

    def __init__(self, name: str, vertices: list[tuple[float, float, float]]):
        self.name = name
        self.vertices = [FakeVertex(co) for co in vertices]
        self.polygons: list[FakePolygon] = []
        self.edges: list[tuple[int, int]] = []
        self.materials: list[Any] = []

    def add_quad(self, indices: list[int]) -> None:
        """Add one quad face and its edges."""
        self.polygons.append(FakePolygon(len(self.polygons), indices))
        for position in range(4):
            self.edges.append((indices[position], indices[(position + 1) % 4]))


class FakeCamera:
    """One orthographic camera definition."""

    def __init__(self, name: str):
        self.name = name
        self.type = "PERSP"
        self.ortho_scale = 6.0
        self.sensor_fit = "AUTO"
        self.clip_start = 0.1
        self.clip_end = 100.0


class FakeLight:
    """One area light definition."""

    def __init__(self, name: str):
        self.name = name
        self.type = "POINT"
        self.energy = 100.0
        self.size = 1.0


class FakeMaterial:
    """One material with a Principled shader node."""

    def __init__(self, name: str):
        self.name = name
        self.use_nodes = False
        self.node_tree = types.SimpleNamespace(
            nodes=_NodeLookup(),
        )


class _NodeLookup:
    """A tiny node collection that only knows the Principled shader."""

    def get(self, name: str) -> Any:
        if name != "Principled BSDF":
            return None
        return _Shader()


class _Shader:
    """A Principled shader node with settable input defaults."""

    def __init__(self) -> None:
        self.inputs = _Inputs()


class _Inputs:
    """Shader inputs that accept a default value."""

    def __init__(self) -> None:
        self.default_value = None
        self.values: dict[str, Any] = {}

    def __getitem__(self, key: str) -> Any:
        return self.values.setdefault(key, _Input())


class _Input:
    """One shader input socket."""

    def __init__(self) -> None:
        self.default_value = None


class FakeObject:
    """One scene object with a transform, a parent, and optional data."""

    def __init__(self, name: str, data: Any, object_type: str):
        self.name = name
        self.data = data
        self.type = object_type
        self.location = FakeVector()
        self.rotation_euler = FakeVector()
        self.scale = FakeVector((1.0, 1.0, 1.0))
        self.parent: Any = None
        self.empty_display_size = 1.0
        self.selected = False

    def select_set(self, selected: bool) -> None:
        """Record the selection state of one object."""
        self.selected = bool(selected)


class FakeTextData:
    """The text curve data of one font object."""

    def __init__(self, name: str):
        self.name = name
        self.body = ""
        self.align_x = "LEFT"
        self.align_y = "BASELINE"
        self.size = 1.0
        self.extrude = 0.0
        self.materials: list[Any] = []


class FakeCollection:
    """A collection of objects that can also create new ones."""

    def __init__(self, kind: str):
        self.kind = kind
        self.items: list[Any] = []
        self.objects: FakeCollection = self

    def new(self, name: str, data: Any = None, **kwargs: Any) -> Any:
        """Create one object whose type follows the data it is given."""
        if self.kind == "MATERIALS":
            material = FakeMaterial(name)
            self.items.append(material)
            return material
        if self.kind == "CAMERAS":
            camera = FakeCamera(name)
            self.items.append(camera)
            return camera
        if self.kind == "LIGHTS":
            light = FakeLight(name)
            light.type = str(kwargs.get("type", light.type))
            self.items.append(light)
            return light
        if data is None:
            object_type = "EMPTY"
        elif isinstance(data, FakeMesh):
            object_type = "MESH"
        elif isinstance(data, FakeTextData):
            object_type = "FONT"
        elif isinstance(data, FakeCamera):
            object_type = "CAMERA"
        elif isinstance(data, FakeLight):
            object_type = "LIGHT"
        else:
            object_type = "MATERIAL"
        item = FakeObject(name, data, object_type)
        self.items.append(item)
        return item

    def link(self, obj: Any) -> None:
        """Link one object into the collection."""
        self.items.append(obj)

    def __iter__(self):
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)


class FakeImageSettings:
    """The image format settings of one render."""

    def __init__(self) -> None:
        self.file_format = "PNG"
        self.color_mode = "RGBA"
        self.color_depth = "8"


class FakeRender:
    """The render settings of one scene."""

    def __init__(self) -> None:
        self.engine = "BLENDER_EEVEE_NEXT"
        self.resolution_x = 1920
        self.resolution_y = 1080
        self.resolution_percentage = 100
        self.film_transparent = False
        self.filepath = ""
        self.image_settings = FakeImageSettings()


class FakeCycles:
    """The Cycles settings of one scene."""

    def __init__(self) -> None:
        self.samples = 64
        self.device = "CPU"
        self.use_denoising = True


class FakeViewSettings:
    """The colour management settings of one scene."""

    def __init__(self) -> None:
        self.view_transform = "AgX"
        self.look = "None"


class FakeScene:
    """The scene that the fake operators write into."""

    def __init__(self) -> None:
        self.render = FakeRender()
        self.cycles = FakeCycles()
        self.view_settings = FakeViewSettings()
        self.collection = FakeCollection("SCENE")
        self.camera: Any = None


class FakeContext:
    """The context that holds the active object and the active scene."""

    def __init__(self, scene: FakeScene):
        self.scene = scene
        self.active_object: Any = None
        self.view_layer = types.SimpleNamespace(objects=types.SimpleNamespace(active=None))


def _torus_mesh(name: str, major: float, minor: float) -> FakeMesh:
    """Return a coarse torus mesh whose polygons sample the tube."""
    mesh = FakeMesh(name, [])
    major_segments = 8
    minor_segments = 8
    for major_index in range(major_segments):
        u = 2.0 * math.pi * major_index / major_segments
        for minor_index in range(minor_segments):
            v = 2.0 * math.pi * minor_index / minor_segments
            radius = major + minor * math.cos(v)
            mesh.vertices.append(
                FakeVertex(
                    (
                        radius * math.cos(u),
                        radius * math.sin(u),
                        minor * math.sin(v),
                    )
                )
            )
    for major_index in range(major_segments):
        for minor_index in range(minor_segments):
            a = major_index * minor_segments + minor_index
            b = major_index * minor_segments + (minor_index + 1) % minor_segments
            c = ((major_index + 1) % major_segments) * minor_segments + (
                minor_index + 1
            ) % minor_segments
            d = ((major_index + 1) % major_segments) * minor_segments + minor_index
            mesh.add_quad([a, b, c, d])
    return mesh


def _cube_mesh(name: str, size: float) -> FakeMesh:
    """Return a unit cube mesh scaled to one requested size."""
    half = size / 2.0
    mesh = FakeMesh(
        name,
        [
            (-half, -half, -half),
            (half, -half, -half),
            (half, half, -half),
            (-half, half, -half),
            (-half, -half, half),
            (half, -half, half),
            (half, half, half),
            (-half, half, half),
        ],
    )
    for face in (
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [1, 2, 6, 5],
        [0, 3, 7, 4],
    ):
        mesh.add_quad(face)
    return mesh


def _cylinder_mesh(name: str, radius: float, depth: float, vertices: int) -> FakeMesh:
    """Return a capped cylinder mesh."""
    mesh = FakeMesh(name, [])
    half = depth / 2.0
    for level in (half, -half):
        for index in range(vertices):
            angle = 2.0 * math.pi * index / vertices
            mesh.vertices.append(
                FakeVertex((radius * math.cos(angle), radius * math.sin(angle), level))
            )
    for index in range(vertices):
        following = (index + 1) % vertices
        mesh.add_quad([index, following, vertices + following, vertices + index])
    mesh.polygons.append(FakePolygon(len(mesh.polygons), list(range(vertices))))
    mesh.polygons.append(FakePolygon(len(mesh.polygons), list(range(vertices, 2 * vertices))))
    return mesh


def _label_mesh(name: str, body: str) -> FakeMesh:
    """Return the mesh that a text object converts into."""
    mesh = FakeMesh(name, [])
    half_width = 0.5 * max(0.1, 0.12 * len(body))
    half_height = 0.065
    depth = 0.008
    for corner in (
        (-half_width, -half_height, -depth),
        (half_width, -half_height, -depth),
        (half_width, half_height, -depth),
        (-half_width, half_height, -depth),
        (-half_width, -half_height, depth),
        (half_width, -half_height, depth),
        (half_width, half_height, depth),
        (-half_width, half_height, depth),
    ):
        mesh.vertices.append(FakeVertex(corner))
    for face in (
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [1, 2, 6, 5],
        [0, 3, 7, 4],
    ):
        mesh.add_quad(face)
    return mesh


class FakeOps:
    """The operator namespace of the fake bpy module.

    Every call is recorded so a test can prove which Blender operators the
    render script speaks and with which arguments.
    """

    wm: Any
    object: Any
    mesh: Any
    render: Any
    export_scene: Any

    def __init__(self, module: types.ModuleType):
        self.module = module
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.gltf_error: Exception | None = None

    def record(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Record one operator call and return a finished result."""
        self.calls.append((name, dict(arguments)))
        return {"FINISHED": True}

    def names(self) -> list[str]:
        """Return the names of every recorded operator call."""
        return [name for name, _ in self.calls]

    def arguments(self, name: str) -> list[dict[str, Any]]:
        """Return the arguments of every recorded call of one operator."""
        return [arguments for called, arguments in self.calls if called == name]

    def wm_read_factory_settings(self, use_empty: bool = False) -> dict[str, Any]:
        """Reset the fake scene the way a factory startup would."""
        self.record("wm.read_factory_settings", {"use_empty": use_empty})
        module = self.module
        module.data.objects.items.clear()
        module.data.materials.items.clear()
        scene = module.context.scene
        scene.collection.items.clear()
        scene.camera = None
        scene.render = FakeRender()
        scene.cycles = FakeCycles()
        return {"FINISHED": True}

    def wm_save_as_mainfile(self, filepath: str, check_existing: bool = True) -> dict[str, Any]:
        """Write a placeholder blend file and record the call."""
        self.record(
            "wm.save_as_mainfile",
            {"filepath": filepath, "check_existing": check_existing},
        )
        Path(filepath).write_bytes(b"BLENDER-placeholder")
        return {"FINISHED": True}

    def object_select_all(self, action: str = "SELECT") -> dict[str, Any]:
        """Record a selection call."""
        self.record("object.select_all", {"action": action})
        return {"FINISHED": True}

    def object_delete(self) -> dict[str, Any]:
        """Record a delete call."""
        self.record("object.delete", {})
        return {"FINISHED": True}

    def mesh_primitive_torus_add(self, **arguments: Any) -> dict[str, Any]:
        """Create the tyre carcass torus."""
        self.record("mesh.primitive_torus_add", arguments)
        mesh = _torus_mesh("torus", arguments["major_radius"], arguments["minor_radius"])
        obj = self.module.data.objects.new("tire-carcass", mesh)
        self.module.context.active_object = obj
        return {"FINISHED": True}

    def mesh_primitive_cube_add(self, **arguments: Any) -> dict[str, Any]:
        """Create one tread or spoke block."""
        self.record("mesh.primitive_cube_add", arguments)
        mesh = _cube_mesh("cube", arguments.get("size", 1.0))
        obj = self.module.data.objects.new("block", mesh)
        self.module.context.active_object = obj
        return {"FINISHED": True}

    def mesh_primitive_cylinder_add(self, **arguments: Any) -> dict[str, Any]:
        """Create one cylinder."""
        self.record("mesh.primitive_cylinder_add", arguments)
        mesh = _cylinder_mesh(
            "cylinder",
            arguments["radius"],
            arguments["depth"],
            arguments.get("vertices", 32),
        )
        obj = self.module.data.objects.new("cylinder", mesh)
        self.module.context.active_object = obj
        return {"FINISHED": True}

    def object_text_add(self, **arguments: Any) -> dict[str, Any]:
        """Create one font object."""
        self.record("object.text_add", arguments)
        data = FakeTextData("tire-label")
        obj = self.module.data.objects.new("tire-label", data)
        self.module.context.active_object = obj
        return {"FINISHED": True}

    def object_convert(self, target: str = "MESH") -> dict[str, Any]:
        """Convert the active font object into a mesh."""
        self.record("object.convert", {"target": target})
        obj = self.module.context.active_object
        if obj is None or obj.type != "FONT":
            raise RuntimeError("no font object is active")
        body = getattr(obj.data, "body", "")
        obj.data = _label_mesh("tire-label", body)
        obj.type = "MESH"
        return {"FINISHED": True}

    def render_render(self, write_still: bool = False) -> dict[str, Any]:
        """Write a placeholder render at the configured filepath."""
        self.record("render.render", {"write_still": write_still})
        filepath = self.module.context.scene.render.filepath
        if not filepath:
            raise RuntimeError("the render filepath is empty")
        Image.new("RGBA", (8, 8), (0, 0, 0, 0)).save(filepath)
        return {"FINISHED": True}

    def export_scene_gltf(self, filepath: str, export_format: str = "GLB") -> dict[str, Any]:
        """Write a placeholder glTF binary or raise the recorded failure."""
        self.record("export_scene.gltf", {"filepath": filepath, "export_format": export_format})
        if self.gltf_error is not None:
            raise self.gltf_error
        Path(filepath).write_bytes(b"GLB-placeholder")
        return {"FINISHED": True}


class FakeBpy(types.ModuleType):
    """The fake bpy module with the attributes the render script uses."""

    app: Any
    data: Any
    context: Any
    ops: Any


def build_fake_bpy() -> FakeBpy:
    """Return a fake bpy module that records the scene the script builds."""
    module = FakeBpy("bpy")
    module.app = types.SimpleNamespace(
        version_string="4.2.0 mock",
        version=(4, 2, 0),
        binary_path="/fake/blender",
    )
    module.data = types.SimpleNamespace(
        objects=FakeCollection("OBJECTS"),
        materials=FakeCollection("MATERIALS"),
        cameras=FakeCollection("CAMERAS"),
        lights=FakeCollection("LIGHTS"),
    )
    module.context = FakeContext(FakeScene())
    module.ops = FakeOps(module)
    module.ops.wm = types.SimpleNamespace(
        read_factory_settings=module.ops.wm_read_factory_settings,
        save_as_mainfile=module.ops.wm_save_as_mainfile,
    )
    module.ops.object = types.SimpleNamespace(
        select_all=module.ops.object_select_all,
        delete=module.ops.object_delete,
        text_add=module.ops.object_text_add,
        convert=module.ops.object_convert,
    )
    module.ops.mesh = types.SimpleNamespace(
        primitive_torus_add=module.ops.mesh_primitive_torus_add,
        primitive_cube_add=module.ops.mesh_primitive_cube_add,
        primitive_cylinder_add=module.ops.mesh_primitive_cylinder_add,
    )
    module.ops.render = types.SimpleNamespace(render=module.ops.render_render)
    module.ops.export_scene = types.SimpleNamespace(gltf=module.ops.export_scene_gltf)
    return module


def euler_matrix(alpha: float, beta: float, gamma: float) -> list[list[float]]:
    """Return the XYZ Euler rotation matrix of three angles in radians."""
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    cg, sg = math.cos(gamma), math.sin(gamma)
    rx = [[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]]
    ry = [[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]]
    rz = [[cg, -sg, 0.0], [sg, cg, 0.0], [0.0, 0.0, 1.0]]
    product = [[sum(rz[i][k] * ry[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    return [[sum(product[i][k] * rx[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


@pytest.fixture()
def fake_bpy(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a fake bpy module for the duration of one test."""
    module = build_fake_bpy()
    monkeypatch.setitem(sys.modules, "bpy", module)
    return module


def load_module() -> Any:
    """Load the in-Blender script with the fake bpy module in place."""
    spec = importlib.util.spec_from_file_location("blender_tire_render_under_test", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_poses(path: Path, *, frames: int = 2, label: str | None = "AXION") -> Path:
    """Write a pose document with the shape the render script consumes."""
    poses = [
        {
            "frame": index,
            "position": [0.1 * index, -0.2, 0.0],
            "rotation_euler_degrees": [0.0, 0.0, 12.0 * index],
            "scale": 0.6,
        }
        for index in range(frames)
    ]
    payload = {
        "schema": "void-fusion-lab.tire-wheel-poses.v1",
        "width": 672,
        "height": 384,
        "meters_per_pixel": 0.01,
        "frame_count": frames,
        "target": {
            "description": "a black tire with a yellow band",
            "tread_color": [0.035, 0.035, 0.045],
            "band_color": [0.86, 0.62, 0.04],
            "label": label,
        },
        "poses": poses,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def blender_binary(tmp_path: Path) -> Path:
    """Return a placeholder Blender executable whose digest can be recorded."""
    path = tmp_path / "blender"
    path.write_bytes(b"blender-placeholder")
    return path


def run_render(
    module: Any,
    fake: types.ModuleType,
    tmp_path: Path,
    *,
    poses: Path,
    extra: list[str] | None = None,
) -> tuple[int, Path]:
    """Run the in-Blender script and return its exit code and output directory."""
    output = tmp_path / "render"
    command = [
        "--poses",
        str(poses),
        "--out",
        str(output),
        "--samples",
        "4",
        "--blender-executable",
        str(blender_binary(tmp_path)),
    ]
    if extra is not None:
        command += extra
    return module.main(command), output


def test_the_script_imports_outside_blender_and_fails_closed(monkeypatch: pytest.MonkeyPatch):
    """The script must refuse to run anywhere the bpy module is missing."""
    monkeypatch.setitem(sys.modules, "bpy", None)
    module = load_module()
    assert module.bpy is None
    with pytest.raises(module.TireRenderError):
        module.require_bpy()


def test_parse_args_requires_the_blender_separator(monkeypatch: pytest.MonkeyPatch):
    """The script must reject a command line without the Blender separator."""
    module = load_module()
    monkeypatch.setattr(sys, "argv", ["blender", "--background", "--python", str(SCRIPT)])
    with pytest.raises(SystemExit):
        module.parse_args()


def test_aim_euler_points_an_object_at_the_origin():
    """The light aiming must map the local -Z axis onto the target direction."""
    module = load_module()
    location = (1.0, -2.0, 3.0)
    alpha, beta, gamma = module.aim_euler(location)
    matrix = euler_matrix(alpha, beta, gamma)
    pointing = [-matrix[0][2], -matrix[1][2], -matrix[2][2]]
    radius = math.sqrt(sum(value * value for value in location))
    expected = [-value / radius for value in location]
    assert pointing == pytest.approx(expected, abs=1e-9)


def test_the_render_writes_a_manifest_with_mesh_counts_and_digests(
    fake_bpy: types.ModuleType, tmp_path: Path
):
    """The manifest must record the Blender identity, the digests, and the counts."""
    module = load_module()
    poses = write_poses(tmp_path / "poses.json")
    blend = tmp_path / "tire.blend"
    glb = tmp_path / "tire.glb"
    exit_code, output = run_render(
        module,
        fake_bpy,
        tmp_path,
        poses=poses,
        extra=["--blend", str(blend), "--glb", str(glb)],
    )
    assert exit_code == 0
    manifest = json.loads((output / "render-manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == module.SCHEMA
    assert manifest["engine"] == "CYCLES"
    assert manifest["width"] == 672
    assert manifest["height"] == 384
    assert manifest["samples"] == 4
    assert manifest["frames"] == 2
    assert manifest["film_transparent"] is True
    assert manifest["color_mode"] == "RGBA"
    assert manifest["blender_version"] == "4.2.0 mock"
    assert manifest["pose_sha256"] == hashlib.sha256(poses.read_bytes()).hexdigest()
    assert manifest["script_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    counts = manifest["counts"]
    mesh_objects = [obj for obj in fake_bpy.data.objects if obj.type == "MESH"]
    assert counts["mesh_objects"] == 38
    assert counts["mesh_objects"] == len(mesh_objects)
    assert counts["mesh_vertices"] == sum(len(obj.data.vertices) for obj in mesh_objects)
    assert counts["mesh_polygons"] == sum(len(obj.data.polygons) for obj in mesh_objects)
    assert counts["material_count"] == 7
    assert counts["material_count"] == len(fake_bpy.data.materials)
    assert "tire-band" in counts["materials"]
    assert manifest["blend"]["sha256"] == hashlib.sha256(blend.read_bytes()).hexdigest()
    assert manifest["glb"]["path"] == str(glb)
    assert manifest["glb"]["sha256"] == hashlib.sha256(glb.read_bytes()).hexdigest()
    renders = sorted(output.glob("frame-*.png"))
    assert [path.name for path in renders] == ["frame-000000.png", "frame-000001.png"]
    for record in manifest["renders"]:
        assert record["sha256"] == hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()


def test_the_wheel_is_built_from_torus_tread_cylinder_spoke_and_text_geometry(
    fake_bpy: types.ModuleType, tmp_path: Path
):
    """Every required geometry family must be created with its own operator."""
    module = load_module()
    poses = write_poses(tmp_path / "poses.json")
    exit_code, _ = run_render(module, fake_bpy, tmp_path, poses=poses)
    assert exit_code == 0
    torus = fake_bpy.ops.arguments("mesh.primitive_torus_add")
    assert len(torus) == 1
    assert torus[0]["major_radius"] == pytest.approx(module.TIRE_MAJOR_RADIUS)
    assert torus[0]["minor_radius"] == pytest.approx(module.TIRE_MINOR_RADIUS)
    cubes = fake_bpy.ops.arguments("mesh.primitive_cube_add")
    assert len(cubes) == module.TREAD_BLOCKS + module.SPOKE_COUNT
    cylinders = fake_bpy.ops.arguments("mesh.primitive_cylinder_add")
    assert len(cylinders) == 3
    assert {call["radius"] for call in cylinders} == {
        module.RIM_RADIUS,
        module.DISC_RADIUS,
        module.HUB_RADIUS,
    }
    assert len(fake_bpy.ops.arguments("object.text_add")) == 1
    assert fake_bpy.ops.arguments("object.convert") == [{"target": "MESH"}]


def test_the_carcass_carries_tread_band_and_sidewall_material_regions(
    fake_bpy: types.ModuleType, tmp_path: Path
):
    """The advertising band must be a real material region of the torus mesh."""
    module = load_module()
    poses = write_poses(tmp_path / "poses.json")
    exit_code, _ = run_render(module, fake_bpy, tmp_path, poses=poses)
    assert exit_code == 0
    carcass = next(obj for obj in fake_bpy.data.objects if obj.name == "tire-carcass")
    assert len(carcass.data.materials) == 3
    indices = {polygon.material_index for polygon in carcass.data.polygons}
    assert indices == {0, 1, 2}
    for polygon in carcass.data.polygons:
        radius = module.polygon_radius(carcass.data, polygon)
        if polygon.material_index == 1:
            assert module.BAND_INNER <= radius <= module.BAND_OUTER
        elif polygon.material_index == 0:
            assert radius >= module.SHOULDER_RADIUS
        else:
            assert not (module.BAND_INNER <= radius <= module.BAND_OUTER)


def test_the_label_is_converted_to_a_mesh_and_bent_onto_the_band(
    fake_bpy: types.ModuleType, tmp_path: Path
):
    """The sidewall label must become real mesh geometry on the band."""
    module = load_module()
    poses = write_poses(tmp_path / "poses.json")
    exit_code, output = run_render(module, fake_bpy, tmp_path, poses=poses)
    assert exit_code == 0
    manifest = json.loads((output / "render-manifest.json").read_text(encoding="utf-8"))
    assert manifest["wheel"]["label"] == "AXION"
    assert manifest["wheel"]["label_record"]["converted"] is True
    assert manifest["wheel"]["label_record"]["bent"] is True
    label = next(obj for obj in fake_bpy.data.objects if obj.name == "tire-label")
    assert label.type == "MESH"
    assert [material.name for material in label.data.materials] == ["tire-label"]
    radii = [math.hypot(vertex.co[0], vertex.co[1]) for vertex in label.data.vertices]
    assert min(radii) >= module.BAND_INNER
    assert max(radii) <= module.BAND_OUTER
    depths = [vertex.co[2] for vertex in label.data.vertices]
    assert min(depths) > 0.0
    for vertex in label.data.vertices:
        radius = math.hypot(vertex.co[0], vertex.co[1])
        expected = module.sidewall_depth(radius) + module.TEXT_LIFT
        assert abs(vertex.co[2] - expected) <= module.TEXT_LIFT + 1e-9


def test_a_missing_gltf_exporter_is_recorded_instead_of_failing(
    fake_bpy: types.ModuleType, tmp_path: Path
):
    """A build without the glTF exporter must still finish and say so."""
    module = load_module()
    fake_bpy.ops.gltf_error = RuntimeError("the glTF exporter addon is disabled")
    poses = write_poses(tmp_path / "poses.json")
    exit_code, output = run_render(module, fake_bpy, tmp_path, poses=poses)
    assert exit_code == 0
    manifest = json.loads((output / "render-manifest.json").read_text(encoding="utf-8"))
    assert manifest["glb"]["path"] is None
    assert "unavailable" in manifest["glb"]["reason"]


def test_the_camera_frames_the_wheel_in_pixel_units(fake_bpy: types.ModuleType, tmp_path: Path):
    """The orthographic camera must scale with the render width."""
    module = load_module()
    poses = write_poses(tmp_path / "poses.json")
    exit_code, _ = run_render(module, fake_bpy, tmp_path, poses=poses)
    assert exit_code == 0
    camera = next(obj for obj in fake_bpy.data.objects if obj.name == "tire-ortho")
    assert camera.data.type == "ORTHO"
    assert camera.data.ortho_scale == pytest.approx(6.72)
    assert camera.data.sensor_fit == "HORIZONTAL"
    assert camera.location == (0.0, 0.0, 20.0)


def test_each_pose_drives_the_wheel_root_transform(fake_bpy: types.ModuleType, tmp_path: Path):
    """The root transform must carry the fitted position, rotation, and scale."""
    module = load_module()
    poses = write_poses(tmp_path / "poses.json")
    exit_code, _ = run_render(module, fake_bpy, tmp_path, poses=poses)
    assert exit_code == 0
    root = next(obj for obj in fake_bpy.data.objects if obj.name == "tire-root")
    assert root.location == pytest.approx((0.1, -0.2, 0.0))
    assert root.scale == pytest.approx((0.6, 0.6, 0.6))
    assert root.rotation_euler == pytest.approx((0.0, 0.0, math.radians(12.0)))
    parts = [obj for obj in fake_bpy.data.objects if obj.type == "MESH"]
    assert parts
    assert all(obj.parent is root for obj in parts)


def test_a_malformed_pose_document_fails_closed(
    fake_bpy: types.ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """A pose document without frames must fail closed with the error exit code."""
    module = load_module()
    poses = tmp_path / "poses.json"
    poses.write_text(json.dumps({"width": 672, "height": 384}), encoding="utf-8")
    exit_code, _ = run_render(module, fake_bpy, tmp_path, poses=poses)
    assert exit_code == module.EXIT_RENDER_ERROR
    assert "error:" in capsys.readouterr().err


def test_a_malformed_pose_shape_fails_closed(
    fake_bpy: types.ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """A wrong-length transform must return the render error code, not a traceback."""
    module = load_module()
    poses = write_poses(tmp_path / "poses.json")
    payload = json.loads(poses.read_text(encoding="utf-8"))
    payload["poses"][0]["position"] = [0.0, 0.0]
    poses.write_text(json.dumps(payload), encoding="utf-8")

    exit_code, _ = run_render(module, fake_bpy, tmp_path, poses=poses)

    assert exit_code == module.EXIT_RENDER_ERROR
    assert "three-element position" in capsys.readouterr().err
