#!/usr/bin/env python3
"""Construct and render a true three-dimensional advertising tire in Blender.

The script runs inside Blender's own Python interpreter.  It builds a real
mesh tire from a torus carcass, radial tread blocks, rim, brake-disc, and hub
cylinders, spokes, and a text object that is converted to a mesh and bent onto
the sidewall band.  It renders transparent RGBA frames with Cycles, exports the
scene to a ``.blend`` file and, when the glTF exporter is available, to a
``.glb`` file, and writes a render manifest that records the Blender identity,
the file digests, and the mesh object, vertex, and material counts.

No third-party asset, checkpoint, credential, or network call is used, and the
script reads nothing outside the paths handed to it on the command line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

try:
    import bpy  # ty: ignore[unresolved-import]
except ImportError:  # Permit import and protocol checks outside Blender.
    bpy = cast(Any, None)

SCHEMA = "void-fusion-lab.blender-tire-render.v1"
EXIT_RENDER_ERROR = 3

# The wheel is modelled with an outer radius of exactly 1.0 in local units, so
# a pose scale of one metre renders a one-metre tyre.
TIRE_MAJOR_RADIUS = 0.78
TIRE_MINOR_RADIUS = 0.22
TREAD_RADIUS = 0.975
TREAD_THICKNESS = 0.05
TREAD_WIDTH = 0.16
TREAD_DEPTH = 0.34
TREAD_BLOCKS = 28
BAND_RADIUS = 0.66
BAND_INNER = 0.60
BAND_OUTER = 0.72
SHOULDER_RADIUS = 0.86
RIM_RADIUS = 0.56
RIM_DEPTH = 0.20
RIM_OFFSET = -0.10
DISC_RADIUS = 0.42
DISC_DEPTH = 0.04
DISC_OFFSET = -0.02
HUB_RADIUS = 0.15
HUB_DEPTH = 0.30
HUB_OFFSET = 0.02
SPOKE_COUNT = 5
SPOKE_INNER = 0.15
SPOKE_OUTER = 0.56
SPOKE_WIDTH = 0.10
SPOKE_THICKNESS = 0.14
SPOKE_OFFSET = 0.06
TEXT_ARC = 0.75
TEXT_HEIGHT = 0.10
TEXT_LIFT = 0.008
TEXT_SIZE = 0.13
TEXT_CENTER_ANGLE = math.pi / 2.0
RIM_COLOR = (0.55, 0.56, 0.58)
DISC_COLOR = (0.20, 0.21, 0.23)
HUB_COLOR = (0.36, 0.37, 0.40)


class TireRenderError(RuntimeError):
    """A pose, scene, export, or render operation for the tyre lane failed."""


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Return the command-line contract of the in-Blender render script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--blender-executable", type=Path, required=True)
    parser.add_argument("--blend", type=Path, default=None)
    parser.add_argument("--glb", type=Path, default=None)
    if argv is None:
        argv = sys.argv
        try:
            separator = argv.index("--")
        except ValueError:
            parser.error("missing '--' separator between Blender and script arguments")
        argv = argv[separator + 1 :]
    return parser.parse_args(argv)


def require_bpy() -> Any:
    """Fail closed when the script is not running inside Blender."""
    if bpy is None:
        raise TireRenderError("the bpy module is unavailable; this script must run inside Blender")
    return bpy


def aim_euler(location: tuple[float, float, float]) -> tuple[float, float, float]:
    """Return the XYZ Euler angles that aim a -Z facing object at the origin.

    The camera and the lights in this scene all point down their local -Z
    axis.  The returned triple reproduces the rotation ``Rz(phi - pi/2) @
    Rx(epsilon - pi/2)``, which maps ``(0, 0, -1)`` onto the unit vector from
    ``location`` to the origin, using Blender's XYZ Euler order.
    """
    x, y, z = location
    radius = math.sqrt(x * x + y * y + z * z)
    if radius < 1e-9:
        return (0.0, 0.0, 0.0)
    azimuth = math.atan2(y, x)
    elevation = math.asin(z / radius)
    return (elevation - math.pi / 2.0, 0.0, azimuth - math.pi / 2.0)


def sidewall_depth(radius: float) -> float:
    """Return the front surface depth of the tyre torus at one radial offset."""
    offset = radius - TIRE_MAJOR_RADIUS
    squared = TIRE_MINOR_RADIUS * TIRE_MINOR_RADIUS - offset * offset
    if squared <= 0.0:
        return 0.0
    return math.sqrt(squared)


def luminance(color: tuple[float, float, float]) -> float:
    """Return the perceptual luminance of one linear color."""
    return 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]


def color_triple(values: Any, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    """Return one linear color triple from a JSON value."""
    try:
        sequence = [float(value) for value in values]
    except (TypeError, ValueError):
        return fallback
    if len(sequence) != 3:
        return fallback
    return (sequence[0], sequence[1], sequence[2])


def make_material(
    name: str,
    color: tuple[float, float, float],
    *,
    roughness: float = 0.6,
    metallic: float = 0.0,
) -> Any:
    """Create one Principled material with a linear base color."""
    api = require_bpy()
    item = api.data.materials.new(name)
    item.use_nodes = True
    shader = item.node_tree.nodes.get("Principled BSDF")
    if shader is not None:
        shader.inputs["Base Color"].default_value = (color[0], color[1], color[2], 1.0)
        shader.inputs["Roughness"].default_value = roughness
        shader.inputs["Metallic"].default_value = metallic
    return item


def apply_material(obj: Any, material: Any) -> None:
    """Attach one material to a mesh object."""
    obj.data.materials.append(material)


def set_transform(obj: Any, location, rotation, scale=(1.0, 1.0, 1.0)) -> None:
    """Place one object with a location, an XYZ Euler rotation, and a scale."""
    obj.location = tuple(float(value) for value in location)
    obj.rotation_euler = tuple(float(value) for value in rotation)
    obj.scale = tuple(float(value) for value in scale)


def polygon_radius(mesh: Any, polygon: Any) -> float:
    """Return the radial distance of one polygon from the wheel axis."""
    indices = list(polygon.vertices)
    if not indices:
        return 0.0
    total_x = 0.0
    total_y = 0.0
    for index in indices:
        co = mesh.vertices[index].co
        total_x += float(co[0])
        total_y += float(co[1])
    count = float(len(indices))
    return math.hypot(total_x / count, total_y / count)


def assign_torus_materials(obj: Any, tread: Any, band: Any, sidewall: Any) -> None:
    """Split the torus into tread, advertising band, and sidewall faces.

    The band is a real material region on the torus mesh: the polygons whose
    radial distance falls inside the band window receive the band material, so
    the band is painted geometry rather than a projected decal.  The material
    order is tread, band, sidewall, which fixes the polygon material indices.
    """
    for material in (tread, band, sidewall):
        apply_material(obj, material)
    mesh = obj.data
    for polygon in mesh.polygons:
        radius = polygon_radius(mesh, polygon)
        if BAND_INNER <= radius <= BAND_OUTER:
            polygon.material_index = 1
        elif radius >= SHOULDER_RADIUS:
            polygon.material_index = 0
        else:
            polygon.material_index = 2


def add_torus(name: str) -> Any:
    """Add the tyre carcass torus."""
    api = require_bpy()
    api.ops.mesh.primitive_torus_add(
        major_radius=TIRE_MAJOR_RADIUS,
        minor_radius=TIRE_MINOR_RADIUS,
        major_segments=64,
        minor_segments=24,
    )
    obj = api.context.active_object
    obj.name = name
    set_transform(obj, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def add_tread_blocks(name: str, material: Any) -> list[Any]:
    """Add the radial tread blocks that break the tyre's outer surface."""
    api = require_bpy()
    blocks: list[Any] = []
    for index in range(TREAD_BLOCKS):
        angle = 2.0 * math.pi * index / TREAD_BLOCKS
        api.ops.mesh.primitive_cube_add(size=1.0)
        obj = api.context.active_object
        obj.name = f"{name}-{index:02d}"
        set_transform(
            obj,
            (TREAD_RADIUS * math.cos(angle), TREAD_RADIUS * math.sin(angle), 0.0),
            (0.0, 0.0, angle),
            (TREAD_THICKNESS, TREAD_WIDTH, TREAD_DEPTH),
        )
        apply_material(obj, material)
        blocks.append(obj)
    return blocks


def add_cylinder(
    name: str,
    material: Any,
    *,
    radius: float,
    depth: float,
    vertices: int,
    offset: float,
    smooth: bool = True,
) -> Any:
    """Add one cylinder whose axis is the wheel axis."""
    api = require_bpy()
    api.ops.mesh.primitive_cylinder_add(radius=radius, depth=depth, vertices=vertices)
    obj = api.context.active_object
    obj.name = name
    set_transform(obj, (0.0, 0.0, offset), (0.0, 0.0, 0.0))
    if smooth:
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
    apply_material(obj, material)
    return obj


def add_spokes(name: str, material: Any) -> list[Any]:
    """Add the radial spokes that join the hub to the rim."""
    api = require_bpy()
    length = SPOKE_OUTER - SPOKE_INNER
    middle = (SPOKE_OUTER + SPOKE_INNER) / 2.0
    spokes: list[Any] = []
    for index in range(SPOKE_COUNT):
        angle = 2.0 * math.pi * index / SPOKE_COUNT
        api.ops.mesh.primitive_cube_add(size=1.0)
        obj = api.context.active_object
        obj.name = f"{name}-{index:02d}"
        set_transform(
            obj,
            (middle * math.cos(angle), middle * math.sin(angle), SPOKE_OFFSET),
            (0.0, 0.0, angle),
            (length, SPOKE_WIDTH, SPOKE_THICKNESS),
        )
        apply_material(obj, material)
        spokes.append(obj)
    return spokes


def bend_label(obj: Any) -> dict[str, Any]:
    """Bend the converted label mesh onto the advertising band of the sidewall.

    The flat text mesh is mapped onto the front surface of the torus: the
    horizontal axis becomes an arc along the band and the vertical axis
    becomes a radial offset.  The mapping is arc-length preserving, so the
    letters keep their shape while following the sidewall curvature.
    """
    mesh = obj.data
    vertices = list(mesh.vertices)
    if not vertices:
        return {"bent": False, "reason": "the converted label mesh has no vertices"}
    xs = [float(vertex.co[0]) for vertex in vertices]
    ys = [float(vertex.co[1]) for vertex in vertices]
    extent_x = max(xs) - min(xs)
    extent_y = max(ys) - min(ys)
    if extent_x <= 0.0 or extent_y <= 0.0:
        return {"bent": False, "reason": "the converted label mesh is degenerate"}
    scale_x = TEXT_ARC / extent_x
    scale_y = min(1.0, TEXT_HEIGHT / extent_y)
    for vertex in vertices:
        local_x = float(vertex.co[0]) * scale_x
        local_y = float(vertex.co[1]) * scale_y
        depth = float(vertex.co[2])
        radius = BAND_RADIUS + local_y
        angle = TEXT_CENTER_ANGLE - local_x / BAND_RADIUS
        vertex.co = (
            radius * math.cos(angle),
            radius * math.sin(angle),
            sidewall_depth(radius) + TEXT_LIFT + depth,
        )
    return {
        "bent": True,
        "extent_x": extent_x,
        "extent_y": extent_y,
        "arc": TEXT_ARC,
        "band_radius": BAND_RADIUS,
    }


def convert_to_mesh(obj: Any) -> bool:
    """Convert one text object into a mesh and report whether it succeeded."""
    api = require_bpy()
    try:
        api.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        view_layer = getattr(api.context, "view_layer", None)
        if view_layer is not None:
            view_layer.objects.active = obj
        api.ops.object.convert(target="MESH")
    except Exception:  # noqa: BLE001 - the operator is unavailable in some builds.
        return False
    return getattr(api.context.active_object, "type", "") == "MESH"


def add_label(label: str, material: Any) -> tuple[Any, dict[str, Any]]:
    """Add the advertising label as a text object converted to a mesh."""
    api = require_bpy()
    api.ops.object.text_add(location=(0.0, 0.0, 0.0))
    obj = api.context.active_object
    obj.name = "tire-label"
    data = obj.data
    data.body = label
    data.align_x = "CENTER"
    data.align_y = "CENTER"
    data.size = TEXT_SIZE
    data.extrude = TEXT_LIFT
    apply_material(obj, material)
    converted = convert_to_mesh(obj)
    record: dict[str, Any] = {"converted": converted}
    if converted:
        active = api.context.active_object
        if active is not None:
            obj = active
        apply_material(obj, material)
        record.update(bend_label(obj))
    return obj, record


def configure_scene(width: int, height: int, *, samples: int, device: str) -> None:
    """Configure Cycles for a transparent RGBA render at the requested size."""
    api = require_bpy()
    scene = api.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.device = device
    scene.cycles.use_denoising = False
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    try:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
    except Exception:  # noqa: BLE001 - the look list depends on the colour config.
        pass


def configure_camera(width: int, height: int, *, meters_per_pixel: float) -> None:
    """Add the orthographic camera that frames the wheel in pixel units."""
    api = require_bpy()
    scene = api.context.scene
    camera_data = api.data.cameras.new("tire-ortho")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = float(width) * meters_per_pixel

    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.clip_start = 0.05
    camera_data.clip_end = 200.0
    camera = api.data.objects.new("tire-ortho", camera_data)
    camera.location = (0.0, 0.0, 20.0)
    camera.rotation_euler = (0.0, 0.0, 0.0)
    scene.collection.objects.link(camera)
    scene.camera = camera


def validate_pose(pose: Any) -> None:
    """Fail closed on one malformed wheel-pose record before Blender indexing."""
    if not isinstance(pose, dict):
        raise TireRenderError("each pose must be a JSON object")
    position = pose.get("position")
    rotation = pose.get("rotation_euler_degrees")
    if not isinstance(position, (list, tuple)) or len(position) != 3:
        raise TireRenderError("each pose needs a three-element position")
    if not isinstance(rotation, (list, tuple)) or len(rotation) != 3:
        raise TireRenderError("each pose needs a three-element rotation_euler_degrees")
    for label, values in (("position", position), ("rotation_euler_degrees", rotation)):
        for value in values:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TireRenderError(f"pose {label} values must be numbers")
            if not math.isfinite(float(value)):
                raise TireRenderError(f"pose {label} values must be finite")
    scale = pose.get("scale")
    if (
        not isinstance(scale, (int, float))
        or isinstance(scale, bool)
        or not math.isfinite(float(scale))
        or float(scale) <= 0.0
    ):
        raise TireRenderError("each pose needs a positive finite scale")
    frame = pose.get("frame")
    if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
        raise TireRenderError("each pose needs a nonnegative integer frame")


def add_area_light(
    name: str, location: tuple[float, float, float], *, energy: float, size: float
) -> Any:
    """Add one area light that aims at the wheel centre."""
    api = require_bpy()
    scene = api.context.scene
    data = api.data.lights.new(name, type="AREA")
    data.energy = energy
    data.size = size
    light = api.data.objects.new(name, data)
    light.location = (float(location[0]), float(location[1]), float(location[2]))
    light.rotation_euler = aim_euler(light.location)
    scene.collection.objects.link(light)
    return light


def configure_lights() -> None:
    """Add the three-point light rig that shapes the tyre."""
    add_area_light("tire-key", (-1.6, -2.4, 3.2), energy=2600.0, size=3.0)
    add_area_light("tire-fill", (2.2, 1.4, 2.2), energy=900.0, size=4.0)
    add_area_light("tire-rim", (0.4, 2.8, 1.6), energy=1400.0, size=2.5)


def build_wheel(target: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Build the wheel geometry and return the root object and its record."""
    api = require_bpy()
    api.ops.wm.read_factory_settings(use_empty=True)
    scene = api.context.scene
    tread_color = color_triple(target.get("tread_color"), (0.035, 0.035, 0.045))
    band_color = color_triple(target.get("band_color"), (0.86, 0.62, 0.04))
    sidewall_color = (
        min(1.0, tread_color[0] * 0.85 + 0.02),
        min(1.0, tread_color[1] * 0.85 + 0.02),
        min(1.0, tread_color[2] * 0.85 + 0.02),
    )
    label_color = (0.92, 0.92, 0.92) if luminance(band_color) < 0.45 else (0.04, 0.04, 0.05)
    tread_material = make_material("tire-tread", tread_color, roughness=0.85)
    sidewall_material = make_material("tire-sidewall", sidewall_color, roughness=0.72)
    band_material = make_material("tire-band", band_color, roughness=0.45)
    rim_material = make_material("wheel-rim", RIM_COLOR, roughness=0.28, metallic=0.9)
    disc_material = make_material("wheel-disc", DISC_COLOR, roughness=0.45, metallic=0.7)
    hub_material = make_material("wheel-hub", HUB_COLOR, roughness=0.35, metallic=0.8)
    label_material = make_material("tire-label", label_color, roughness=0.5)
    root = api.data.objects.new("tire-root", None)
    root.empty_display_size = 0.25
    scene.collection.objects.link(root)
    carcass = add_torus("tire-carcass")
    assign_torus_materials(carcass, tread_material, band_material, sidewall_material)
    tread = add_tread_blocks("tire-tread-block", tread_material)
    rim = add_cylinder(
        "wheel-rim",
        rim_material,
        radius=RIM_RADIUS,
        depth=RIM_DEPTH,
        vertices=48,
        offset=RIM_OFFSET,
    )
    disc = add_cylinder(
        "wheel-disc",
        disc_material,
        radius=DISC_RADIUS,
        depth=DISC_DEPTH,
        vertices=48,
        offset=DISC_OFFSET,
    )
    hub = add_cylinder(
        "wheel-hub",
        hub_material,
        radius=HUB_RADIUS,
        depth=HUB_DEPTH,
        vertices=32,
        offset=HUB_OFFSET,
    )
    spokes = add_spokes("wheel-spoke", rim_material)
    label = str(target.get("label") or "").strip()
    label_object = None
    label_record: dict[str, Any] = {"converted": False, "bent": False}
    if label:
        label_object, label_record = add_label(label, label_material)
    parts = [carcass, rim, disc, hub, *tread, *spokes]
    if label_object is not None:
        parts.append(label_object)
    for part in parts:
        part.parent = root
    counts = {
        "torus": 1,
        "tread": len(tread),
        "cylinder": 3,
        "spoke": len(spokes),
        "text": 1 if label_object is not None else 0,
    }
    record = {
        "label": label or None,
        "label_record": label_record,
        "parts": counts,
        "colors": {
            "tread": list(tread_color),
            "sidewall": list(sidewall_color),
            "band": list(band_color),
            "label": list(label_color),
        },
    }
    return root, record


def scene_mesh_counts() -> dict[str, Any]:
    """Return the mesh object, vertex, edge, polygon, and material counts."""
    api = require_bpy()
    objects = list(api.data.objects)
    mesh_objects = [obj for obj in objects if getattr(obj, "type", "") == "MESH"]
    vertices = 0
    edges = 0
    polygons = 0
    for obj in mesh_objects:
        mesh = obj.data
        vertices += len(mesh.vertices)
        edges += len(mesh.edges)
        polygons += len(mesh.polygons)
    materials = list(api.data.materials)
    return {
        "mesh_objects": len(mesh_objects),
        "mesh_vertices": vertices,
        "mesh_edges": edges,
        "mesh_polygons": polygons,
        "material_count": len(materials),
        "materials": [item.name for item in materials],
        "object_types": {
            name: sum(1 for obj in objects if getattr(obj, "type", "") == name)
            for name in sorted({getattr(obj, "type", "UNKNOWN") for obj in objects})
        },
    }


def export_blend(path: Path) -> dict[str, Any]:
    """Save the scene as a .blend file and record its digest."""
    api = require_bpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    api.ops.wm.save_as_mainfile(filepath=str(path), check_existing=False)
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def export_glb(path: Path) -> dict[str, Any]:
    """Export the scene as a .glb file when the glTF exporter is available."""
    api = require_bpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        api.ops.export_scene.gltf(filepath=str(path), export_format="GLB")
    except Exception as error:  # noqa: BLE001 - the exporter is optional.
        return {"path": None, "reason": f"the glTF exporter is unavailable: {error}"}
    if not path.is_file():
        return {"path": None, "reason": "the glTF exporter wrote no file"}
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Render every pose and write the render manifest."""
    api = require_bpy()
    if args.samples < 1:
        raise TireRenderError("--samples must be a positive integer")
    payload = json.loads(args.poses.read_text(encoding="utf-8"))
    width = int(payload["width"])
    height = int(payload["height"])
    meters_per_pixel = float(payload.get("meters_per_pixel", 0.01))
    poses = payload["poses"]
    if not poses:
        raise TireRenderError("the pose file holds no frames")
    args.out.mkdir(parents=True, exist_ok=True)
    root, wheel = build_wheel(payload.get("target", {}))
    configure_scene(width, height, samples=args.samples, device=args.device)
    configure_camera(width, height, meters_per_pixel=meters_per_pixel)
    configure_lights()
    scene = api.context.scene
    renders: list[dict[str, Any]] = []
    for pose in poses:
        validate_pose(pose)
        position = pose["position"]
        rotation = pose["rotation_euler_degrees"]
        scale = float(pose["scale"])
        root.location = (float(position[0]), float(position[1]), float(position[2]))
        root.rotation_euler = tuple(math.radians(float(value)) for value in rotation)
        root.scale = (scale, scale, scale)
        index = int(pose["frame"])
        output = args.out / f"frame-{index:06d}.png"
        scene.render.filepath = str(output)
        api.ops.render.render(write_still=True)
        if not output.is_file():
            raise TireRenderError(f"the render produced no file for frame {index}")
        renders.append(
            {
                "frame": index,
                "path": str(output),
                "sha256": sha256(output),
                "bytes": output.stat().st_size,
            }
        )
    blend_path = args.blend if args.blend is not None else args.out / "tire.blend"
    glb_path = args.glb if args.glb is not None else args.out / "tire.glb"
    blend = export_blend(blend_path)
    glb = export_glb(glb_path)
    manifest = {
        "schema": SCHEMA,
        "recorded_at": datetime.now(UTC).isoformat(),
        "blender_version": api.app.version_string,
        "blender_version_tuple": list(api.app.version),
        "blender_binary": str(getattr(api.app, "binary_path", "")),
        "blender_executable": str(args.blender_executable.resolve()),
        "blender_executable_sha256": sha256(args.blender_executable),
        "script_sha256": sha256(Path(__file__).resolve()),
        "pose_sha256": sha256(args.poses),
        "engine": scene.render.engine,
        "device": args.device,
        "samples": args.samples,
        "width": width,
        "height": height,
        "frames": len(poses),
        "film_transparent": bool(scene.render.film_transparent),
        "color_mode": scene.render.image_settings.color_mode,
        "view_transform": scene.view_settings.view_transform,
        "camera": {
            "type": "ORTHO",
            "ortho_scale": float(width) * meters_per_pixel,
            "meters_per_pixel": meters_per_pixel,
        },
        "wheel": wheel,
        "profile": {
            "major_radius": TIRE_MAJOR_RADIUS,
            "minor_radius": TIRE_MINOR_RADIUS,
            "tread_half_depth": TREAD_DEPTH / 2.0,
            "outer_radius": TIRE_MAJOR_RADIUS + TIRE_MINOR_RADIUS,
        },
        "counts": scene_mesh_counts(),
        "renders": renders,
        "blend": blend,
        "glb": glb,
        "non_claims": [
            "The tyre is a procedural Blender mesh, not a scanned or measured "
            "reconstruction of a real tyre.",
            "The label is bent procedural Blender text from the bounded "
            "target; it is not an official brand asset or typographic match.",
            "The render is a proxy fitted to a two-dimensional mask ellipse; it "
            "is not a photometric match of the filmed wheel.",
            "The glTF export is written only when the bundled exporter is "
            "available in the running Blender build.",
        ],
    }
    (args.out / "render-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Run the in-Blender render from the command line."""
    args = parse_args(argv)
    try:
        manifest = run(args)
    except (
        TireRenderError,
        OSError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_RENDER_ERROR
    counts = manifest["counts"]
    print(
        "blender tyre render: "
        f"{manifest['frames']} frames at {manifest['width']}x{manifest['height']}, "
        f"{counts['mesh_objects']} mesh objects, {counts['mesh_vertices']} vertices, "
        f"{counts['material_count']} materials"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
