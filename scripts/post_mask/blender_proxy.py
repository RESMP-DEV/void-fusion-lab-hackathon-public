"""Render the fitted capsule proxy from a contiguous pose JSON series."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, cast

try:
    import bpy  # ty: ignore[unresolved-import]
    from mathutils import Euler, Vector  # ty: ignore[unresolved-import]
except ImportError:  # Permit argparse/import checks outside Blender.
    bpy = cast(Any, None)
    Euler = cast(Any, None)
    Vector = cast(Any, None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--renderer", type=Path, required=True)
    try:
        separator = sys.argv.index("--")
    except ValueError:
        parser.error("missing '--' separator between Blender and script arguments")
    return parser.parse_args(sys.argv[separator + 1 :])


def clear_scene(samples: int) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = False
    scene.cycles.device = "GPU"


def material(name: str, color: tuple[float, float, float, float], roughness: float):
    item = bpy.data.materials.new(name)
    item.use_nodes = True
    base = item.node_tree.nodes.get("Principled BSDF")
    base.inputs["Base Color"].default_value = color
    base.inputs["Roughness"].default_value = roughness
    return item


def rotation_to(axis: Vector) -> Euler:
    unit = Vector((0.0, 0.0, 1.0))
    if axis.length < 1e-6:
        return Euler((0.0, 0.0, 0.0))
    target = axis.normalized()
    if abs(unit.dot(target)) > 0.999999:
        return Euler((0.0, 0.0, 0.0))
    return unit.rotation_difference(target).to_euler()


def add_primitive(operator, name: str, location: Vector, rotation: Euler, mat, **kwargs):
    operator(**kwargs)
    obj = bpy.context.active_object
    obj.name = name
    obj.location = location
    obj.rotation_euler = rotation
    obj.data.materials.append(mat)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def point(value: list[float], *, width: int, height: int) -> Vector:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError(f"expected a two-element point, got {value!r}")
    return Vector(
        ((value[0] - width / 2) / (width / 2), (height / 2 - value[1]) / (width / 2), 0.0)
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capsule(name: str, start: Vector, end: Vector, radius: float, mat) -> None:
    axis = end - start
    length = max(axis.length, radius * 2.02)
    center = (start + end) / 2
    rotation = rotation_to(axis)
    add_primitive(
        bpy.ops.mesh.primitive_cylinder_add,
        f"{name}-shaft",
        center,
        rotation,
        mat,
        radius=radius,
        depth=max(length - radius * 2.0, 0.001),
        vertices=32,
    )
    for suffix, joint in (("start", start), ("end", end)):
        add_primitive(
            bpy.ops.mesh.primitive_uv_sphere_add,
            f"{name}-{suffix}",
            joint,
            Euler((0.0, 0.0, 0.0)),
            mat,
            radius=radius,
            segments=32,
            ring_count=16,
        )


def configure_render(width: int, height: int, output: Path) -> None:
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.filepath = str(output)

    camera_data = bpy.data.cameras.new("front-ortho")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 2.0
    camera_data.sensor_fit = "HORIZONTAL"
    camera = bpy.data.objects.new("front-ortho", camera_data)
    camera.location = Vector((0.0, 0.0, 5.0))
    camera.rotation_euler = Euler((0.0, 0.0, 0.0))
    scene.collection.objects.link(camera)
    scene.camera = camera

    light_data = bpy.data.lights.new("key", type="AREA")
    light_data.energy = 180.0
    light_data.size = 2.5
    light = bpy.data.objects.new("key", light_data)
    light.location = Vector((-1.0, -2.0, 2.5))
    light.rotation_euler = Euler((math.radians(60), 0.0, math.radians(-20)))
    scene.collection.objects.link(light)


def build_frame(row: dict, *, width: int, height: int) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    body = material("capsule-body", (0.94, 0.94, 0.92, 1.0), 0.48)
    band = material("capsule-band", (0.06, 0.06, 0.07, 1.0), 0.35)
    accent = material("capsule-accent", (0.10, 0.14, 0.30, 1.0), 0.42)
    values = row.get("points")
    if not isinstance(values, dict):
        raise ValueError(f"pose frame {row.get('frame')!r} lacks a points object")
    missing = {
        "nose",
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    } - set(values)
    if missing:
        raise ValueError(f"pose frame {row.get('frame')!r} lacks landmarks {sorted(missing)}")
    core_missing = [
        name
        for name in ("nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip")
        if values[name] is None
    ]
    if core_missing:
        raise ValueError(
            f"pose frame {row.get('frame')!r} lacks finite core landmarks {core_missing}"
        )
    shoulders = (
        point(values["left_shoulder"], width=width, height=height),
        point(values["right_shoulder"], width=width, height=height),
    )
    hips = (
        point(values["left_hip"], width=width, height=height),
        point(values["right_hip"], width=width, height=height),
    )
    shoulder_center = (shoulders[0] + shoulders[1]) / 2
    hip_center = (hips[0] + hips[1]) / 2
    torso_axis = shoulder_center - hip_center
    shoulder_width = (shoulders[0] - shoulders[1]).length

    radii = (0.46, 0.54, 0.50, 0.42)
    fractions = (0.08, 0.38, 0.68, 0.94)
    for index, (fraction, radius_factor) in enumerate(zip(fractions, radii, strict=False)):
        center = hip_center + torso_axis * fraction
        radius = shoulder_width * radius_factor
        section = add_primitive(
            bpy.ops.mesh.primitive_uv_sphere_add,
            f"body-section-{index}",
            center,
            rotation_to(torso_axis),
            body,
            radius=radius,
            segments=48,
            ring_count=24,
        )
        section.scale = Vector((1.0, 1.0, 0.48))
        add_primitive(
            bpy.ops.mesh.primitive_torus_add,
            f"body-band-{index}",
            center + torso_axis.normalized() * radius * 0.08,
            rotation_to(torso_axis),
            band if index != 1 else accent,
            major_radius=radius * 0.95,
            minor_radius=radius * 0.13,
            major_segments=48,
            minor_segments=16,
        )

    head_center = point(values["nose"], width=width, height=height)
    head_center = head_center + torso_axis.normalized() * shoulder_width * 0.08
    add_primitive(
        bpy.ops.mesh.primitive_uv_sphere_add,
        "head",
        head_center,
        rotation_to(torso_axis),
        body,
        radius=shoulder_width * 0.38,
        segments=48,
        ring_count=24,
    )
    add_primitive(
        bpy.ops.mesh.primitive_torus_add,
        "head-band",
        head_center,
        rotation_to(torso_axis),
        accent,
        major_radius=shoulder_width * 0.34,
        minor_radius=shoulder_width * 0.045,
        major_segments=48,
        minor_segments=16,
    )

    arm_radius = shoulder_width * 0.15
    leg_radius = shoulder_width * 0.17
    for side in ("left", "right"):
        for distal, proximal in (("elbow", "shoulder"), ("wrist", "elbow")):
            start = values[f"{side}_{proximal}"]
            end = values[f"{side}_{distal}"]
            if start is None or end is None:
                continue
            capsule(
                f"{side}-{proximal}-{distal}",
                point(start, width=width, height=height),
                point(end, width=width, height=height),
                arm_radius,
                body,
            )
        for distal, proximal in (("knee", "hip"), ("ankle", "knee")):
            start = values[f"{side}_{proximal}"]
            end = values[f"{side}_{distal}"]
            if start is None or end is None:
                continue
            capsule(
                f"{side}-{proximal}-{distal}",
                point(start, width=width, height=height),
                point(end, width=width, height=height),
                leg_radius,
                body,
            )


def main() -> None:
    args = parse_args()
    payload = json.loads(args.poses.read_text())
    width = int(payload["width"])
    height = int(payload["height"])
    args.out.mkdir(parents=True, exist_ok=True)
    clear_scene(args.samples)
    for row in payload["frames"]:
        build_frame(row, width=width, height=height)
        configure_render(width, height, args.out / f"alpha-{int(row['frame']):06d}.png")
        bpy.ops.render.render(write_still=True)
    (args.out / "render-manifest.json").write_text(
        json.dumps(
            {
                "blender": bpy.app.version_string,
                "blender_executable": str(args.renderer.resolve()),
                "blender_executable_sha256": file_sha256(args.renderer),
                "engine": bpy.context.scene.render.engine,
                "samples": args.samples,
                "frames": len(payload["frames"]),
                "width": width,
                "height": height,
                "script_sha256": file_sha256(Path(__file__).resolve()),
                "pose_sha256": hashlib.sha256(args.poses.read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
