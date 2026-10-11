"""Build the oak at one age from scratch and render it with Cycles (headless).

    blender -b --factory-startup --python blender/render_age.py -- \
        --npz data/oak195.npz --textures textures --age 30 --out out/oak_030.png

One fixed camera for every age: framing comes from the FULL-age tree, so frames line up
pixel-for-pixel and the app can lay the full-age silhouette behind any age.
Transparent film + shadow catcher: only the tree and its contact shadow are in the PNG — the app
draws the background (dark/light theme).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import oak_growth as og  # noqa: E402
import oak_mesh as om  # noqa: E402

MINT = (0.306, 0.933, 0.690)  # #4EEEB0, brand accent


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=Path, required=True)
    p.add_argument("--textures", type=Path, required=True)
    p.add_argument("--age", type=float, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--percent", type=int, default=100)
    p.add_argument("--width", type=int, default=1080)
    p.add_argument("--height", type=int, default=1440)
    p.add_argument("--leaf-scale", type=float, default=1.35)
    p.add_argument("--trunk-boost", type=float, default=1.5)
    p.add_argument("--sun", type=float, default=4.5)
    p.add_argument("--sun-from", type=float, nargs=3, default=(-40.0, 12.0, 30.0),
                   help="sun position (it shines towards the tree); behind-left makes leaves glow")
    p.add_argument("--rim", type=float, default=110000.0, help="mint back light power, W")
    p.add_argument("--fill", type=float, default=12000.0)
    return p.parse_args(argv)


# ---------------------------------------------------------------- mesh / material helpers

def make_mesh(name: str, data: om.MeshData, value_attr: str, material) -> bpy.types.Object:
    me = bpy.data.meshes.new(name)
    me.from_pydata(data.verts.tolist(), [], data.quads.tolist())
    uv = me.uv_layers.new(name="UVMap")
    uv.data.foreach_set("uv", data.loop_uv.astype(np.float32).ravel())
    attr = me.attributes.new(name=value_attr, type="FLOAT", domain="POINT")
    attr.data.foreach_set("value", data.point_value.astype(np.float32))
    me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), dtype=bool))
    me.validate()
    me.update()
    me.materials.append(material)
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def image_node(nodes, path: Path, non_color: bool):
    node = nodes.new("ShaderNodeTexImage")
    node.image = bpy.data.images.load(str(path))
    if non_color:
        node.image.colorspace_settings.name = "Non-Color"
    return node


def mix_rgba(nodes):
    node = nodes.new("ShaderNodeMix")
    node.data_type = "RGBA"
    a = next(s for s in node.inputs if s.name == "A" and s.type == "RGBA")
    b = next(s for s in node.inputs if s.name == "B" and s.type == "RGBA")
    out = next(s for s in node.outputs if s.type == "RGBA")
    return node, node.inputs["Factor"], a, b, out


def bark_material(tex: Path):
    mat = bpy.data.materials.new("Bark")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    d = tex / "bark"
    diff = image_node(nodes, d / "jolcham_oak_bark_01_diff_2k.jpg", False)
    rough = image_node(nodes, d / "jolcham_oak_bark_01_rough_2k.jpg", True)
    nor = image_node(nodes, d / "jolcham_oak_bark_01_nor_gl_2k.jpg", True)
    disp = image_node(nodes, d / "jolcham_oak_bark_01_disp_2k.jpg", True)

    # Young, thin wood is smooth and grey-olive; the fissured mossy bark belongs to old thick wood.
    radius = nodes.new("ShaderNodeAttribute")
    radius.attribute_name = "radius"
    age_of_wood = nodes.new("ShaderNodeMapRange")
    age_of_wood.inputs["From Min"].default_value = 0.012
    age_of_wood.inputs["From Max"].default_value = 0.09
    links.new(radius.outputs["Fac"], age_of_wood.inputs["Value"])
    _, fac, a, b, out = mix_rgba(nodes)
    a.default_value = (0.055, 0.047, 0.038, 1.0)
    links.new(age_of_wood.outputs["Result"], fac)
    # The scan is pale and mossy-green; keep its light/dark pattern, recolour it to oak grey-brown.
    luminance = nodes.new("ShaderNodeRGBToBW")
    oak_tone = nodes.new("ShaderNodeValToRGB")
    oak_tone.color_ramp.elements[0].position = 0.15
    oak_tone.color_ramp.elements[0].color = (0.018, 0.014, 0.011, 1.0)
    oak_tone.color_ramp.elements[1].position = 0.75
    oak_tone.color_ramp.elements[1].color = (0.15, 0.115, 0.085, 1.0)
    links.new(diff.outputs["Color"], luminance.inputs["Color"])
    links.new(luminance.outputs["Val"], oak_tone.inputs["Fac"])
    links.new(oak_tone.outputs["Color"], b)
    links.new(out, bsdf.inputs["Base Color"])
    links.new(rough.outputs["Color"], bsdf.inputs["Roughness"])

    normal_map = nodes.new("ShaderNodeNormalMap")
    links.new(nor.outputs["Color"], normal_map.inputs["Color"])
    links.new(age_of_wood.outputs["Result"], normal_map.inputs["Strength"])
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35
    bump.inputs["Distance"].default_value = 0.02
    links.new(disp.outputs["Color"], bump.inputs["Height"])
    links.new(normal_map.outputs["Normal"], bump.inputs["Normal"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def leaf_material(tex: Path):
    mat = bpy.data.materials.new("OakLeaf")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    output = nodes["Material Output"]
    d = tex / "leaf"
    pick = lambda key: next(p for p in d.iterdir() if key in p.name and p.suffix.lower() in (".jpg", ".png"))
    padded = d / "LeafColorPadded.png"  # written by tools/leaf_atlas.py (no white alpha fringes)
    color = image_node(nodes, padded if padded.exists() else pick("_Color"), False)
    opacity = image_node(nodes, pick("_Opacity"), True)
    normal = image_node(nodes, pick("_NormalGL"), True)
    rough = image_node(nodes, pick("_Roughness"), True)

    # The atlas leaves are pale spring green; push them to a deep summer oak green with a little
    # per-leaf variation so the crown doesn't look cloned.
    rand = nodes.new("ShaderNodeAttribute")
    rand.attribute_name = "leaf_rand"
    hue = nodes.new("ShaderNodeMapRange")
    hue.inputs["To Min"].default_value = 0.535
    hue.inputs["To Max"].default_value = 0.575
    value = nodes.new("ShaderNodeMapRange")
    value.inputs["To Min"].default_value = 0.17
    value.inputs["To Max"].default_value = 0.32
    links.new(rand.outputs["Fac"], hue.inputs["Value"])
    links.new(rand.outputs["Fac"], value.inputs["Value"])
    hsv = nodes.new("ShaderNodeHueSaturation")
    hsv.inputs["Saturation"].default_value = 1.25
    links.new(hue.outputs["Result"], hsv.inputs["Hue"])
    links.new(value.outputs["Result"], hsv.inputs["Value"])
    links.new(color.outputs["Color"], hsv.inputs["Color"])
    links.new(hsv.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(rough.outputs["Color"], bsdf.inputs["Roughness"])
    for name in ("Specular IOR Level", "Specular"):  # waxy, not glossy: no white glints from the rim
        if name in bsdf.inputs:
            bsdf.inputs[name].default_value = 0.2
            break
    nmap = nodes.new("ShaderNodeNormalMap")
    nmap.inputs["Strength"].default_value = 0.6
    links.new(normal.outputs["Color"], nmap.inputs["Color"])
    links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

    translucent = nodes.new("ShaderNodeBsdfTranslucent")
    links.new(hsv.outputs["Color"], translucent.inputs["Color"])
    lit = nodes.new("ShaderNodeMixShader")
    lit.inputs["Fac"].default_value = 0.4
    links.new(bsdf.outputs["BSDF"], lit.inputs[1])
    links.new(translucent.outputs["BSDF"], lit.inputs[2])
    cutout = nodes.new("ShaderNodeMixShader")
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    links.new(opacity.outputs["Color"], cutout.inputs["Fac"])
    links.new(transparent.outputs["BSDF"], cutout.inputs[1])
    links.new(lit.outputs["Shader"], cutout.inputs[2])
    links.new(cutout.outputs["Shader"], output.inputs["Surface"])
    return mat


# ---------------------------------------------------------------- scene

def point_light(name, kind, energy, colour, location, target, size=None):
    light = bpy.data.lights.new(name, kind)
    light.energy = energy
    light.color = colour
    if size is not None:
        light.size = size
    obj = bpy.data.objects.new(name, light)
    obj.location = location
    direction = Vector(target) - Vector(location)
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.collection.objects.link(obj)
    return obj


def frame_camera(scene, full_points: np.ndarray, width: int, height: int):
    """Straight verticals (no tilt): horizontal camera + lens shift, framed on the full-age tree.

    Framing is solved in projection, so crown parts nearer the camera (which look bigger) still fit.
    """
    lens, sensor = 50.0, 36.0
    top, bottom, side = 0.05, 0.10, 0.06
    eye = 1.6
    aspect = width / height
    pts = full_points[np.random.default_rng(0).choice(len(full_points), min(len(full_points), 200_000), replace=False)]
    x_c = float(np.median(pts[:, 0]))
    distance = 40.0
    for _ in range(6):
        k = distance / (distance + pts[:, 1])           # perspective scale towards the y=0 plane
        x = x_c + (pts[:, 0] - x_c) * k
        z = eye + (pts[:, 2] - eye) * k
        xmin, xmax = np.percentile(x, [0.02, 99.98])
        zmax = float(np.percentile(z, 99.98))
        x_c = float((xmin + xmax) / 2)
        frame_h = max(zmax / (1 - top - bottom), (xmax - xmin) / (1 - 2 * side) / aspect)
        distance = frame_h / (sensor / lens)
    centre_z = -bottom * frame_h + frame_h / 2

    cam = bpy.data.cameras.new("Camera")
    cam.lens = lens
    cam.sensor_fit = "VERTICAL"
    cam.sensor_height = sensor
    cam.shift_y = (centre_z - eye) / frame_h
    cam.clip_end = distance * 4
    obj = bpy.data.objects.new("Camera", cam)
    obj.location = (x_c, -distance, eye)
    obj.rotation_euler = (math.pi / 2, 0.0, 0.0)
    scene.collection.objects.link(obj)
    scene.camera = obj
    return {"frame_height_m": float(frame_h), "distance_m": float(distance), "x_centre_m": x_c}


def main() -> None:
    args = parse_args()
    t0 = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    cards = [om.LeafCard(**c) for c in json.loads((args.textures / "leaf_cards.json").read_text())]
    skel = og.load_npz(args.npz)
    pre = og.precompute(skel)
    look = dict(leaf_scale=args.leaf_scale, trunk_boost=args.trunk_boost)
    full_tubes, full_leaves, _ = om.build_age(skel, pre, og.FULL_AGE, cards, **look)
    tubes, leaves, tree = om.build_age(skel, pre, args.age, cards, **look)
    t_geo = time.time()

    make_mesh("Oak", tubes, "radius", bark_material(args.textures))
    if len(leaves.quads):
        make_mesh("Leaves", leaves, "leaf_rand", leaf_material(args.textures))

    ground = bpy.data.meshes.new("Ground")
    ground.from_pydata([(-200, -200, 0), (200, -200, 0), (200, 200, 0), (-200, 200, 0)], [], [(0, 1, 2, 3)])
    soil = bpy.data.materials.new("Soil")  # a white catcher bounces sun + rim up onto the bark
    soil.use_nodes = True
    soil.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.03, 0.035, 0.025, 1.0)
    soil.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 1.0
    ground.materials.append(soil)
    ground_obj = bpy.data.objects.new("Ground", ground)
    ground_obj.is_shadow_catcher = True
    scene.collection.objects.link(ground_obj)

    framing = frame_camera(scene, np.vstack([full_tubes.verts, full_leaves.verts]), args.width, args.height)
    height = float(full_tubes.verts[:, 2].max())
    point_light("Key", "SUN", args.sun, (1.0, 0.93, 0.82), tuple(args.sun_from), (0.0, 0.0, 0.0))
    scene.objects["Key"].data.angle = math.radians(3.0)
    point_light("Rim", "AREA", args.rim, MINT, (0.0, 24.0, height * 0.75), (0.0, -10.0, height * 0.5), size=14.0)
    point_light("Fill", "AREA", args.fill, (0.85, 0.95, 1.0), (15.0, -35.0, 12.0), (0.0, 0.0, height * 0.4), size=30.0)

    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.006, 0.016, 0.013, 1.0)
    scene.world = world

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = args.samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.max_bounces = 6
    scene.cycles.transparent_max_bounces = 16
    scene.render.film_transparent = True
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = args.percent
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "AgX"
    for look in ("AgX - Medium High Contrast", "Medium High Contrast"):
        try:
            scene.view_settings.look = look
            break
        except TypeError:
            continue

    args.out.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(args.out.resolve())
    bpy.ops.render.render(write_still=True)
    t_end = time.time()

    meta = {
        "age": args.age, "samples": args.samples, "percent": args.percent,
        "tree_scale": tree.scale, "tube_faces": int(len(tubes.quads)), "leaves": int(len(leaves.quads)),
        "geometry_s": round(t_geo - t0, 1), "render_s": round(t_end - t_geo, 1), **framing,
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print("RENDER_DONE", json.dumps(meta))


if __name__ == "__main__":
    main()
