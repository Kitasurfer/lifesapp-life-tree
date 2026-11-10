"""Build the oak at one age from scratch and render it with Cycles (headless).

    blender -b --factory-startup --python blender/render_age.py -- \
        --npz data/oak195.npz --textures textures --age 30 --out out/oak_030.png

One fixed camera for every age: framing comes from the FULL-age tree, so frames line up
pixel-for-pixel and the app can lay the full-age silhouette behind any age.
Transparent film: only the tree above the ground is in the PNG — the app draws the sky, the ground
and the tree's shadow (dark/light theme).
"""
import argparse
import json
import math as _math
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import oak_camera as oc  # noqa: E402
import oak_growth as og  # noqa: E402
import oak_mesh as om  # noqa: E402

WARM = (1.0, 0.86, 0.66)  # back light: neutral enough for the app's light and dark scenes


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
    p.add_argument("--leaf-scale", type=float, default=1.8,
                   help="leaf size against the real 11 cm: larger, so single leaves read in a 1080 px frame")
    p.add_argument("--leaves", type=int, default=om.LEAVES_AT_FULL_AGE, help="leaves on the full-age tree")
    p.add_argument("--gaps", type=float, default=1.0, help="0 = closed shell of foliage, 1 = pads with openings")
    p.add_argument("--trunk-boost", type=float, default=3.0)
    p.add_argument("--sun", type=float, default=4.5)
    p.add_argument("--sun-from", type=float, nargs=3, default=(-42.0, -34.0, 34.0),
                   help="sun position (it shines towards the tree); front-left, low enough to reach the stem under the crown")
    p.add_argument("--sky", type=float, default=1.0, help="sky dome strength (soft daylight fill)")
    p.add_argument("--rim", type=float, default=0.0, help="warm back light power, W (0 = daylight only)")
    p.add_argument("--fill", type=float, default=0.0, help="cool front fill power, W (0 = daylight only)")
    p.add_argument("--exposure", type=float, default=0.0)
    return p.parse_args(argv)


# ---------------------------------------------------------------- mesh / material helpers

def make_mesh(name: str, data: om.MeshData, value_attr: str, material) -> bpy.types.Object:
    me = bpy.data.meshes.new(name)
    me.from_pydata(data.verts.tolist(), [], data.quads.tolist())
    uv = me.uv_layers.new(name="UVMap")
    uv.data.foreach_set("uv", data.loop_uv.astype(np.float32).ravel())
    if data.loop_uv_m is not None:
        metres = me.uv_layers.new(name="Metres")
        metres.data.foreach_set("uv", data.loop_uv_m.astype(np.float32).ravel())
    for attr_name, values in {value_attr: data.point_value, **data.extra}.items():
        attr = me.attributes.new(name=attr_name, type="FLOAT", domain="POINT")
        attr.data.foreach_set("value", values.astype(np.float32))
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


def attribute(nodes, name: str):
    node = nodes.new("ShaderNodeAttribute")
    node.attribute_name = name
    return node.outputs["Fac"]


def map_range(nodes, links, value, from_min, from_max, to_min=0.0, to_max=1.0):
    node = nodes.new("ShaderNodeMapRange")
    for key, v in (("From Min", from_min), ("From Max", from_max), ("To Min", to_min), ("To Max", to_max)):
        node.inputs[key].default_value = v
    links.new(value, node.inputs["Value"])
    return node.outputs["Result"]


def math(nodes, links, op: str, a, b=None, clamp: bool = False):
    node = nodes.new("ShaderNodeMath")
    node.operation = op
    node.use_clamp = clamp
    for socket, v in zip(node.inputs, (a, b)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            socket.default_value = v
        else:
            links.new(v, socket)
    return node.outputs["Value"]


def mix_colour(nodes, links, fac, a, b, blend: str = "MIX"):
    node, f, sa, sb, out = mix_rgba(nodes)
    node.blend_type = blend
    for socket, v in ((f, fac), (sa, a), (sb, b)):
        if isinstance(v, float):
            socket.default_value = v
        elif isinstance(v, tuple):
            socket.default_value = v
        else:
            links.new(v, socket)
    return out


def set_displacement(mat) -> None:
    for owner in (mat, getattr(mat, "cycles", None)):  # the setting moved between Blender versions
        if owner is not None and hasattr(owner, "displacement_method"):
            owner.displacement_method = "BOTH"
            return


def bark_material(tex: Path):
    """Old oak bark: the photoscan for texture and tone, deep lengthwise furrows as real relief.

    Thin young wood is smooth and grey-olive. The thicker (older) the wood, the deeper it is
    furrowed: ridges a hand wide run along the stem, split and rejoin; the furrows between them
    are displaced geometry on the stem and limbs, so they catch the light and break the outline.
    """
    mat = bpy.data.materials.new("Bark")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    output = nodes["Material Output"]
    d = tex / "bark"
    diff = image_node(nodes, d / "jolcham_oak_bark_01_diff_2k.jpg", False)
    rough = image_node(nodes, d / "jolcham_oak_bark_01_rough_2k.jpg", True)
    nor = image_node(nodes, d / "jolcham_oak_bark_01_nor_gl_2k.jpg", True)
    disp = image_node(nodes, d / "jolcham_oak_bark_01_disp_2k.jpg", True)

    radius = attribute(nodes, "radius")
    age_of_wood = map_range(nodes, links, radius, 0.012, 0.09)
    old = map_range(nodes, links, radius, 0.10, 0.75)      # 0 smooth-barked bough .. 1 the furrowed stem

    # Furrows, in metres of bark: the contour lines of a noise stretched along the wood. They run
    # lengthwise for a forearm or two, wander, fork and rejoin, and never close into scales.
    metres = nodes.new("ShaderNodeUVMap")
    metres.uv_map = "Metres"

    def furrows(across: float, along: float, width: float, offset: float):
        stretch = nodes.new("ShaderNodeMapping")
        stretch.inputs["Location"].default_value = (offset, offset * 0.37, 0.0)
        stretch.inputs["Scale"].default_value = (across, along, 1.0)
        links.new(metres.outputs["UV"], stretch.inputs["Vector"])
        noise = nodes.new("ShaderNodeTexNoise")
        noise.noise_dimensions = "2D"
        noise.inputs["Scale"].default_value = 1.0
        noise.inputs["Detail"].default_value = 2.5
        noise.inputs["Roughness"].default_value = 0.55
        noise.inputs["Distortion"].default_value = 0.6
        links.new(stretch.outputs["Vector"], noise.inputs["Vector"])
        off_contour = math(nodes, links, "ABSOLUTE", math(nodes, links, "SUBTRACT", noise.outputs["Fac"], 0.5))
        return map_range(nodes, links, off_contour, 0.0, width)   # 0 on the furrow line .. 1 on a ridge

    wide = furrows(7.0, 0.9, 0.085, 0.0)       # main furrows, a hand apart
    fine = furrows(15.0, 2.2, 0.10, 31.0)      # cracks across the ridges
    relief = math(nodes, links, "MULTIPLY", math(nodes, links, "POWER", wide, 0.7),
                  map_range(nodes, links, fine, 0.0, 1.0, 0.72, 1.0))
    # 0 furrow floor .. 1 ridge top; no furrows at the UV seam (the pattern does not wrap round the wood)
    ridge = math(nodes, links, "SUBTRACT", 1.0, math(
        nodes, links, "MULTIPLY", attribute(nodes, "seam"), math(nodes, links, "SUBTRACT", 1.0, relief)))

    # The scan is pale and mossy-green; keep its light/dark pattern, recolour it to oak grey-brown.
    luminance = nodes.new("ShaderNodeRGBToBW")
    oak_tone = nodes.new("ShaderNodeValToRGB")
    oak_tone.color_ramp.elements[0].position = 0.06
    oak_tone.color_ramp.elements[0].color = (0.035, 0.027, 0.020, 1.0)
    oak_tone.color_ramp.elements[1].position = 0.5
    oak_tone.color_ramp.elements[1].color = (0.37, 0.285, 0.20, 1.0)
    links.new(diff.outputs["Color"], luminance.inputs["Color"])
    links.new(luminance.outputs["Val"], oak_tone.inputs["Fac"])
    colour = mix_colour(nodes, links, age_of_wood, (0.13, 0.11, 0.085, 1.0), oak_tone.outputs["Color"])

    # Furrow floors are dark and damp; the ridge tops weather pale grey.
    in_furrow = math(nodes, links, "MULTIPLY", math(nodes, links, "SUBTRACT", 1.0, ridge), old)
    colour = mix_colour(nodes, links, math(nodes, links, "MULTIPLY", in_furrow, 0.5),
                        colour, (0.03, 0.022, 0.016, 1.0))
    weathered = math(nodes, links, "MULTIPLY", math(nodes, links, "POWER", ridge, 3.0),
                     math(nodes, links, "MULTIPLY", old, 0.16))
    colour = mix_colour(nodes, links, weathered, colour, (0.36, 0.31, 0.25, 1.0))

    # Big soft patches over the stem: no two metres of an old stem have the same tone.
    position = nodes.new("ShaderNodeNewGeometry")
    tone = nodes.new("ShaderNodeTexNoise")
    tone.inputs["Scale"].default_value = 0.55
    tone.inputs["Detail"].default_value = 4.0
    links.new(position.outputs["Position"], tone.inputs["Vector"])
    colour = mix_colour(nodes, links, 1.0, colour,
                        map_range(nodes, links, tone.outputs["Fac"], 0.3, 0.7, 0.62, 1.2), "MULTIPLY")
    # Grey-green lichen on the weather side of the old wood, in patches.
    lichen_noise = nodes.new("ShaderNodeTexNoise")
    lichen_noise.inputs["Scale"].default_value = 1.7
    lichen_noise.inputs["Detail"].default_value = 7.0
    links.new(position.outputs["Position"], lichen_noise.inputs["Vector"])
    lichen = math(nodes, links, "MULTIPLY", map_range(nodes, links, lichen_noise.outputs["Fac"], 0.58, 0.72, 0.0, 0.35),
                  math(nodes, links, "MULTIPLY", old, ridge))
    colour = mix_colour(nodes, links, lichen, colour, (0.30, 0.34, 0.24, 1.0))
    # Moss creeps up from the soil over the root flare, in patches, mostly on the ridge tops.
    xyz = nodes.new("ShaderNodeSeparateXYZ")
    links.new(position.outputs["Position"], xyz.inputs["Vector"])
    low = map_range(nodes, links, xyz.outputs["Z"], 0.0, 1.3, 0.75, 0.0)
    patches = nodes.new("ShaderNodeTexNoise")
    patches.inputs["Scale"].default_value = 2.2
    patches.inputs["Detail"].default_value = 6.0
    links.new(position.outputs["Position"], patches.inputs["Vector"])
    moss = math(nodes, links, "MULTIPLY", low, map_range(nodes, links, patches.outputs["Fac"], 0.42, 0.62))
    moss = math(nodes, links, "MULTIPLY", moss, map_range(nodes, links, ridge, 0.0, 1.0, 0.45, 1.0))
    colour = mix_colour(nodes, links, moss, colour, (0.075, 0.115, 0.03, 1.0))
    links.new(colour, bsdf.inputs["Base Color"])
    links.new(rough.outputs["Color"], bsdf.inputs["Roughness"])

    normal_map = nodes.new("ShaderNodeNormalMap")
    links.new(nor.outputs["Color"], normal_map.inputs["Color"])
    links.new(age_of_wood, normal_map.inputs["Strength"])
    links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])

    # Relief: furrows carry it on old wood, the photoscan's own height adds the fine grain.
    grain = math(nodes, links, "MULTIPLY", disp.outputs["Color"], 0.35)
    height = math(nodes, links, "ADD", math(nodes, links, "MULTIPLY", ridge, 0.65), grain)
    depth = math(nodes, links, "ADD", math(nodes, links, "MULTIPLY", old, 0.075),
                 math(nodes, links, "MULTIPLY", age_of_wood, 0.008))
    displacement = nodes.new("ShaderNodeDisplacement")
    displacement.inputs["Midlevel"].default_value = 0.75   # furrows cut in; ridges hardly stand out
    links.new(height, displacement.inputs["Height"])
    links.new(depth, displacement.inputs["Scale"])
    links.new(displacement.outputs["Displacement"], output.inputs["Displacement"])
    set_displacement(mat)
    return mat


def leaf_material(tex: Path):
    """Oak leaves: a dark waxy upper side, a paler matt underside, light shining through.

    Tone follows where a leaf grows: deep in the crown leaves are darker and bluer, on the lit
    outside lighter and more yellow; neighbouring sprays share a tone, single leaves vary a little.
    """
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

    rand, depth, tint = attribute(nodes, "leaf_rand"), attribute(nodes, "leaf_depth"), attribute(nodes, "leaf_tint")
    # The atlas leaves are pale spring green: turn them to summer oak green (hue 0.5 = unchanged).
    hue = math(nodes, links, "ADD", map_range(nodes, links, tint, 0.0, 1.0, 0.500, 0.545),
               math(nodes, links, "ADD", math(nodes, links, "MULTIPLY", depth, 0.02),
                    map_range(nodes, links, rand, 0.0, 1.0, -0.008, 0.008)))
    value = math(nodes, links, "MULTIPLY", map_range(nodes, links, depth, 0.0, 1.0, 0.66, 0.40),
                 math(nodes, links, "MULTIPLY", map_range(nodes, links, rand, 0.0, 1.0, 0.85, 1.15),
                      map_range(nodes, links, tint, 0.0, 1.0, 1.08, 0.92)))
    hsv = nodes.new("ShaderNodeHueSaturation")
    hsv.inputs["Saturation"].default_value = 1.03
    links.new(hue, hsv.inputs["Hue"])
    links.new(value, hsv.inputs["Value"])
    links.new(color.outputs["Color"], hsv.inputs["Color"])
    upper = hsv.outputs["Color"]
    # Underside: paler, greyer, no gloss.
    under = nodes.new("ShaderNodeHueSaturation")
    under.inputs["Saturation"].default_value = 0.72
    under.inputs["Value"].default_value = 1.35
    links.new(upper, under.inputs["Color"])
    geometry = nodes.new("ShaderNodeNewGeometry")
    side = mix_colour(nodes, links, geometry.outputs["Backfacing"], upper, under.outputs["Color"])
    links.new(side, bsdf.inputs["Base Color"])
    links.new(math(nodes, links, "MAXIMUM", rough.outputs["Color"],
                   math(nodes, links, "MULTIPLY", geometry.outputs["Backfacing"], 0.8)), bsdf.inputs["Roughness"])
    for name in ("Specular IOR Level", "Specular"):  # waxy, not glossy: a sheen, no white glints
        if name in bsdf.inputs:
            bsdf.inputs[name].default_value = 0.3
            break
    nmap = nodes.new("ShaderNodeNormalMap")
    nmap.inputs["Strength"].default_value = 0.6
    links.new(normal.outputs["Color"], nmap.inputs["Color"])
    links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

    # Light through a leaf is warmer and more saturated than light off it.
    through = nodes.new("ShaderNodeHueSaturation")
    through.inputs["Hue"].default_value = 0.485
    through.inputs["Saturation"].default_value = 1.2
    through.inputs["Value"].default_value = 1.25
    links.new(upper, through.inputs["Color"])
    translucent = nodes.new("ShaderNodeBsdfTranslucent")
    links.new(through.outputs["Color"], translucent.inputs["Color"])
    lit = nodes.new("ShaderNodeMixShader")
    lit.inputs["Fac"].default_value = 0.45
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
    obj.visible_camera = False  # the rim sits behind the crown: seen through gaps it reads as specks
    obj.location = location
    direction = Vector(target) - Vector(location)
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.collection.objects.link(obj)
    return obj


def frame_camera(scene, full_points: np.ndarray, width: int, height: int):
    """The fixed camera of every frame (pipeline/oak_camera.py), framed on the full-age tree."""
    framing = oc.solve(full_points, width, height)
    cam = bpy.data.cameras.new("Camera")
    cam.lens = oc.LENS_MM
    cam.sensor_fit = "VERTICAL"
    cam.sensor_height = oc.SENSOR_MM
    cam.shift_y = framing.shift_y
    cam.clip_end = framing.distance * 4
    obj = bpy.data.objects.new("Camera", cam)
    obj.location = (framing.x_c, -framing.distance, oc.EYE_M)
    obj.rotation_euler = (_math.pi / 2, 0.0, 0.0)
    scene.collection.objects.link(obj)
    scene.camera = obj
    return {"frame_height_m": framing.frame_h, "distance_m": framing.distance, "x_centre_m": framing.x_c}


def sky(scene, strength: float) -> None:
    """Daylight dome: pale and bright at the horizon, deeper blue overhead."""
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    direction = nodes.new("ShaderNodeTexCoord")
    xyz = nodes.new("ShaderNodeSeparateXYZ")
    links.new(direction.outputs["Generated"], xyz.inputs["Vector"])
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0.80, 0.87, 0.96, 1.0)
    ramp.color_ramp.elements[1].position = 0.7
    ramp.color_ramp.elements[1].color = (0.36, 0.57, 1.0, 1.0)
    links.new(xyz.outputs["Z"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], nodes["Background"].inputs["Color"])
    nodes["Background"].inputs["Strength"].default_value = strength
    scene.world = world


def main() -> None:
    args = parse_args()
    t0 = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    cards = [om.LeafCard(**c) for c in json.loads((args.textures / "leaf_cards.json").read_text())]
    skel = og.straighten(og.load_npz(args.npz))
    pre = og.precompute(skel)
    look = dict(leaf_scale=args.leaf_scale, trunk_boost=args.trunk_boost, leaves=args.leaves, gaps=args.gaps)
    full_tubes, full_leaves, _ = om.build_age(skel, pre, og.FULL_AGE, cards, **look)
    tubes, leaves, tree = om.build_age(skel, pre, args.age, cards, **look)
    t_geo = time.time()

    make_mesh("Oak", tubes, "radius", bark_material(args.textures))
    if len(leaves.quads):
        make_mesh("Leaves", leaves, "leaf_rand", leaf_material(args.textures))

    ground = bpy.data.meshes.new("Ground")
    ground.from_pydata([(-200, -200, 0), (200, -200, 0), (200, 200, 0), (-200, 200, 0)], [], [(0, 1, 2, 3)])
    # Holdout soil: it bounces green meadow light up under the crown and cuts the frame at the ground,
    # so buried root ends are not in the PNG (seen through an invisible ground they were flat blades).
    # No shadow catcher: the app draws its own ground and the tree's shadow in both themes.
    soil = bpy.data.materials.new("Soil")
    soil.use_nodes = True
    soil.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.10, 0.15, 0.05, 1.0)
    soil.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 1.0
    ground.materials.append(soil)
    ground_obj = bpy.data.objects.new("Ground", ground)
    ground_obj.is_holdout = True
    scene.collection.objects.link(ground_obj)

    framing = frame_camera(scene, np.vstack([full_tubes.verts, full_leaves.verts]), args.width, args.height)
    height = float(full_tubes.verts[:, 2].max())
    point_light("Key", "SUN", args.sun, (1.0, 0.93, 0.82), tuple(args.sun_from), (0.0, 0.0, 0.0))
    scene.objects["Key"].data.angle = _math.radians(1.2)
    if args.rim > 0:
        point_light("Rim", "AREA", args.rim, WARM, (0.0, 24.0, height * 0.75), (0.0, -10.0, height * 0.5), size=14.0)
    if args.fill > 0:
        point_light("Fill", "AREA", args.fill, (0.85, 0.95, 1.0), (15.0, -35.0, 12.0), (0.0, 0.0, height * 0.4), size=30.0)
    sky(scene, args.sky)

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = args.samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.max_bounces = 8
    scene.cycles.diffuse_bounces = 4
    scene.cycles.transmission_bounces = 8
    scene.cycles.transparent_max_bounces = 32
    scene.render.film_transparent = True
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = args.percent
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = args.exposure
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
