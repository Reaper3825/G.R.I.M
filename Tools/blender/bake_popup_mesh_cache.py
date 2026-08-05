"""
Bake a Blender geometry-node (or any animated) mesh to a GRIM Mesh Cache (.gmc)
for the popup 3D sprite animation system.

The .gmc format is a flat little-endian binary consumed by
popup_ui/objects/popup_mesh_cache_loader.cpp:

    char     magic[8]   = "GRIMMC03"
    float32  fps
    uint32   frameCount
        uint32   flags        (bit0 = HAS_NORMALS, bit1 = HAS_UV, bit2 = HAS_COLOR,
                                                     bit3 = HAS_MATERIAL_PROGRAM)
    uint32   maxVertices
    uint32   maxIndices
        --- material program ---
            uint32 registerCount
            uint32 instructionCount
            uint32 parameterCount
            uint32 instructions[8 * instructionCount]
            float32 parameters[4 * parameterCount]
    --- per frame ---
      uint32  vertCount
      uint32  indexCount
      float32 positions[3 * vertCount]
      float32 normals[3 * vertCount]    (if HAS_NORMALS)
      float32 uvs[2 * vertCount]        (if HAS_UV)
            uint32  colorsABGR[vertCount]      (if HAS_COLOR)
      uint16  indices[indexCount]

Constraints:
  * Each baked frame must have <= 65535 vertices (16-bit index buffer).
  * Geometry nodes that "instance" geometry should use a Realize Instances
    node before output so the evaluated mesh has real vertices.
  * The mesh is exported in its OBJECT-LOCAL space (apply the same transform
    as grim_popup.obj). The popup renderer expects the model centered near the
    origin at roughly unit scale.

Usage (headless):
    blender my_scene.blend --background --python tools/blender/bake_popup_mesh_cache.py -- \
        --object GrimPopup --start 1 --end 60 --fps 30 \
        --out resources/popup_3d/grim_popup_load_in.gmc

Usage (inside Blender's Scripting tab): set ARGS below or call bake(...) directly.
"""

import bpy
import bmesh
import struct
import sys
import argparse
import os
from mathutils import Matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from popup_material_export import compile_object_material_program

MAGIC = b"GRIMMC03"
FLAG_HAS_NORMALS = 1 << 0
FLAG_HAS_UV = 1 << 1
FLAG_HAS_COLOR = 1 << 2
FLAG_HAS_MATERIAL_PROGRAM = 1 << 3

# Z-up (Blender) -> Y-up, matching Blender's default OBJ export (forward -Z, up Y),
# i.e. (x, y, z) -> (x, z, -y). This keeps a baked clip aligned with grim_popup.obj.
AXIS_YUP = Matrix(((1, 0, 0, 0),
                   (0, 0, 1, 0),
                   (0, -1, 0, 0),
                   (0, 0, 0, 1)))
AXIS_RAW = Matrix.Identity(4)


def _parse_args(argv):
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser(description="Bake a mesh animation to .gmc")
    p.add_argument("--object", required=True, help="Name of the object to bake")
    p.add_argument("--start", type=int, default=None, help="Start frame (default: scene start)")
    p.add_argument("--end", type=int, default=None, help="End frame (default: scene end)")
    p.add_argument("--fps", type=float, default=None, help="Playback fps (default: scene fps)")
    p.add_argument("--out", required=True, help="Output .gmc path")
    p.add_argument("--no-uv", action="store_true", help="Do not export UVs")
    p.add_argument("--axis", choices=["yup", "raw"], default="yup",
                   help="Coordinate conversion: 'yup' (default, matches grim_popup.obj) "
                        "or 'raw' (keep Blender Z-up local coords)")
    return p.parse_args(argv)


def _linear_to_srgb(value):
    value = max(0.0, min(1.0, value))
    if value <= 0.0031308:
        return value * 12.92
    return 1.055 * (value ** (1.0 / 2.4)) - 0.055


def _color_to_abgr(color):
    red = round(_linear_to_srgb(float(color[0])) * 255.0)
    green = round(_linear_to_srgb(float(color[1])) * 255.0)
    blue = round(_linear_to_srgb(float(color[2])) * 255.0)
    alpha = round(max(0.0, min(1.0, float(color[3]))) * 255.0)
    return red | (green << 8) | (blue << 16) | (alpha << 24)


def _material_color_source(material, object_name):
    if material is None:
        raise RuntimeError(f"Object '{object_name}' has an unassigned material slot")
    if material.node_tree is None:
        raise RuntimeError(
            f"Material '{material.name}' must use nodes with one Principled BSDF"
        )

    principled_nodes = [
        node for node in material.node_tree.nodes
        if node.type == "BSDF_PRINCIPLED"
    ]
    if len(principled_nodes) == 1:
        principled = principled_nodes[0]
        base_color_input = principled.inputs.get("Base Color")
        alpha_input = principled.inputs.get("Alpha")
        if base_color_input is None or alpha_input is None:
            raise RuntimeError(f"Material '{material.name}' has no Base Color or Alpha input")
        if not base_color_input.is_linked and not alpha_input.is_linked:
            base_color = base_color_input.default_value
            color = (base_color[0], base_color[1], base_color[2],
                     float(alpha_input.default_value) * float(base_color[3]))
            return ("constant", color)

    color_attribute_nodes = [
        node for node in material.node_tree.nodes
        if node.type == "VERTEX_COLOR" and node.layer_name
    ]
    if len(color_attribute_nodes) == 1:
        return ("attribute", color_attribute_nodes[0].layer_name)

    if len(principled_nodes) > 1:
        raise RuntimeError(
            f"Material '{material.name}' must contain exactly one Principled BSDF"
        )
    raise RuntimeError(
        f"Material '{material.name}' must use one constant Principled BSDF or "
        "one named Color Attribute node"
    )


def _evaluate_mesh(obj, depsgraph, axis_mtx):
    """Return triangulated (verts, normals, uvs, colors, indices) for the object's
    evaluated (modifier + geometry-node) mesh, transformed into engine space.

    Vertices are baked in WORLD space (so object transforms are included, like
    Blender's OBJ exporter) and then run through `axis_mtx` for the up-axis swap.
    """
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()

    material_sources = [
        _material_color_source(slot.material, obj.name)
        for slot in eval_obj.material_slots
    ]
    if not material_sources:
        raise RuntimeError(f"Object '{obj.name}' has no material slots")

    point_colors = {}
    for source_type, source_value in material_sources:
        if source_type != "attribute" or source_value in point_colors:
            continue
        color_attribute = mesh.color_attributes.get(source_value)
        if color_attribute is None:
            raise RuntimeError(
                f"Object '{obj.name}' evaluated mesh has no '{source_value}' color attribute"
            )
        if color_attribute.domain != "POINT":
            raise RuntimeError(
                f"Object '{obj.name}' color attribute '{source_value}' must use POINT domain"
            )
        point_colors[source_value] = [
            _color_to_abgr(item.color) for item in color_attribute.data
        ]

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.normal_update()
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()

    world = eval_obj.matrix_world.copy()
    pos_mtx = axis_mtx @ world
    # Normal matrix = inverse-transpose of the 3x3, then axis swap.
    nrm_mtx = (axis_mtx.to_3x3() @ world.to_3x3().inverted_safe().transposed())

    verts = []
    normals = []
    uvs = []
    colors = []
    indices = []
    vertex_map = {}
    uv_layer = bm.loops.layers.uv.active
    for face in bm.faces:
        if face.material_index >= len(material_sources):
            raise RuntimeError(
                f"Object '{obj.name}' face references missing material slot "
                f"{face.material_index}"
            )
        source_type, source_value = material_sources[face.material_index]
        for loop in face.loops:
            color = (_color_to_abgr(source_value) if source_type == "constant"
                     else point_colors[source_value][loop.vert.index])
            uv = ((loop[uv_layer].uv.x, loop[uv_layer].uv.y)
                  if uv_layer is not None else (0.0, 0.0))
            key = (loop.vert.index, uv[0], uv[1], color)
            index = vertex_map.get(key)
            if index is None:
                source_vert = loop.vert
                position = pos_mtx @ source_vert.co
                normal = nrm_mtx @ source_vert.normal
                normal.normalize()
                index = len(verts)
                vertex_map[key] = index
                verts.append((position.x, position.y, position.z))
                normals.append((normal.x, normal.y, normal.z))
                uvs.append(uv)
                colors.append(color)
            indices.append(index)

    bm.free()
    eval_obj.to_mesh_clear()
    return verts, normals, uvs, colors, indices


def bake(object_name, out_path, start=None, end=None, fps=None, export_uv=True, axis="yup"):
    scene = bpy.context.scene
    obj = bpy.data.objects.get(object_name)
    if obj is None:
        raise RuntimeError(f"Object '{object_name}' not found")

    start = scene.frame_start if start is None else start
    end = scene.frame_end if end is None else end
    fps = (scene.render.fps / scene.render.fps_base) if fps is None else fps

    axis_mtx = AXIS_YUP if axis == "yup" else AXIS_RAW
    register_count, material_instructions, material_parameters = \
        compile_object_material_program(obj)

    flags = (FLAG_HAS_NORMALS | FLAG_HAS_COLOR | FLAG_HAS_MATERIAL_PROGRAM |
             (FLAG_HAS_UV if export_uv else 0))

    frames = []  # list of (verts, normals, uvs, colors, indices)
    max_v = 0
    max_i = 0

    for f in range(start, end + 1):
        scene.frame_set(f)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        verts, normals, uvs, colors, indices = _evaluate_mesh(obj, depsgraph, axis_mtx)

        if len(verts) > 65535:
            raise RuntimeError(
                f"Frame {f} has {len(verts)} vertices (> 65535). "
                f"Decimate the geo-node result or reduce density."
            )
        max_v = max(max_v, len(verts))
        max_i = max(max_i, len(indices))
        frames.append((verts, normals, uvs, colors, indices))
        print(f"  frame {f}: {len(verts)} verts, {len(indices)} indices")

    with open(out_path, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<f", float(fps)))
        fh.write(struct.pack("<I", len(frames)))
        fh.write(struct.pack("<I", flags))
        fh.write(struct.pack("<I", max_v))
        fh.write(struct.pack("<I", max_i))
        fh.write(struct.pack("<III", register_count,
                             len(material_instructions), len(material_parameters)))
        for instruction in material_instructions:
            fh.write(struct.pack("<8I", *instruction))
        for parameter in material_parameters:
            fh.write(struct.pack("<4f", *parameter))

        for verts, normals, uvs, colors, indices in frames:
            fh.write(struct.pack("<I", len(verts)))
            fh.write(struct.pack("<I", len(indices)))
            fh.write(struct.pack(f"<{len(verts) * 3}f",
                                 *[c for v in verts for c in v]))
            fh.write(struct.pack(f"<{len(normals) * 3}f",
                                 *[c for n in normals for c in n]))
            if flags & FLAG_HAS_UV:
                fh.write(struct.pack(f"<{len(uvs) * 2}f",
                                     *[c for uv in uvs for c in uv]))
            fh.write(struct.pack(f"<{len(colors)}I", *colors))
            fh.write(struct.pack(f"<{len(indices)}H", *indices))

    print(f"Wrote {out_path}: {len(frames)} frames, fps={fps}, "
          f"maxVerts={max_v}, maxIndices={max_i}")


def main():
    args = _parse_args(sys.argv)
    bake(args.object, args.out, args.start, args.end, args.fps, not args.no_uv, args.axis)


if __name__ == "__main__":
    main()
