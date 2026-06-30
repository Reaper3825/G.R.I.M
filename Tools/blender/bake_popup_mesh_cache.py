"""
Bake a Blender geometry-node (or any animated) mesh to a GRIM Mesh Cache (.gmc)
for the popup 3D sprite animation system.

The .gmc format is a flat little-endian binary consumed by
popup_ui/objects/popup_mesh_cache_loader.cpp:

    char     magic[8]   = "GRIMMC01"
    float32  fps
    uint32   frameCount
    uint32   flags        (bit0 = HAS_NORMALS, bit1 = HAS_UV)
    uint32   maxVertices
    uint32   maxIndices
    --- per frame ---
      uint32  vertCount
      uint32  indexCount
      float32 positions[3 * vertCount]
      float32 normals[3 * vertCount]    (if HAS_NORMALS)
      float32 uvs[2 * vertCount]        (if HAS_UV)
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
from mathutils import Matrix

MAGIC = b"GRIMMC01"
FLAG_HAS_NORMALS = 1 << 0
FLAG_HAS_UV = 1 << 1

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


def _evaluate_mesh(obj, depsgraph, axis_mtx):
    """Return triangulated (verts, normals, uvs, indices) for the object's
    evaluated (modifier + geometry-node) mesh, transformed into engine space.

    Vertices are baked in WORLD space (so object transforms are included, like
    Blender's OBJ exporter) and then run through `axis_mtx` for the up-axis swap.
    """
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.normal_update()

    world = eval_obj.matrix_world.copy()
    pos_mtx = axis_mtx @ world
    # Normal matrix = inverse-transpose of the 3x3, then axis swap.
    nrm_mtx = (axis_mtx.to_3x3() @ world.to_3x3().inverted_safe().transposed())

    verts = []
    normals = []
    for v in bm.verts:
        p = pos_mtx @ v.co
        verts.append((p.x, p.y, p.z))
        n = (nrm_mtx @ v.normal)
        n.normalize()
        normals.append((n.x, n.y, n.z))

    # UVs: per-loop in Blender; collapse to per-vertex (last loop wins).
    uvs = [(0.0, 0.0)] * len(bm.verts)
    uv_layer = bm.loops.layers.uv.active
    if uv_layer is not None:
        for face in bm.faces:
            for loop in face.loops:
                uvs[loop.vert.index] = (loop[uv_layer].uv.x, loop[uv_layer].uv.y)

    indices = []
    for face in bm.faces:
        for loop in face.loops:
            indices.append(loop.vert.index)

    bm.free()
    eval_obj.to_mesh_clear()
    return verts, normals, uvs, indices


def bake(object_name, out_path, start=None, end=None, fps=None, export_uv=True, axis="yup"):
    scene = bpy.context.scene
    obj = bpy.data.objects.get(object_name)
    if obj is None:
        raise RuntimeError(f"Object '{object_name}' not found")

    start = scene.frame_start if start is None else start
    end = scene.frame_end if end is None else end
    fps = (scene.render.fps / scene.render.fps_base) if fps is None else fps

    axis_mtx = AXIS_YUP if axis == "yup" else AXIS_RAW

    flags = FLAG_HAS_NORMALS | (FLAG_HAS_UV if export_uv else 0)

    frames = []  # list of (verts, normals, uvs, indices)
    max_v = 0
    max_i = 0

    for f in range(start, end + 1):
        scene.frame_set(f)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        verts, normals, uvs, indices = _evaluate_mesh(obj, depsgraph, axis_mtx)

        if len(verts) > 65535:
            raise RuntimeError(
                f"Frame {f} has {len(verts)} vertices (> 65535). "
                f"Decimate the geo-node result or reduce density."
            )
        max_v = max(max_v, len(verts))
        max_i = max(max_i, len(indices))
        frames.append((verts, normals, uvs, indices))
        print(f"  frame {f}: {len(verts)} verts, {len(indices)} indices")

    with open(out_path, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<f", float(fps)))
        fh.write(struct.pack("<I", len(frames)))
        fh.write(struct.pack("<I", flags))
        fh.write(struct.pack("<I", max_v))
        fh.write(struct.pack("<I", max_i))

        for verts, normals, uvs, indices in frames:
            fh.write(struct.pack("<I", len(verts)))
            fh.write(struct.pack("<I", len(indices)))
            fh.write(struct.pack(f"<{len(verts) * 3}f",
                                 *[c for v in verts for c in v]))
            fh.write(struct.pack(f"<{len(normals) * 3}f",
                                 *[c for n in normals for c in n]))
            if flags & FLAG_HAS_UV:
                fh.write(struct.pack(f"<{len(uvs) * 2}f",
                                     *[c for uv in uvs for c in uv]))
            fh.write(struct.pack(f"<{len(indices)}H", *indices))

    print(f"Wrote {out_path}: {len(frames)} frames, fps={fps}, "
          f"maxVerts={max_v}, maxIndices={max_i}")


def main():
    args = _parse_args(sys.argv)
    bake(args.object, args.out, args.start, args.end, args.fps, not args.no_uv, args.axis)


if __name__ == "__main__":
    main()
