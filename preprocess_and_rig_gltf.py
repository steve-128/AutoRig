#!/usr/bin/env python3
"""
preprocess_and_rig_gltf.py
Generates a rigged 2.5D mesh from a child drawing and exports as .glb for Unity.
Produces:
  - <out>/character.glb       (mesh + skin + texture)
  - <out>/character_tex.png   (texture image)
  - debug images: mask.png, skeleton_overlay.png, joints.png
  - rig.json (bones, vertices, weights, bbox)
"""

#  python3 -m pip install numpy scipy opencv-python scikit-image imageio pygltflib
import argparse
import os
import json
from pathlib import Path

import numpy as np
import imageio
import cv2
from scipy import ndimage as ndi
from scipy.spatial import Delaunay
from skimage import color, filters, morphology, measure, util

from pygltflib import GLTF2, Scene, Node, Mesh, Buffer, BufferView, Accessor, \
    Asset, Skin, Image as GLTFImage, Texture as GLTFTexture, TextureInfo, Material, \
    Primitive, Sampler, ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER
# Accessor component types (GLTF numeric constants)
COMP_FLOAT = 5126
COMP_UNSIGNED_SHORT = 5123
COMP_UNSIGNED_INT = 5125

# -------------------- PHASE 1: helpers (from previous script) -------------------- #

def load_image(fn):
    im = imageio.v2.imread(fn)
    if im.ndim == 2:
        im = np.stack([im, im, im], axis=-1)
    if im.shape[2] == 4:
        im = cv2.cvtColor(im, cv2.COLOR_RGBA2RGB)
    return im

def compute_foreground_mask(im, blur=3, block_size=51, offset=0.0):
    gray = color.rgb2gray(im)
    blurred = filters.gaussian(gray, sigma=blur)
    try:
        thresh = filters.threshold_local(blurred, block_size=block_size)
        mask = blurred > (thresh + offset)
    except Exception:
        t = filters.threshold_otsu(blurred)
        mask = blurred > (t + offset)
    mask = morphology.remove_small_holes(mask, area_threshold=64)
    mask = morphology.remove_small_objects(mask, min_size=64)
    mask = morphology.binary_closing(mask, morphology.disk(3))
    return mask.astype(bool)

def skeletonize_mask(mask):
    sk = morphology.thin(mask)
    return (sk > 0).astype(np.uint8)

def find_skeleton_keypoints(skel):
    b = (skel > 0).astype(np.uint8)
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    neigh = ndi.convolve(b, kernel, mode='constant', cval=0)
    ys, xs = np.nonzero(b)
    keypoints = []
    for y, x in zip(ys, xs):
        n = int(neigh[y,x])
        if n == 1:
            keypoints.append({"x": int(x), "y": int(y), "type": "endpoint"})
        elif n >= 3:
            keypoints.append({"x": int(x), "y": int(y), "type": "junction"})
    return keypoints

def extract_contours(mask):
    contours = measure.find_contours(mask.astype(np.uint8), level=0.5)
    if len(contours) == 0:
        return np.empty((0,2))
    longest = max(contours, key=lambda c: c.shape[0])
    contour_xy = np.fliplr(longest)
    return contour_xy

def sample_interior_points(mask, n_points=1200):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((0,2))
    idx = np.random.choice(len(xs), size=min(n_points, len(xs)), replace=False)
    pts = np.vstack([xs[idx], ys[idx]]).T
    return pts

def build_delaunay(points):
    if len(points) < 3:
        return None
    tri = Delaunay(points)
    return tri

def prune_and_reduce_joints(joints, mask_shape, max_joints=15):
    if len(joints) == 0:
        return []
    pts = np.array([[j["x"], j["y"]] for j in joints])
    h, w = mask_shape
    cell = max(10, int(max(w,h)/50))
    keys = {}
    for i, (x,y) in enumerate(pts):
        k = (int(x//cell), int(y//cell))
        keys.setdefault(k, []).append(i)
    clusters = []
    for idxs in keys.values():
        cluster_pts = pts[idxs].mean(axis=0)
        clusters.append(tuple(cluster_pts.astype(int)))
    clusters = np.array(clusters)
    if len(clusters) <= max_joints:
        out = [{"x": int(p[0]), "y": int(p[1]), "name": f"j{ii}"} for ii,p in enumerate(clusters)]
        return out
    selected = [clusters[0]]
    while len(selected) < max_joints:
        dists = np.min([np.sum((clusters - s)**2, axis=1) for s in selected], axis=0)
        idx = np.argmax(dists)
        selected.append(clusters[idx])
    out = [{"x": int(p[0]), "y": int(p[1]), "name": f"j{ii}"} for ii,p in enumerate(np.array(selected))]
    return out

def make_simple_hierarchy(joints):
    if len(joints) == 0:
        return []
    pts = np.array([[j["x"], j["y"]] for j in joints])
    root_idx = int(np.argmax(pts[:,1]))
    names = [j.get("name", f"j{i}") for i,j in enumerate(joints)]
    parents = {}
    for i, p in enumerate(pts):
        if i == root_idx:
            parents[names[i]] = None
            continue
        candidates = [k for k,q in enumerate(pts) if q[1] >= p[1] and k != i]
        if not candidates:
            dists = np.sum((pts - p)**2, axis=1)
            cand = int(np.argmin(np.where(np.arange(len(pts))!=i, dists, np.inf)))
            parents[names[i]] = names[cand]
        else:
            cand_dists = [np.sum((pts[c] - p)**2) for c in candidates]
            cand_idx = candidates[int(np.argmin(cand_dists))]
            parents[names[i]] = names[cand_idx]
    bones = []
    for i,j in enumerate(joints):
        bones.append({"name": names[i], "parent": parents[names[i]], "x": int(j["x"]), "y": int(j["y"])})
    return bones

def compute_vertex_weights(points, bones, falloff=40.0, max_influences=4):
    if len(bones) == 0:
        return [[] for _ in range(len(points))]
    bone_pts = np.array([[b["x"], b["y"]] for b in bones])
    N = len(points)
    M = len(bones)
    d2 = np.sum((points[:,None,:] - bone_pts[None,:,:])**2, axis=2)
    sigma = float(falloff)
    raw = np.exp(-d2/(2*sigma*sigma))
    raw[raw < 1e-6] = 0.0
    s = raw.sum(axis=1, keepdims=True)
    s[s==0] = 1.0
    norm = raw / s
    weights = []
    for i in range(N):
        row = norm[i]
        top_idx = np.argsort(row)[::-1][:max_influences]
        wlist = []
        for idx in top_idx:
            if row[idx] <= 0:
                continue
            wlist.append((bones[idx]["name"], float(row[idx])))
        weights.append(wlist)
    return weights

def save_debug_images(out_dir, im, mask, skel, joints):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    imageio.v2.imwrite(os.path.join(out_dir, "mask.png"), util.img_as_ubyte(mask))
    vis = np.stack([util.img_as_ubyte(mask)]*3, axis=-1)
    vis[skel > 0] = [255, 0, 0]
    imageio.v2.imwrite(os.path.join(out_dir, "skeleton_overlay.png"), vis)
    vis2 = cv2.cvtColor(util.img_as_ubyte(mask), cv2.COLOR_GRAY2BGR)
    for j in joints:
        cv2.circle(vis2, (j["x"], j["y"]), radius=3, color=(0,255,0), thickness=-1)
    imageio.v2.imwrite(os.path.join(out_dir, "joints.png"), vis2)

# -------------------- glTF export function (complete) -------------------- #

def export_gltf(points, simplices, uv_coords, bones, weights, texture_img, out_fn="character.glb"):
    """
    Export mesh + skin + texture as glTF2 (.glb).
    points: (N,2)
    simplices: (T,3)
    uv_coords: (N,2)
    bones: list of dicts with 'name','parent','x','y'
    weights: per-vertex list of [(bone_name, weight), ...]
    texture_img: np.array (H,W,3)
    """
    gltf = GLTF2(asset=Asset(version="2.0"))

    # Prepare arrays
    pos = np.array([[float(x), float(-y), 0.0] for (x,y) in points], dtype=np.float32)   # flip Y
    uvs = np.array(uv_coords, dtype=np.float32)
    idx = np.array(simplices.flatten(), dtype=np.uint32)

    bone_map = {b["name"]: i for i,b in enumerate(bones)}
    J = np.zeros((len(points),4), dtype=np.uint16)
    W = np.zeros((len(points),4), dtype=np.float32)
    for vi, wlist in enumerate(weights):
        for k, (bn, w) in enumerate(wlist[:4]):
            if bn in bone_map:
                J[vi,k] = bone_map[bn]
                W[vi,k] = w
        s = W[vi].sum()
        if s > 0:
            W[vi] /= s

    # pack bytes in order: pos, uv, joints, weights, indices
    def to_bytes(a):
        return a.tobytes()

    chunks = [
        to_bytes(pos),
        to_bytes(uvs),
        to_bytes(J),
        to_bytes(W),
        to_bytes(idx)
    ]
    offsets = {}
    cursor = 0
    for name, chunk in zip(["pos","uv","joints","weights","indices"], chunks):
        offsets[name] = (cursor, len(chunk))
        cursor += len(chunk)
    blob = b"".join(chunks)

    # create Buffer
    gltf.buffers.append(Buffer(byteLength=len(blob)))

    # helper to create BufferView and Accessor
    def make_view(name, target=None):
        off, size = offsets[name]
        bv = BufferView(buffer=0, byteOffset=off, byteLength=size)
        if target is not None:
            bv.target = target
        gltf.bufferViews.append(bv)
        return len(gltf.bufferViews) - 1

    def make_accessor(bv_idx, compType, typeStr, count):
        acc = Accessor(bufferView=bv_idx, componentType=compType, count=count, type=typeStr)
        gltf.accessors.append(acc)
        return len(gltf.accessors) - 1

    bv_pos = make_view("pos", ARRAY_BUFFER)
    bv_uv  = make_view("uv", ARRAY_BUFFER)
    bv_j   = make_view("joints", ARRAY_BUFFER)
    bv_w   = make_view("weights", ARRAY_BUFFER)
    bv_idx = make_view("indices", ELEMENT_ARRAY_BUFFER)

    a_pos = make_accessor(bv_pos, COMP_FLOAT, "VEC3", len(pos))
    a_uv  = make_accessor(bv_uv, COMP_FLOAT, "VEC2", len(uvs))
    a_j   = make_accessor(bv_j, COMP_UNSIGNED_SHORT, "VEC4", len(J))
    a_w   = make_accessor(bv_w, COMP_FLOAT, "VEC4", len(W))
    a_idx = make_accessor(bv_idx, COMP_UNSIGNED_INT, "SCALAR", len(idx))

    # Save texture image next to out_fn
    out_dir = os.path.dirname(out_fn)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tex_path = os.path.join(out_dir, os.path.basename(out_fn).replace(".glb", "_tex.png"))
    imageio.imwrite(tex_path, texture_img)

    # glTF image/texture/material
    gltf.images.append(GLTFImage(uri=os.path.basename(tex_path)))
    gltf.textures.append(GLTFTexture(source=0, sampler=0))
    gltf.samplers.append(Sampler())
    mat = Material(pbrMetallicRoughness={"baseColorTexture": TextureInfo(index=0)})
    gltf.materials.append(mat)

    # Mesh & Primitive
    prim = Primitive(attributes={"POSITION": a_pos, "TEXCOORD_0": a_uv, "JOINTS_0": a_j, "WEIGHTS_0": a_w},
                     indices=a_idx, material=0)
    mesh = Mesh(primitives=[prim])
    gltf.meshes.append(mesh)

    # --- Skin + Nodes ---
    nodes = []

    # Create skeleton root node
    skeleton_root = Node(name="Armature")
    skeleton_root_index = 0
    nodes.append(skeleton_root)

    # Create joint nodes (children of root or each other)
    for i, b in enumerate(bones):
        n = Node(name=b["name"], translation=[float(b["x"]), float(-b["y"]), 0.0])
        nodes.append(n)

    bone_map = {b["name"]: i+1 for i,b in enumerate(bones)}  # offset by +1 because root is index 0

    # Parent relationships
    for i, b in enumerate(bones):
        node_idx = bone_map[b["name"]]
        if b.get("parent") is None:
            # attach to skeleton root
            if nodes[skeleton_root_index].children is None:
                nodes[skeleton_root_index].children = []
            nodes[skeleton_root_index].children.append(node_idx)
        else:
            parent_idx = bone_map[b["parent"]]
            if nodes[parent_idx].children is None:
                nodes[parent_idx].children = []
            nodes[parent_idx].children.append(node_idx)

    # Mesh node
    mesh_node = Node(mesh=0, skin=0, name="CharacterMesh")
    nodes.append(mesh_node)
    mesh_node_idx = len(nodes) - 1

    # Add to glTF
    gltf.nodes.extend(nodes)

    # Skin (note: joints indices shifted by +1)
    gltf.skins.append(Skin(joints=list(bone_map.values()), skeleton=skeleton_root_index))

    # Scene
    gltf.scenes.append(Scene(nodes=[mesh_node_idx]))
    gltf.scene = 0

    # set binary blob
    gltf.set_binary_blob(blob)

    # save glb
    gltf.save(out_fn)
    print("Saved glTF:", out_fn)
    print("Saved texture PNG:", tex_path)

# -------------------- MAIN: pipeline + export call -------------------- #

def main(args):
    im = load_image(args.input)
    h, w = im.shape[:2]

    # 1) mask
    mask = compute_foreground_mask(im, blur=2, block_size=51, offset=0.0)
    if mask.sum() == 0:
        raise RuntimeError("Foreground mask is empty. Try different input or tweak threshold params.")

    # 2) skeleton + raw keypoints
    skel = skeletonize_mask(mask)
    raw_joints = find_skeleton_keypoints(skel)

    # 3) contours + points for triangulation
    contour = extract_contours(mask)
    interior = sample_interior_points(mask, n_points=2000)
    if contour.shape[0] > 0:
        pts = np.vstack([contour, interior])
    else:
        pts = interior
    tri = build_delaunay(pts)
    if tri is None:
        raise RuntimeError("Not enough points for triangulation.")

    points = pts.astype(float)
    simplices = tri.simplices.astype(int)

    # 4) reduce joints and build bones
    joints = prune_and_reduce_joints(raw_joints, mask.shape, max_joints=args.max_joints)
    bones = make_simple_hierarchy(joints)

    # 5) compute vertex weights
    weights = compute_vertex_weights(points, bones, falloff=args.falloff, max_influences=4)

    # 6) compute UVs and crop texture
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        minx, miny, maxx, maxy = 0, 0, w-1, h-1
    else:
        minx, miny, maxx, maxy = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    pad = 4
    minx = max(0, minx-pad); miny = max(0, miny-pad)
    maxx = min(w-1, maxx+pad); maxy = min(h-1, maxy+pad)
    crop = im[miny:maxy+1, minx:maxx+1]

    uv_coords = []
    denom_x = max(1.0, (maxx - minx))
    denom_y = max(1.0, (maxy - miny))
    for (x,y) in points:
        u = (x - minx) / denom_x
        v = (y - miny) / denom_y
        uv_coords.append((u, 1.0 - v))

    # 7) output debug files, rig.json
    out_dir = args.out
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    save_debug_images(out_dir, im, mask, skel, joints)
    rig_json = {
        "bones": bones,
        "vertices": [{"x": float(p[0]), "y": float(p[1])} for p in points],
        "weights": [[[bn, w] for (bn,w) in wlist] for wlist in weights],
        "bbox": {"minx": int(minx), "miny": int(miny), "maxx": int(maxx), "maxy": int(maxy)}
    }
    with open(os.path.join(out_dir, "rig.json"), "w") as f:
        json.dump(rig_json, f, indent=2)

    # 8) export to glb
    glb_path = os.path.join(out_dir, "character.glb")
    export_gltf(points, simplices, uv_coords, bones, weights, crop, out_fn=glb_path)

    print("Pipeline complete. Outputs in:", out_dir)
    print(f"Vertices: {len(points)}, Triangles: {len(simplices)}, Bones: {len(bones)}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="input drawing (png/jpg)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--falloff", type=float, default=40.0, help="sigma for Gaussian skin weights (pixels)")
    p.add_argument("--max_joints", type=int, default=15)
    args = p.parse_args()
    main(args)
