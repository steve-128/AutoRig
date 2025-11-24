#!/usr/bin/env python3
"""
hybrid_rigger.py

Combines AI-based pose detection (YOLO) with classical mesh generation
(Delaunay) to create a rigged 2.5D GLB model and Unity-compatible
JSON files (mesh_data.json, skeleton_data.json).

This version generates data that is CENTERED at (0,0,0) and
FLIPPED to be right-side up in Unity.
"""

# --- Imports from Script 3 (YOLO/OpenCV) ---
import cv2
import yaml
import os
from ultralytics import YOLO

# --- Imports from Script 1 (Mesh/Rig/GLTF) ---
import argparse
import json
from pathlib import Path
import numpy as np
import imageio
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

# ---------------------------------------------------------------------
# --- HELPER FUNCTIONS (from Script 1) ---
# ---------------------------------------------------------------------

def extract_contours(mask):
    """Finds the longest contour in a binary mask."""
    contours = measure.find_contours(mask.astype(np.uint8), level=0.5)
    if len(contours) == 0:
        return np.empty((0,2))
    longest = max(contours, key=lambda c: c.shape[0])
    contour_xy = np.fliplr(longest)
    return contour_xy

def sample_interior_points(mask, n_points=2000):
    """Samples random points from within a binary mask."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((0,2))
    idx = np.random.choice(len(xs), size=min(n_points, len(xs)), replace=False)
    pts = np.vstack([xs[idx], ys[idx]]).T
    return pts

def build_delaunay(points):
    """Performs Delaunay triangulation on a set of 2D points."""
    if len(points) < 3:
        return None
    tri = Delaunay(points)
    return tri

def compute_vertex_weights(points, bones, falloff=40.0, max_influences=4):
    """
    Computes skin weights for each vertex based on proximity to bones.
    'bones' is a list of dicts: [{"name":..., "x":..., "y":...}, ...]
    """
    if len(bones) == 0:
        return [[] for _ in range(len(points))]
    
    bone_pts = np.array([[b["x"], b["y"]] for b in bones])
    N = len(points)
    
    # Calculate squared distances
    d2 = np.sum((points[:,None,:] - bone_pts[None,:,:])**2, axis=2)
    
    # Gaussian weighting
    sigma = float(falloff)
    raw = np.exp(-d2/(2*sigma*sigma))
    raw[raw < 1e-6] = 0.0
    
    # Normalize
    s = raw.sum(axis=1, keepdims=True)
    s[s==0] = 1.0
    norm = raw / s
    
    # Get top 4 influences
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

def export_gltf(points, simplices, uv_coords, bones, weights, texture_img, out_fn="character.glb", root_loc=None):
    """
    Export mesh + skin + texture as glTF2 (.glb).
    ** UPDATED ** to accept a root_loc to center the model.
    """
    gltf = GLTF2(asset=Asset(version="2.0"))
    
    # --- *** NEW: CENTER AND FLIP *** ---
    origin_x, origin_y = (root_loc[0], root_loc[1]) if root_loc else (0, 0)

    # Prepare arrays
    # Apply offset and flip Y-axis
    pos = np.array([[float(x - origin_x), float(-(y - origin_y)), 0.0] for (x,y) in points], dtype=np.float32)
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
    def to_bytes(a): return a.tobytes()
    chunks = [to_bytes(pos), to_bytes(uvs), to_bytes(J), to_bytes(W), to_bytes(idx)]
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
        if target is not None: bv.target = target
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
    # This is the cropped texture for the GLB
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
    skeleton_root = Node(name="Armature")
    skeleton_root_index = 0
    nodes.append(skeleton_root)

    # Create joint nodes
    for i, b in enumerate(bones):
        # --- *** NEW: CENTER AND FLIP *** ---
        # Apply offset and flip Y-axis to bone translations
        trans_x = float(b["x"] - origin_x)
        trans_y = float(-(b["y"] - origin_y))
        n = Node(name=b["name"], translation=[trans_x, trans_y, 0.0])
        nodes.append(n)
    bone_map = {b["name"]: i+1 for i,b in enumerate(bones)}  # offset by +1

    # Parent relationships
    for i, b in enumerate(bones):
        node_idx = bone_map[b["name"]]
        if b.get("parent") is None:
            if nodes[skeleton_root_index].children is None:
                nodes[skeleton_root_index].children = []
            nodes[skeleton_root_index].children.append(node_idx)
        elif b["parent"] in bone_map:
            parent_idx = bone_map[b["parent"]]
            if nodes[parent_idx].children is None:
                nodes[parent_idx].children = []
            nodes[parent_idx].children.append(node_idx)

    # Mesh node
    mesh_node = Node(mesh=0, skin=0, name="CharacterMesh")
    nodes.append(mesh_node)
    mesh_node_idx = len(nodes) - 1
    gltf.nodes.extend(nodes)

    # Skin
    gltf.skins.append(Skin(joints=list(bone_map.values()), skeleton=skeleton_root_index))

    # Scene
    gltf.scenes.append(Scene(nodes=[mesh_node_idx]))
    gltf.scene = 0
    gltf.set_binary_blob(blob)
    gltf.save(out_fn)
    print(f"✅ Saved glTF: {out_fn}")
    print(f"✅ Saved texture PNG: {tex_path}")


# ---------------------------------------------------------------------
# --- MAIN CLASS (from Script 3, updated) ---
# ---------------------------------------------------------------------

class CharacterExtractorYOLO:
    def __init__(self, image_path, output_dir="output", args=None):
        self.image_path = image_path
        self.output_dir = output_dir
        self.args = args if args else {} # Store args like 'falloff'
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Could not read image {image_path}")
        
        # Ensure image is 3-channel BGR
        if len(self.image.shape) == 2 or self.image.shape[2] == 1:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        elif self.image.shape[2] == 4:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGRA2BGR)

        self.height, self.width = self.image.shape[:2]
        self.model = YOLO("yolov8n-pose.pt")  # full-body pose model
        self.keypoints = None
        self.skeleton = []
        self.mask = None

    # --- Methods from Script 3 ---

    def detect_pose(self):
        results = self.model(self.image, verbose=False)
        if not results or len(results[0].keypoints.data) == 0:
            print("⚠ No pose detected.")
            return False
        kp = results[0].keypoints.data[0].cpu().numpy()[:, :2]  # (17,2)
        self.keypoints = kp
        return True

    def create_mask_and_texture(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Otsu's method finds the best threshold automatically
        _, self.mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological cleanup
        kernel = np.ones((3,3), np.uint8)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        if self.mask.sum() == 0:
            print("⚠ Foreground mask is empty. Aborting.")
            return False

        mask_path = os.path.join(self.output_dir, "mask.png")
        cv2.imwrite(mask_path, self.mask)
        print(f"✅ Saved mask: {mask_path}")
        
        # Save original texture
        original_image_path = os.path.join(self.output_dir, "original_image.png")
        cv2.imwrite(original_image_path, self.image)
        print(f"✅ Saved original image: {original_image_path}")
        return True

    def get_face_anchor_points(self):
        if self.keypoints is None: return None
        kp = self.keypoints
        nose = kp[0]
        left_eye = kp[1]
        right_eye = kp[2]
        left_ear = kp[3] if len(kp) > 3 else None
        right_ear = kp[4] if len(kp) > 4 else None
        eye_mid_y = (left_eye[1] + right_eye[1]) / 2
        eye_span = abs(left_eye[0] - right_eye[0])
        if left_ear is None or right_ear is None or np.any(np.isnan(left_ear)) or np.any(np.isnan(right_ear)):
            face_left = [left_eye[0] - 2.0 * eye_span, eye_mid_y]
            face_right = [right_eye[0] + 2.0 * eye_span, eye_mid_y]
        else:
            face_left = left_ear
            face_right = right_ear
        return {"nose": nose, "left_eye": left_eye, "right_eye": right_eye,
                "face_left": face_left, "face_right": face_right}

    def build_skeleton(self):
        if self.keypoints is None: return
        kp = self.keypoints
        def xy(i): return [float(kp[i][0]), float(kp[i][1])]
        joints = {
            "nose": xy(0), "left_eye": xy(1), "right_eye": xy(2),
            "left_ear": xy(3), "right_ear": xy(4),
            "left_shoulder": xy(5), "right_shoulder": xy(6),
            "left_elbow": xy(7), "right_elbow": xy(8),
            "left_hand": xy(9), "right_hand": xy(10),
            "left_hip": xy(11), "right_hip": xy(12),
            "left_knee": xy(13), "right_knee": xy(14),
            "left_foot": xy(15), "right_foot": xy(16),
        }
        face_points = self.get_face_anchor_points()
        if face_points:
            for k, v in face_points.items():
                joints[k] = [float(v[0]), float(v[1])]
        def mid(a, b): return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]
        
        self.skeleton = [
            {"name": "root", "loc": mid(joints["left_hip"], joints["right_hip"]), "parent": None},
            {"name": "hip", "loc": mid(joints["left_hip"], joints["right_hip"]), "parent": "root"},
            {"name": "torso", "loc": mid(joints["left_shoulder"], joints["right_shoulder"]), "parent": "hip"},
            {"name": "neck", "loc": joints["nose"], "parent": "torso"},
            {"name": "left_eye", "loc": joints["left_eye"], "parent": "neck"},
            {"name": "right_eye", "loc": joints["right_eye"], "parent": "neck"},
            {"name": "nose", "loc": joints["nose"], "parent": "neck"},
            {"name": "face_left", "loc": joints["face_left"], "parent": "neck"},
            {"name": "face_right", "loc": joints["face_right"], "parent": "neck"},
            {"name": "left_shoulder", "loc": joints["left_shoulder"], "parent": "torso"},
            {"name": "left_elbow", "loc": joints["left_elbow"], "parent": "left_shoulder"},
            {"name": "left_hand", "loc": joints["left_hand"], "parent": "left_elbow"},
            {"name": "right_shoulder", "loc": joints["right_shoulder"], "parent": "torso"},
            {"name": "right_elbow", "loc": joints["right_elbow"], "parent": "right_shoulder"},
            {"name": "right_hand", "loc": joints["right_hand"], "parent": "right_elbow"},
            {"name": "left_hip", "loc": joints["left_hip"], "parent": "root"},
            {"name": "left_knee", "loc": joints["left_knee"], "parent": "left_hip"},
            {"name": "left_foot", "loc": joints["left_foot"], "parent": "left_knee"},
            {"name": "right_hip", "loc": joints["right_hip"], "parent": "root"},
            {"name": "right_knee", "loc": joints["right_knee"], "parent": "right_hip"},
            {"name": "right_foot", "loc": joints["right_foot"], "parent": "right_knee"},
        ]
        for j in self.skeleton:
            j["loc"] = [int(round(j["loc"][0])), int(round(j["loc"][1]))]

    def draw_overlay(self):
        overlay = self.image.copy()
        for j in self.skeleton:
            x, y = j["loc"]
            cv2.circle(overlay, (x, y), 4, (0, 0, 255), -1) # Red dots
            if j["parent"]:
                parent = next((p for p in self.skeleton if p["name"] == j["parent"]), None)
                if parent:
                    px, py = parent["loc"]
                    cv2.line(overlay, (x, y), (px, py), (0, 255, 0), 2) # Green lines
        out_path = os.path.join(self.output_dir, "joint_overlay.png")
        cv2.imwrite(out_path, overlay)
        print(f"✅ Saved overlay: {out_path}")

    def save_yaml(self):
        data = {"height": int(self.height), "width": int(self.width), "skeleton": self.skeleton}
        out_path = os.path.join(self.output_dir, "char_cfg.yaml")
        with open(out_path, "w") as f:
            yaml.dump(data, f, sort_keys=False)
        print(f"✅ Saved YAML: {out_path}")

    # --- Adapter method ---
    
    def get_bones_for_export(self):
        """Converts self.skeleton (YOLO) to 'bones' format (Script 1)."""
        bones = []
        for joint in self.skeleton:
            bones.append({
                "name": joint["name"],
                "parent": joint.get("parent"),
                "x": int(joint["loc"][0]),
                "y": int(joint["loc"][1])
            })
        return bones

    # --- *** UPDATED METHOD *** ---
    
    def save_unity_json_files(self, points, simplices, uv_coords, bones, weights, root_loc):
        """
        Saves mesh and skeleton data in a JSON format compatible with the
        provided Unity C# importer script.
        
        ** UPDATED ** to accept a root_loc to center and flip the data.
        """
        print("Saving Unity JSON files...")
        
        # --- *** NEW: CENTER AND FLIP *** ---
        origin_x, origin_y = root_loc
        
        # --- 1. Create bone name to index mapping ---
        bone_name_to_index = {b["name"]: i for i, b in enumerate(bones)}
        
        # --- 2. Format MeshData ---
        
        # Convert weights to the Unity C# script's format
        bone_weights_list = []
        for wlist in weights:
            bw = {
                "boneIndex0": 0, "boneIndex1": 0, "boneIndex2": 0, "boneIndex3": 0,
                "weight0": 0.0, "weight1": 0.0, "weight2": 0.0, "weight3": 0.0
            }
            # Sum weights for normalization
            total_weight = sum(w for _, w in wlist[:4] if w > 0)
            if total_weight == 0:
                total_weight = 1.0 # Avoid division by zero
                
            for idx, (bone_name, weight) in enumerate(wlist[:4]):
                if bone_name in bone_name_to_index:
                    bw[f"boneIndex{idx}"] = bone_name_to_index[bone_name]
                    bw[f"weight{idx}"] = float(weight) / total_weight
            bone_weights_list.append(bw)
        
        # Create MeshData dictionary
        mesh_data = {
            # --- *** NEW: CENTER AND FLIP *** ---
            # Vertices: [x, -y, 0.0] (Flipped and Centered)
            "vertices": [[float(p[0] - origin_x), float(-(p[1] - origin_y)), 0.0] for p in points],
            "triangles": [[int(s[0]), int(s[1]), int(s[2])] for s in simplices],
            "uvs": [[float(uv[0]), float(uv[1])] for uv in uv_coords],
            "boneWeights": bone_weights_list
        }
        
        mesh_json_path = os.path.join(self.output_dir, "mesh_data.json")
        with open(mesh_json_path, "w") as f:
            json.dump(mesh_data, f, indent=2)
        print(f"✅ Saved MeshData JSON: {mesh_json_path}")
    
        # --- 3. Format SkeletonData ---
        
        bone_data_list = []
        bone_lookup = {b["name"]: b for b in bones}
        
        for i, b in enumerate(bones):
            # --- *** NEW: CENTER AND FLIP *** ---
            # Calculate LOCAL position relative to parent
            if b.get("parent") is None:
                # The Root bone is now AT the origin
                local_pos = {"x": 0.0, "y": 0.0, "z": 0.0}
            else:
                parent_bone = bone_lookup.get(b["parent"])
                if parent_bone:
                    # Calculate local offset and flip Y
                    local_pos = {
                        "x": float(b["x"] - parent_bone["x"]),
                        "y": float(-(b["y"] - parent_bone["y"])), # Flipped Y
                        "z": 0.0
                    }
                else:
                    # Fallback (shouldn't happen), flip and center
                    local_pos = {"x": float(b["x"] - origin_x), "y": float(-(b["y"] - origin_y)), "z": 0.0}
            
            bone_data_list.append({
                "name": b["name"],
                "parent": b.get("parent"),
                "index": i,
                "localPosition": local_pos
            })
        
        # Create bind poses (4x4 identity matrices), one for each bone
        bind_poses_list = []
        identity_pose = {
            "m00": 1.0, "m01": 0.0, "m02": 0.0, "m03": 0.0,
            "m10": 0.0, "m11": 1.0, "m12": 0.0, "m13": 0.0,
            "m20": 0.0, "m21": 0.0, "m22": 1.0, "m23": 0.0,
            "m30": 0.0, "m31": 0.0, "m32": 0.0, "m33": 1.0
        }
        for _ in bones:
            bind_poses_list.append(identity_pose)
        
        skeleton_data = {
            "bones": bone_data_list,
            "bindPoses": bind_poses_list
        }
        
        skeleton_json_path = os.path.join(self.output_dir, "skeleton_data.json")
        with open(skeleton_json_path, "w") as f:
            json.dump(skeleton_data, f, indent=2)
        print(f"✅ Saved SkeletonData JSON: {skeleton_json_path}")


    # --- *** UPDATED *** Combined Run Pipeline ---
    
    def run(self):
        # 1. Run YOLO pose detection
        if not self.detect_pose():
            print("❌ Pose not detected. Aborting.")
            return

        # 2. Create foreground mask and save original image
        if not self.create_mask_and_texture():
            print("❌ Mask creation failed. Aborting.")
            return
        
        # 3. Build skeleton hierarchy from YOLO points
        self.build_skeleton()

        # --- *** NEW: GET ORIGIN *** ---
        if not self.skeleton:
            print("❌ Skeleton not built. Aborting.")
            return
        # Get the root bone's location (the first bone in the list)
        root_loc = self.skeleton[0]["loc"]

        # 4. Get points for triangulation from mask
        contour = extract_contours(self.mask)
        interior = sample_interior_points(self.mask, n_points=2000)
        if contour.shape[0] > 0:
            pts = np.vstack([contour, interior])
        else:
            pts = interior
            
        if len(pts) < 3:
            print("❌ Not enough points for triangulation. Aborting.")
            return
            
        # 5. Build mesh
        tri = build_delaunay(pts)
        if tri is None:
            print("❌ Delaunay triangulation failed. Aborting.")
            return
        
        points = pts.astype(float)
        simplices_original = tri.simplices.astype(int)

        # --- *** NEW: REVERSE TRIANGLE WINDING ORDER *** ---
        # Swap the 2nd and 3rd vertex of each triangle to flip the face
        simplices = simplices_original[:, [0, 2, 1]]

        # 6. Get bones in the right format
        bones = self.get_bones_for_export()

        # 7. Compute skin weights
        falloff = self.args.get("falloff", 40.0)
        weights = compute_vertex_weights(points, bones, falloff=falloff, max_influences=4)

        # 8. Compute UVs and get cropped texture
        ys, xs = np.nonzero(self.mask)
        minx, miny, maxx, maxy = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        pad = 4
        minx = max(0, minx-pad); miny = max(0, miny-pad)
        maxx = min(self.width-1, maxx+pad); maxy = min(self.height-1, maxy+pad)
        
        # Crop texture (need to convert from BGR to RGB for imageio)
        crop_bgr = self.image[miny:maxy+1, minx:maxx+1]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        uv_coords = []
        denom_x = max(1.0, (maxx - minx))
        denom_y = max(1.0, (maxy - miny))
        for (x,y) in points:
            u = (x - minx) / denom_x
            v = (y - miny) / denom_y
            uv_coords.append((u, 1.0 - v)) # Flip V for UV coords

        # 9. Export the final .glb file (pass root_loc)
        glb_path = os.path.join(self.output_dir, "character.glb")
        export_gltf(points, simplices, uv_coords, bones, weights, crop_rgb, out_fn=glb_path, root_loc=root_loc)

        # 10. Save Unity JSON files (pass root_loc)
        self.save_unity_json_files(points, simplices, uv_coords, bones, weights, root_loc=root_loc)

        # 11. Save debug files
        self.draw_overlay()
        self.save_yaml()
        
        print(f"--- Successfully processed {self.image_path} ---")
        print(f"Vertices: {len(points)}, Triangles: {len(simplices)}, Bones: {len(bones)}")


# ---------------------------------------------------------------------
# --- EXECUTION BLOCK: Handle Single File or Directory ---
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid 2D-to-3D Rigging Pipeline")
    parser.add_argument("--input", required=True, help="Input file or directory of images")
    parser.add_argument("--output", required=True, help="Root output directory")
    parser.add_argument("--falloff", type=float, default=40.0, help="Sigma for Gaussian skin weights (pixels)")
    args = parser.parse_args()

    cli_args = {"falloff": args.falloff}
    os.makedirs(args.output, exist_ok=True)
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

    # --- Logic to determine if input is a single file or a directory ---

    input_paths = []
    
    if os.path.isfile(args.input):
        # Case 1: Single File Input
        input_paths.append(args.input)
    elif os.path.isdir(args.input):
        # Case 2: Directory Input (Batch Mode)
        for filename in os.listdir(args.input):
            if filename.lower().endswith(valid_exts):
                input_paths.append(os.path.join(args.input, filename))
    else:
        print(f"❌ Error: Input path '{args.input}' is neither a file nor a directory.")
        exit(1)

    # --- Process all identified input paths ---
    
    if not input_paths:
        print(f"⚠ Warning: No valid images found in '{args.input}'.")

    for input_path in input_paths:
        
        # Determine unique output folder name from the input filename
        filename = os.path.basename(input_path)
        name_no_ext = os.path.splitext(filename)[0]

        output_dir = os.path.join(args.output, name_no_ext)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n🔹 Processing: {filename}")
        try:
            extractor = CharacterExtractorYOLO(input_path, output_dir, args=cli_args)
            extractor.run()
        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")
            import traceback
            traceback.print_exc()

    print("\n✅ All done! Results saved in:", args.output)