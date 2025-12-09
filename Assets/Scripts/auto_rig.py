#!/usr/bin/env python3
"""
mesh_generator_v2_modified.py

End-to-end pipeline:
- Simple baseline pose detection
- Advanced mesh generation with shapely + grid sampling (from mesh_generator.py)
- Exports: character_tex.png, mesh_data.json, skeleton_data.json, mask/debug assets

Usage:
    python mesh_generator_v2_modified.py --input <image> --output <output_dir> [--falloff 40.0]
"""
import argparse
import json
import math
import os
from pathlib import Path

import cv2
import imageio
import numpy as np
import yaml
import torch
from scipy import ndimage as ndi
from scipy.spatial import Delaunay
from shapely import geometry
from skimage import color, filters, measure, morphology, util
from typing import List, Dict, Tuple

from simplebaseline_pipeline import (
    load_pretrained_simplebaseline,
    get_keypoints_from_heatmaps,
    improve_face_detection,
)
# NOTE: PyGLTFLIB imports are no longer needed but kept for minimal change principle.
# They will not be used in the export_gltf function, which is removed.
from pygltflib import (
    GLTF2,
    Scene,
    Node,
    Mesh,
    Buffer,
    BufferView,
    Accessor,
    Asset,
    Skin,
    Image as GLTFImage,
    Texture as GLTFTexture,
    TextureInfo,
    Material,
    Primitive,
    Sampler,
    ARRAY_BUFFER,
    ELEMENT_ARRAY_BUFFER,
)

COMP_FLOAT = 5126
COMP_UNSIGNED_SHORT = 5123
COMP_UNSIGNED_INT = 5125
COCO_PAIRS = [
    (5,7),(7,9),     # left arm
    (6,8),(8,10),    # right arm
    (11,13),(13,15), # left leg
    (12,14),(14,16), # right leg
    (5,6),           # shoulders
    (11,12),         # hips
    (5,11),(6,12)    # torso
]

def extract_contours(mask):
    contours = measure.find_contours(mask.astype(np.uint8), level=0.5)
    if len(contours) == 0:
        return np.empty((0, 2))
    longest = max(contours, key=lambda c: c.shape[0])
    contour_xy = np.fliplr(longest)
    return contour_xy


def sample_interior_points(mask, n_points=2000):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((0, 2))
    idx = np.random.choice(len(xs), size=min(n_points, len(xs)), replace=False)
    pts = np.vstack([xs[idx], ys[idx]]).T
    return pts

def center_points_and_bones(points, bones, root_name="root"):
    """
    Shift both vertices (points) and bones so that the root bone is at (0, 0).

    points: np.ndarray of shape (N, 2) or (N, 3)
    bones:  list of dicts with keys at least: 'name', 'x', 'y'
    """

    if not bones:
        return points, bones

    # 1) Find root bone index
    root_idx = None
    for i, b in enumerate(bones):
        if b.get("name") == root_name:
            root_idx = i
            break
    if root_idx is None:
        # fall back: first bone with parent=None, or bone 0
        for i, b in enumerate(bones):
            if b.get("parent") in (None, "", "root"):
                root_idx = i
                break
    if root_idx is None:
        root_idx = 0

    root_x = float(bones[root_idx]["x"])
    root_y = float(bones[root_idx]["y"])

    # 2) Shift points
    pts = np.asarray(points, dtype=float)
    pts[:, 0] -= root_x
    pts[:, 1] -= root_y

    # 3) Shift bones copy
    centered_bones = []
    for b in bones:
        b2 = dict(b)
        b2["x"] = float(b2["x"]) - root_x
        b2["y"] = float(b2["y"]) - root_y
        centered_bones.append(b2)

    return pts, centered_bones

def compute_vertex_weights(
    points,
    bones,
    falloff=40.0,
    max_influences=4,
    max_distance_factor=3.0,   # how many sigmas away we still allow influence
    min_rel_weight=0.1         # keep weights >= 10% of the strongest one
):
    """
    Compute per-vertex bone weights using a Gaussian falloff.

    - points: (N, 2) vertex positions
    - bones:  list of {'name': str, 'x': float, 'y': float}
    """

    if len(bones) == 0:
        return [[] for _ in range(len(points))]

    points = np.asarray(points, dtype=float).reshape(-1, 2)
    bone_pts = np.array([[b["x"], b["y"]] for b in bones], dtype=float)
    N = len(points)

    # Squared distances vertex <-> bone
    d2 = np.sum((points[:, None, :] - bone_pts[None, :, :]) ** 2, axis=2)

    sigma = float(falloff)
    raw = np.exp(-d2 / (2.0 * sigma * sigma))  # Gaussian weights (not yet normalized)

    # Distance cutoff: ignore bones farther than max_distance_factor * sigma
    max_d2 = (max_distance_factor * sigma) ** 2
    raw[d2 > max_d2] = 0.0

    weights = []
    for i in range(N):
        row = raw[i].copy()

        total = row.sum()
        if total <= 0.0:
            # 🔁 Fallback: bind this vertex to the NEAREST bone (by distance),
            # not always to root. This lets far fingers still follow the hand.
            nearest = int(np.argmin(d2[i]))
            weights.append([(bones[nearest]["name"], 1.0)])
            continue

        # Normalize
        row /= total

        # Keep only reasonably strong influences
        max_w = row.max()
        keep_mask = row >= max_w * float(min_rel_weight)
        keep_indices = np.where(keep_mask)[0]

        if keep_indices.size == 0:
            # Extremely conservative fallback: just nearest bone
            nearest = int(np.argmax(row))
            keep_indices = np.array([nearest])

        # Sort kept bones by weight descending and cap to max_influences
        keep_indices = keep_indices[np.argsort(row[keep_indices])[::-1][:max_influences]]

        wlist = [(bones[idx]["name"], float(row[idx])) for idx in keep_indices]
        weights.append(wlist)

    return weights


# --- REMOVED export_gltf function ---

# --- REMOVED save_yaml function ---
class CharacterExtractorSimpleBaseline:
    def __init__(self, image_path, output_dir="output", falloff=40.0):
        self.image_path = image_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.falloff = falloff

        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Could not read image {image_path}")

        # Ensure we always have a 3-channel BGR image
        if len(self.image.shape) == 2 or self.image.shape[2] == 1:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        elif self.image.shape[2] == 4:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGRA2BGR)

        self.height, self.width = self.image.shape[:2]

        # --- SimpleBaseline ---
        self.sb_model = load_pretrained_simplebaseline()
        if self.sb_model is None:
            raise RuntimeError(
                "Failed to load SimpleBaseline model. "
                "Make sure pretrained_weights/pose_resnet_50_256x192.pth.tar exists."
            )

        # internal state
        self.keypoints = None  # np.array shape [17, 2]
        self.skeleton = []
        self.mask = None

    def detect_pose(self):
        """
        Run SimpleBaseline on self.image and fill self.keypoints as
        a [17, 2] array in (x, y) image coordinates.
        """
        orig = self.image
        orig_h, orig_w = orig.shape[:2]

        # Preprocess to 192x256 like in simplebaseline_pipeline.py
        inp = cv2.resize(orig, (192, 256))
        inp_rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp_rgb = (inp_rgb - mean) / std

        tensor = torch.from_numpy(inp_rgb).permute(2, 0, 1).unsqueeze(0).float()

        print("[INFO] Running SimpleBaseline pose estimation...")
        with torch.no_grad():
            heatmaps = self.sb_model(tensor)

        # Decode heatmaps → 17 keypoints in heatmap space
        kpts = get_keypoints_from_heatmaps(heatmaps)
        hm_H, hm_W = heatmaps.shape[2], heatmaps.shape[3]

        # Scale back to original image size
        final_kpts = []
        for (hx, hy) in kpts:
            x = int(hx / hm_W * orig_w)
            y = int(hy / hm_H * orig_h)
            final_kpts.append((x, y))

        # Optional face refinement (same logic as simplebaseline_pipeline)
        try:
            refinement_result = improve_face_detection(orig, final_kpts, self.sb_model)
            if refinement_result is not None:
                refined_face_kpts, _, _ = refinement_result
                # Replace the first 5 (face) keypoints
                final_kpts[:5] = refined_face_kpts
                print("[INFO] Face keypoints refined by SimpleBaseline.")
            else:
                print("[WARN] Face refinement skipped; using original face keypoints.")
        except Exception as e:
            print(f"[WARN] Error during face refinement: {e}")
            print("[WARN] Using original SimpleBaseline keypoints.")

        # Store as numpy array in the same format as before
        self.keypoints = np.array(final_kpts, dtype=np.float32)
        print(f"[INFO] Final keypoints (first 5): {self.keypoints[:5]}")
        return True

    def create_mask_and_texture(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Using a simple threshold for foreground mask
        _, self.mask = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        kernel = np.ones((3, 3), np.uint8)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel, iterations=1)

        if self.mask.sum() == 0:
            print("⚠ Foreground mask is empty. Aborting.")
            return False

        mask_path = os.path.join(self.output_dir, "mask.png")
        cv2.imwrite(mask_path, self.mask)
        print(f"Success: Saved mask: {mask_path}")
        return True

    def get_face_anchor_points(self):
        if self.keypoints is None:
            return None

        kp = self.keypoints
        nose = kp[0]
        left_eye = kp[1]
        right_eye = kp[2]
        left_ear = kp[3] if len(kp) > 3 else None
        right_ear = kp[4] if len(kp) > 4 else None

        eye_mid_y = (left_eye[1] + right_eye[1]) / 2
        eye_span = abs(left_eye[0] - right_eye[0])

        if (
            left_ear is None
            or right_ear is None
            or np.any(np.isnan(left_ear))
            or np.any(np.isnan(right_ear))
        ):
            face_left = [left_eye[0] - 2.0 * eye_span, eye_mid_y]
            face_right = [right_eye[0] + 2.0 * eye_span, eye_mid_y]
        else:
            face_left = left_ear
            face_right = right_ear

        return {
            "nose": nose,
            "left_eye": left_eye,
            "right_eye": right_eye,
            "face_left": face_left,
            "face_right": face_right,
        }

    def build_skeleton(self):
        if self.keypoints is None:
            return

        kp = self.keypoints

        def xy(i):
            return [float(kp[i][0]), float(kp[i][1])]

        joints = {
            "nose": xy(0),
            "left_eye": xy(1),
            "right_eye": xy(2),
            "left_ear": xy(3),
            "right_ear": xy(4),
            "left_shoulder": xy(5),
            "right_shoulder": xy(6),
            "left_elbow": xy(7),
            "right_elbow": xy(8),
            "left_hand": xy(9),
            "right_hand": xy(10),
            "left_hip": xy(11),
            "right_hip": xy(12),
            "left_knee": xy(13),
            "right_knee": xy(14),
            "left_foot": xy(15),
            "right_foot": xy(16),
        }

        face_points = self.get_face_anchor_points()
        if face_points:
            for k, v in face_points.items():
                joints[k] = [float(v[0]), float(v[1])]

        def mid(a, b):
            return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]

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

    def draw_skeleton(self, image, kpts):
        """Draw skeleton with keypoints and connections"""
        # Draw lines (skeleton connections)
        pts: list[tuple[int, int] | None] = []
        for i, (x, y) in enumerate(kpts):
            if math.isnan(x) or math.isnan(y):
                pts.append(None)
                continue
            px = int(round(float(x)))
            py = int(round(float(y)))
            pts.append((px, py))

        for (i, j) in COCO_PAIRS:
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Draw keypoints (circles)
        for pt in pts:
            cv2.circle(image, pt, 5, (0, 0, 255), -1)

        return image

    def draw_overlay(self):
        skel_img = self.image.copy()
        skel_img = self.draw_skeleton(skel_img, self.keypoints)
    
        out_path = os.path.join(self.output_dir, "joint_overlay.png")
        cv2.imwrite(out_path, skel_img)
        print(f"[INFO] Success: Saved skeleton overlay to {out_path}")

    def get_bones_for_export(self):
        return [
            {"name": joint["name"], "parent": joint.get("parent"),
             "x": int(joint["loc"][0]), "y": int(joint["loc"][1])}
            for joint in self.skeleton
        ]

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        mask_array = np.array(self.mask)
        try:
            contours = measure.find_contours(mask_array, 128)
        except Exception as e:
            print(f'Error finding contours: {e}')
            raise
        if len(contours) > 1:
            print(f'Found {len(contours)} contours, using largest')
            contours.sort(key=len, reverse=True)
        if len(contours) == 0:
            raise ValueError("No contours found in mask")
        main_contour = contours[0]
        outside_vertices = measure.approximate_polygon(main_contour, tolerance=0.25)
        character_outline = geometry.Polygon([(pt[1], pt[0]) for pt in main_contour])

        inside_vertices_xy = []
        grid_resolution = 40
        _x = np.linspace(0, self.width, grid_resolution)
        _y = np.linspace(0, self.height, grid_resolution)
        xv, yv = np.meshgrid(_x, _y)
        for x, y in zip(xv.flatten(), yv.flatten()):
            if character_outline.contains(geometry.Point(x, y)):
                inside_vertices_xy.append([x, y])

        outside_vertices_xy = np.array([[pt[1], pt[0]] for pt in outside_vertices])
        if len(inside_vertices_xy) > 0:
            inside_vertices = np.array(inside_vertices_xy)
            vertices = np.concatenate([outside_vertices_xy, inside_vertices]).astype(np.float32)
        else:
            vertices = outside_vertices_xy.astype(np.float32)

        tri = Delaunay(vertices)
        valid_triangles = []
        for simplex in tri.simplices:
            tri_vertices = vertices[simplex]
            tri_centroid = geometry.Point(np.mean(tri_vertices, 0))
            if character_outline.contains(tri_centroid):
                valid_triangles.append(simplex)

        triangles = np.array(valid_triangles)
        print(f"Success: Generated mesh: {len(vertices)} vertices, {len(triangles)} triangles")
        return vertices, triangles

    # def save_unity_json_files(self, points, simplices, uv_coords, bones, weights):
    #     # 0) Make sure points is an array
    #     points = np.asarray(points, dtype=float)

    #     # 1) Flip Y for geometry & bones to go from image coords (y down)
    #     #    to Unity coords (y up). UVs are NOT touched.
    #     points_flipped = points.copy()
    #     points_flipped[:, 1] = -points_flipped[:, 1]

    #     bones_flipped = []
    #     for b in bones:
    #         bf = dict(b)
    #         bf["y"] = -float(bf["y"])
    #         bones_flipped.append(bf)

    #     # 1) Center points & bones around the root **only for geometry**
    #     points_centered, bones_centered = center_points_and_bones(points_flipped, bones_flipped)

    #     # 2) Build mesh_data.json from centered geometry
    #     vertices = [[float(p[0]), float(p[1]), 0.0] for p in points_centered]
    #     triangles = [[int(t[0]), int(t[1]), int(t[2])] for t in simplices]
    #     uvs = [[float(u), float(v)] for (u, v) in uv_coords]  # UVs stay as given

    #     bone_name_to_index = {b["name"]: i for i, b in enumerate(bones_centered)}

    #     bone_weights_list = []
    #     for vw in weights:
    #         idxs = [bone_name_to_index[name] for (name, _) in vw]
    #         ws = [w for (_, w) in vw]

    #         # pad to 4
    #         while len(idxs) < 4:
    #             idxs.append(0)
    #             ws.append(0.0)

    #         s = sum(ws)
    #         if s > 0:
    #             ws = [w / s for w in ws]

    #         bone_weights_list.append({
    #             "boneIndex0": int(idxs[0]),
    #             "boneIndex1": int(idxs[1]),
    #             "boneIndex2": int(idxs[2]),
    #             "boneIndex3": int(idxs[3]),
    #             "weight0": float(ws[0]),
    #             "weight1": float(ws[1]),
    #             "weight2": float(ws[2]),
    #             "weight3": float(ws[3]),
    #         })

    #     mesh_data = {
    #         "vertices": vertices,
    #         "triangles": triangles,
    #         "uvs": uvs,
    #         "boneWeights": bone_weights_list,
    #     }
    #     bones_out = []
    #     for i, b in enumerate(bones_centered):
    #         bones_out.append({
    #             "name": b["name"],
    #             "parent": b.get("parent"),
    #             "index": i,
    #             "localPosition": {"x": float(b["x"]), "y": float(b["y"]), "z": 0.0},
    #         })

    #     skeleton_data = {
    #         "bones": bones_out,
    #         "bindPoses": [],
    #     }

    #     mesh_json_path = os.path.join(self.output_dir, "mesh_data.json")
    #     with open(mesh_json_path, "w") as f:
    #         json.dump(mesh_data, f, indent=2)
    #     print(f"Success: Saved MeshData JSON: {mesh_json_path}")

    #     skeleton_json_path = os.path.join(self.output_dir, "skeleton_data.json")
    #     with open(skeleton_json_path, "w") as f:
    #         json.dump(skeleton_data, f, indent=2)
    #     print(f"Success: Saved SkeletonData JSON: {skeleton_json_path}")

    def save_unity_json_files(self, points, simplices, uv_coords, bones, weights):
        # 0) Make sure points is an array
        points = np.asarray(points, dtype=float)

        # 1) Flip Y for geometry & bones (image coords -> Unity coords)
        points_flipped = points.copy()
        points_flipped[:, 1] = -points_flipped[:, 1]

        bones_flipped = []
        for b in bones:
            bf = dict(b)
            bf["y"] = -float(bf["y"])
            bones_flipped.append(bf)

        # 1) Center points & bones around root (only for geometry)
        points_centered, bones_centered = center_points_and_bones(points_flipped, bones_flipped)

        # 2) Mesh data
        vertices = [[float(p[0]), float(p[1]), 0.0] for p in points_centered]
        triangles = [[int(t[0]), int(t[1]), int(t[2])] for t in simplices]
        uvs = [[float(u), float(v)] for (u, v) in uv_coords]

        bone_name_to_index = {b["name"]: i for i, b in enumerate(bones_centered)}

        bone_weights_list = []
        for vw in weights:
            idxs = [bone_name_to_index[name] for (name, _) in vw]
            ws   = [w for (_, w) in vw]
            
            while len(idxs) < 4:
                idxs.append(0)
                ws.append(0.0)

            s = sum(ws)
            if s > 0:
                ws = [w / s for w in ws]

            bone_weights_list.append({
                "boneIndex0": int(idxs[0]),
                "boneIndex1": int(idxs[1]),
                "boneIndex2": int(idxs[2]),
                "boneIndex3": int(idxs[3]),
                "weight0": float(ws[0]),
                "weight1": float(ws[1]),
                "weight2": float(ws[2]),
                "weight3": float(ws[3]),
            })

        mesh_data = {
            "vertices": vertices,
            "triangles": triangles,
            "uvs": uvs,
            "boneWeights": bone_weights_list,
        }

        bones_out = []
        for i, b in enumerate(bones_centered):
            bones_out.append({
                "name": b["name"],
                "parent": b.get("parent"),
                "index": i,
                "localPosition": {"x": float(b["x"]), "y": float(b["y"]), "z": 0.0},
            })

        skeleton_data = {"bones": bones_out, "bindPoses": []}

        mesh_json_path = os.path.join(self.output_dir, "mesh_data.json")
        with open(mesh_json_path, "w") as f:
            json.dump(mesh_data, f, indent=2)
        print(f"Success: Saved MeshData JSON: {mesh_json_path}")

        skeleton_json_path = os.path.join(self.output_dir, "skeleton_data.json")
        with open(skeleton_json_path, "w") as f:
            json.dump(skeleton_data, f, indent=2)
        print(f"Success: Saved SkeletonData JSON: {skeleton_json_path}")

    def run(self):
        print("\n" + "=" * 70)
        print("  HYBRID AUTO-RIGGER PIPELINE (Unity-Export) ".center(70, "="))
        print("=" * 70)

        print("\n[1/5] Detecting pose with SimpleBaseline...")
        if not self.detect_pose():
            print("Error: Pose not detected. Aborting.")
            return False

        print("\n[2/5] Creating foreground mask...")
        if not self.create_mask_and_texture():
            print("Error: Mask creation failed. Aborting.")
            return False

        print("\n[3/5] Building skeleton hierarchy...")
        self.build_skeleton()
        if not self.skeleton:
            print("Error: Skeleton not built. Aborting.")
            return False

        print("\n[4/5] Generating mesh with Delaunay triangulation...")
        try:
            points, simplices = self.generate_mesh()
        except Exception as e:
            print(f"Error: Mesh generation failed: {e}")
            return False

        points = np.asarray(points, dtype=float)

        print("\n[5/5] Calculating bone weights and exporting files...")
        bones = self.get_bones_for_export()
        weights = compute_vertex_weights(points, bones, falloff=self.falloff, max_influences=4)
        print(f"Success: Calculated weights for {len(weights)} vertices")

        # --- Calculate cropped UVs and Texture ---
        ys, xs = np.nonzero(self.mask)
        if len(xs) == 0:
            minx, miny, maxx, maxy = 0, 0, self.width - 1, self.height - 1
        else:
            minx, miny, maxx, maxy = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        pad = 4
        minx = max(0, minx - pad)
        miny = max(0, miny - pad)
        maxx = min(self.width - 1, maxx + pad)
        maxy = min(self.height - 1, maxy + pad)

        crop_bgr = self.image[miny:maxy + 1, minx:maxx + 1]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        uv_coords = []
        denom_x = max(1.0, (maxx - minx))
        denom_y = max(1.0, (maxy - miny))
        for (x, y) in points:
            u = (x - minx) / denom_x
            v = (y - miny) / denom_y
            uv_coords.append((u, 1.0 - v)) # Flip V for UV convention

        # --- Exporting ---
        
        # 1. Save JSON files
        mesh_points = np.array([[float(p[0]), float(p[1]), 0.0] for p in points])
        self.save_unity_json_files(mesh_points, simplices, uv_coords, bones, weights)

        # 2. Save Texture
        character_tex_path = os.path.join(self.output_dir, "character_tex.png")
        imageio.imwrite(character_tex_path, crop_rgb)
        print(f"Success: Saved cropped texture: {character_tex_path}")

        # 3. Save Debug Overlay
        self.draw_overlay()
        

        print("\n" + "=" * 70)
        print("Success: PIPELINE COMPLETE!".center(70))
        print("=" * 70)
        print(f"\nOutput directory: {self.output_dir}")
        print("Generated files:")
        print("  • character_tex.png")
        print("  • mesh_data.json")
        print("  • skeleton_data.json")
        print("  • mask.png")
        print("  • joint_overlay.png")
        print("\n")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid Auto-Rigger: SimpleBaseline pose detection + shapely mesh for Unity/GLB"
    )
    parser.add_argument("--input", required=True, help="Input character drawing (PNG/JPG)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--falloff", type=float, default=40.0, help="Gaussian falloff for bone weights (pixels)")
    args = parser.parse_args()

    try:
        rigger = CharacterExtractorSimpleBaseline(
            image_path=args.input,
            output_dir=args.output,
            falloff=args.falloff
        )
        rigger.run()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)