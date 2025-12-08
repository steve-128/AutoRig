#!/usr/bin/env python3
"""
mesh_generator_v2_modified.py

End-to-end pipeline:
- YOLO pose detection (from test_main4.py)
- Advanced mesh generation with shapely + grid sampling (from mesh_generator.py)
- Exports: character_tex.png, mesh_data.json, skeleton_data.json, mask/debug assets, CharacterImporter.cs

Usage:
    python mesh_generator_v2_modified.py --input <image> --output <output_dir> [--falloff 40.0]
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import imageio
import numpy as np
import yaml
from scipy import ndimage as ndi
from scipy.spatial import Delaunay
from shapely import geometry
from skimage import color, filters, measure, morphology, util
from ultralytics import YOLO
from typing import List, Dict, Tuple

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


def compute_vertex_weights(points, bones, falloff=40.0, max_influences=4):
    if len(bones) == 0:
        return [[] for _ in range(len(points))]
    bone_pts = np.array([[b["x"], b["y"]] for b in bones])
    N = len(points)

    d2 = np.sum((points[:, None, :] - bone_pts[None, :, :]) ** 2, axis=2)
    sigma = float(falloff)
    raw = np.exp(-d2 / (2 * sigma * sigma))
    raw[raw < 1e-6] = 0.0

    s = raw.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    norm = raw / s

    weights = []
    for i in range(N):
        row = norm[i]
        top_idx = np.argsort(row)[::-1][:max_influences]
        wlist = []
        for idx in top_idx:
            if row[idx] <= 0:
                continue
            # Keep only the name and the weight (not the index)
            wlist.append((bones[idx]["name"], float(row[idx])))
        weights.append(wlist)
    return weights


# --- REMOVED export_gltf function ---


class CharacterExtractorYOLO:
    def __init__(self, image_path, output_dir="output", falloff=40.0):
        self.image_path = image_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.falloff = falloff

        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Could not read image {image_path}")

        if len(self.image.shape) == 2 or self.image.shape[2] == 1:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        elif self.image.shape[2] == 4:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGRA2BGR)

        self.height, self.width = self.image.shape[:2]
        self.model = YOLO("yolov8n-pose.pt")
        self.keypoints = None
        self.skeleton = []
        self.mask = None

    def detect_pose(self):
        results = self.model(self.image, verbose=False)
        if not results or len(results[0].keypoints.data) == 0:
            print("⚠ No pose detected.")
            return False
        self.keypoints = results[0].keypoints.data[0].cpu().numpy()[:, :2]
        return True

    def create_mask_and_texture(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Using a simple threshold for foreground mask
        _, self.mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel, iterations=1)

        if self.mask.sum() == 0:
            print("⚠ Foreground mask is empty. Aborting.")
            return False

        mask_path = os.path.join(self.output_dir, "mask.png")
        cv2.imwrite(mask_path, self.mask)
        print(f"✅ Saved mask: {mask_path}")
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
        if left_ear is None or right_ear is None or np.any(np.isnan(left_ear)) or np.any(np.isnan(right_ear)):
            face_left = [left_eye[0] - 2.0 * eye_span, eye_mid_y]
            face_right = [right_eye[0] + 2.0 * eye_span, eye_mid_y]
        else:
            face_left = left_ear
            face_right = right_ear
        return {"nose": nose, "left_eye": left_eye, "right_eye": right_eye,
                "face_left": face_left, "face_right": face_right}

    def build_skeleton(self):
        if self.keypoints is None:
            return
        kp = self.keypoints

        def xy(i):
            return [float(kp[i][0]), float(kp[i][1])]

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

    def draw_overlay(self):
        overlay = self.image.copy()
        for j in self.skeleton:
            x, y = j["loc"]
            cv2.circle(overlay, (x, y), 4, (0, 0, 255), -1)
            if j["parent"]:
                parent = next((p for p in self.skeleton if p["name"] == j["parent"]), None)
                if parent:
                    px, py = parent["loc"]
                    cv2.line(overlay, (x, y), (px, py), (0, 255, 0), 2)
        out_path = os.path.join(self.output_dir, "joint_overlay.png")
        cv2.imwrite(out_path, overlay)
        print(f"✅ Saved overlay: {out_path}")

    # --- REMOVED save_yaml function ---

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
        print(f"✓ Generated mesh: {len(vertices)} vertices, {len(triangles)} triangles")
        return vertices, triangles

    def save_unity_json_files(self, points, simplices, uv_coords, bones, weights):
        bone_name_to_index = {b["name"]: i for i, b in enumerate(bones)}

        bone_weights_list = []
        for wlist in weights:
            bw = {
                "boneIndex0": 0, "boneIndex1": 0, "boneIndex2": 0, "boneIndex3": 0,
                "weight0": 0.0, "weight1": 0.0, "weight2": 0.0, "weight3": 0.0
            }
            # Recalculate total_weight for the top 4 influences to ensure they sum to 1.0
            top_weights = [w for _, w in wlist[:4] if w > 0]
            total_weight = sum(top_weights)
            if total_weight == 0:
                # Fallback in case weights are zeroed out (shouldn't happen with Gaussian)
                total_weight = 1.0

            for idx, (bone_name, weight) in enumerate(wlist[:4]):
                if bone_name in bone_name_to_index:
                    bw[f"boneIndex{idx}"] = bone_name_to_index[bone_name]
                    bw[f"weight{idx}"] = float(weight) / total_weight
            bone_weights_list.append(bw)

        # NOTE: Vertices are scaled/translated in the C# script, but here they are exported as
        # raw screen coordinates with a Y-flip, matching the skeleton position format.
        mesh_data = {
            "vertices": [[float(p[0]), float(-p[1]), 0.0] for p in points],
            # Triangles are flattened in Unity, but this format is easier to read in JSON
            "triangles": [[int(s[0]), int(s[1]), int(s[2])] for s in simplices],
            "uvs": [[float(uv[0]), float(uv[1])] for uv in uv_coords],
            "boneWeights": bone_weights_list
        }
        mesh_json_path = os.path.join(self.output_dir, "mesh_data.json")
        with open(mesh_json_path, "w") as f:
            json.dump(mesh_data, f, indent=2)
        print(f"✅ Saved MeshData JSON: {mesh_json_path}")

        bone_data_list = []
        bind_poses_list = []
        for i, b in enumerate(bones):
            # Calculate local position relative to parent, ensuring Y-flip for Unity
            if b.get("parent") is None:
                local_pos = {"x": float(b["x"]), "y": float(-b["y"]), "z": 0.0}
            else:
                parent_bone = next((p for p in bones if p["name"] == b["parent"]), None)
                if parent_bone:
                    local_pos = {
                        "x": float(b["x"] - parent_bone["x"]),
                        "y": float(-(b["y"] - parent_bone["y"])), # Flip Y-axis difference
                        "z": 0.0
                    }
                else:
                    local_pos = {"x": float(b["x"]), "y": float(-b["y"]), "z": 0.0}

            bone_data_list.append({
                "name": b["name"],
                "parent": b.get("parent"),
                "index": i,
                "localPosition": local_pos
            })

            # Bind pose: Identity matrix is used here for simplicity,
            # implying the C# script will calculate world space transforms or
            # rely entirely on the local position.
            bind_pose = {
                "m00": 1.0, "m01": 0.0, "m02": 0.0, "m03": 0.0,
                "m10": 0.0, "m11": 1.0, "m12": 0.0, "m13": 0.0,
                "m20": 0.0, "m21": 0.0, "m22": 1.0, "m23": 0.0,
                "m30": 0.0, "m31": 0.0, "m32": 0.0, "m33": 1.0
            }
            bind_poses_list.append(bind_pose)

        skeleton_data = {
            "bones": bone_data_list,
            "bindPoses": bind_poses_list
        }
        skeleton_json_path = os.path.join(self.output_dir, "skeleton_data.json")
        with open(skeleton_json_path, "w") as f:
            json.dump(skeleton_data, f, indent=2)
        print(f"✅ Saved SkeletonData JSON: {skeleton_json_path}")

    def _generate_unity_import_script(self, output_path: Path):
        """Generate Unity C# script for importing with SkinnedMeshRenderer (Copied from Script 1)."""
        script = '''using UnityEngine;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json;

public class CharacterImporter : MonoBehaviour
{
    [System.Serializable]
    public class MeshData
    {
        public List<List<float>> vertices;
        public List<List<int>> triangles;
        public List<List<float>> uvs;
        public List<BoneWeightData> boneWeights;
    }

    [System.Serializable]
    public class BoneWeightData
    {
        public int boneIndex0;
        public int boneIndex1;
        public int boneIndex2;
        public int boneIndex3;
        public float weight0;
        public float weight1;
        public float weight2;
        public float weight3;
    }

    [System.Serializable]
    public class SkeletonData
    {
        public List<BoneData> bones;
        public List<BindPoseData> bindPoses;
    }

    [System.Serializable]
    public class BoneData
    {
        public string name;
        public string parent;
        public int index;
        public Vector3Data localPosition;
    }

    [System.Serializable]
    public class Vector3Data
    {
        public float x;
        public float y;
        public float z;
    }

    [System.Serializable]
    public class BindPoseData
    {
        public float m00, m01, m02, m03;
        public float m10, m11, m12, m13;
        public float m20, m21, m22, m23;
        public float m30, m31, m32, m33;
    }

    public Texture2D characterTexture;
    public TextAsset meshDataFile;
    public TextAsset skeletonDataFile;

    void Start()
    {
        ImportCharacter();
    }

    void ImportCharacter()
    {
        // Parse JSON data
        MeshData meshData = JsonConvert.DeserializeObject<MeshData>(meshDataFile.text);
        SkeletonData skeletonData = JsonConvert.DeserializeObject<SkeletonData>(skeletonDataFile.text);

        // Create root GameObject
        GameObject characterRoot = new GameObject("Character");
        characterRoot.transform.position = transform.position;

        // Create bone GameObjects
        Dictionary<string, Transform> boneTransforms = new Dictionary<string, Transform>();
        Transform[] boneArray = new Transform[skeletonData.bones.Count];
        
        // First pass: create all bone objects
        foreach (var boneData in skeletonData.bones)
        {
            GameObject boneObj = new GameObject(boneData.name);
            boneObj.transform.position = new Vector3(
                boneData.localPosition.x,
                boneData.localPosition.y,
                boneData.localPosition.z
            );
            
            boneTransforms[boneData.name] = boneObj.transform;
            boneArray[boneData.index] = boneObj.transform;
        }
        
        // Second pass: set up hierarchy
        foreach (var boneData in skeletonData.bones)
        {
            if (boneData.parent != null && boneTransforms.ContainsKey(boneData.parent))
            {
                boneTransforms[boneData.name].SetParent(boneTransforms[boneData.parent]);
            }
            else
            {
                boneTransforms[boneData.name].SetParent(characterRoot.transform);
            }
            
            // Reset local position for proper hierarchy
            boneTransforms[boneData.name].localPosition = Vector3.zero;
        }

        // Create mesh
        Mesh mesh = new Mesh();
        mesh.name = "CharacterMesh";
        
        // Set vertices
        Vector3[] vertices = new Vector3[meshData.vertices.Count];
        for (int i = 0; i < meshData.vertices.Count; i++)
        {
            vertices[i] = new Vector3(
                meshData.vertices[i][0],
                meshData.vertices[i][1],
                0
            );
        }
        mesh.vertices = vertices;
        
        // Set triangles
        int[] triangles = new int[meshData.triangles.Count * 3];
        for (int i = 0; i < meshData.triangles.Count; i++)
        {
            triangles[i * 3] = meshData.triangles[i][0];
            triangles[i * 3 + 1] = meshData.triangles[i][1];
            triangles[i * 3 + 2] = meshData.triangles[i][2];
        }
        mesh.triangles = triangles;
        
        // Set UVs
        Vector2[] uvs = new Vector2[meshData.uvs.Count];
        for (int i = 0; i < meshData.uvs.Count; i++)
        {
            uvs[i] = new Vector2(meshData.uvs[i][0], meshData.uvs[i][1]);
        }
        mesh.uv = uvs;
        
        // Set bone weights
        BoneWeight[] boneWeights = new BoneWeight[meshData.boneWeights.Count];
        for (int i = 0; i < meshData.boneWeights.Count; i++)
        {
            var bw = meshData.boneWeights[i];
            boneWeights[i].boneIndex0 = bw.boneIndex0;
            boneWeights[i].boneIndex1 = bw.boneIndex1;
            boneWeights[i].boneIndex2 = bw.boneIndex2;
            boneWeights[i].boneIndex3 = bw.boneIndex3;
            boneWeights[i].weight0 = bw.weight0;
            boneWeights[i].weight1 = bw.weight1;
            boneWeights[i].weight2 = bw.weight2;
            boneWeights[i].weight3 = bw.weight3;
        }
        mesh.boneWeights = boneWeights;
        
        // Set bind poses
        Matrix4x4[] bindPoses = new Matrix4x4[skeletonData.bindPoses.Count];
        for (int i = 0; i < skeletonData.bindPoses.Count; i++)
        {
            var bp = skeletonData.bindPoses[i];
            bindPoses[i] = new Matrix4x4();
            bindPoses[i].m00 = bp.m00; bindPoses[i].m01 = bp.m01; bindPoses[i].m02 = bp.m02; bindPoses[i].m03 = bp.m03;
            bindPoses[i].m10 = bp.m10; bindPoses[i].m11 = bp.m11; bindPoses[i].m12 = bp.m12; bindPoses[i].m13 = bp.m13;
            bindPoses[i].m20 = bp.m20; bindPoses[i].m21 = bp.m21; bindPoses[i].m22 = bp.m22; bindPoses[i].m23 = bp.m23;
            bindPoses[i].m30 = bp.m30; bindPoses[i].m31 = bp.m31; bindPoses[i].m32 = bp.m32; bindPoses[i].m33 = bp.m33;
        }
        mesh.bindposes = bindPoses;
        
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();

        // Create SkinnedMeshRenderer
        GameObject meshObj = new GameObject("SkinnedMesh");
        meshObj.transform.SetParent(characterRoot.transform);
        meshObj.transform.localPosition = Vector3.zero;
        
        SkinnedMeshRenderer skinnedRenderer = meshObj.AddComponent<SkinnedMeshRenderer>();
        skinnedRenderer.sharedMesh = mesh;
        skinnedRenderer.bones = boneArray;
        skinnedRenderer.rootBone = boneArray[0];
        
        // Create and assign material
        Material material = new Material(Shader.Find("Sprites/Default"));
        material.mainTexture = characterTexture;
        skinnedRenderer.material = material;
        
        // Set proper rendering
        skinnedRenderer.updateWhenOffscreen = true;

        Debug.Log("✓ Character imported successfully with full skinned mesh!");
        Debug.Log($"  - {vertices.Length} vertices");
        Debug.Log($"  - {triangles.Length / 3} triangles");
        Debug.Log($"  - {boneArray.Length} bones");
        Debug.Log($"  - Skinning enabled: bones will deform mesh!");
    }
}

'''
        
        script_path = output_path / "CharacterImporter.cs"
        with open(script_path, 'w') as f:
            f.write(script)
        print(f"✅ Generated Unity C# script: {script_path.name}")


    def run(self):
        print("\n" + "=" * 70)
        print("  HYBRID AUTO-RIGGER PIPELINE (Unity-Export) ".center(70, "="))
        print("=" * 70)

        print("\n[1/5] Detecting pose with YOLO...")
        if not self.detect_pose():
            print("❌ Pose not detected. Aborting.")
            return False

        print("\n[2/5] Creating foreground mask...")
        if not self.create_mask_and_texture():
            print("❌ Mask creation failed. Aborting.")
            return False

        print("\n[3/5] Building skeleton hierarchy...")
        self.build_skeleton()
        if not self.skeleton:
            print("❌ Skeleton not built. Aborting.")
            return False

        print("\n[4/5] Generating mesh with Delaunay triangulation...")
        try:
            points, simplices = self.generate_mesh()
        except Exception as e:
            print(f"❌ Mesh generation failed: {e}")
            return False

        print("\n[5/5] Calculating bone weights and exporting files...")
        bones = self.get_bones_for_export()
        weights = compute_vertex_weights(points, bones, falloff=self.falloff, max_influences=4)
        print(f"✓ Calculated weights for {len(weights)} vertices")

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
        print(f"✅ Saved cropped texture: {character_tex_path}")

        # 3. Save Debug Overlay
        self.draw_overlay()
        
        # 4. Generate C# Import Script
        self._generate_unity_import_script(Path(self.output_dir))

        print("\n" + "=" * 70)
        print("✓ PIPELINE COMPLETE!".center(70))
        print("=" * 70)
        print(f"\nOutput directory: {self.output_dir}")
        print("Generated files:")
        print("  • character_tex.png")
        print("  • mesh_data.json")
        print("  • skeleton_data.json")
        print("  • mask.png")
        print("  • joint_overlay.png")
        print("  • CharacterImporter.cs")
        print("\n")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid Auto-Rigger: YOLO pose detection + shapely mesh for Unity/GLB"
    )
    parser.add_argument("--input", required=True, help="Input character drawing (PNG/JPG)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--falloff", type=float, default=40.0, help="Gaussian falloff for bone weights (pixels)")
    args = parser.parse_args()

    try:
        rigger = CharacterExtractorYOLO(
            image_path=args.input,
            output_dir=args.output,
            falloff=args.falloff
        )
        rigger.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)