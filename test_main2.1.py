import cv2
import yaml
import numpy as np
import os
from ultralytics import YOLO

class CharacterExtractorYOLO:
    def __init__(self, image_path, output_dir="output"):
        self.image_path = image_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Could not read image {image_path}")
        self.height, self.width = self.image.shape[:2]
        self.model = YOLO("yolov8n-pose.pt")  # full-body pose model
        self.keypoints = None
        self.skeleton = []
        self.mask = None  # new

    # ---------------------- Pose Detection ----------------------
    def detect_pose(self):
        results = self.model(self.image, verbose=False)
        if not results or len(results[0].keypoints.data) == 0:
            print("⚠ No pose detected.")
            return None
        kp = results[0].keypoints.data[0].cpu().numpy()[:, :2]  # (17,2)
        self.keypoints = kp

    # ---------------------- Mask + Texture ----------------------
    def create_mask_and_texture(self):
        # Create foreground mask using Otsu's thresholding
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, self.mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological cleanup
        kernel = np.ones((3,3), np.uint8)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Save mask
        mask_path = os.path.join(self.output_dir, "mask.png")
        cv2.imwrite(mask_path, self.mask)
        print(f"✅ Saved mask: {mask_path}")
        
        # Save RGBA texture
        texture = cv2.cvtColor(self.image, cv2.COLOR_BGR2BGRA)
        texture_path = os.path.join(self.output_dir, "texture.png")
        cv2.imwrite(texture_path, texture)
        print(f"✅ Saved texture: {texture_path}")

    # ---------------------- Face Anchors (5 points) ----------------------
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
        return {
            "nose": nose,
            "left_eye": left_eye,
            "right_eye": right_eye,
            "face_left": face_left,
            "face_right": face_right,
        }

    # ---------------------- Skeleton Building ----------------------
    def build_skeleton(self):
        if self.keypoints is None:
            return
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

    # ---------------------- Draw Overlay ----------------------
    def draw_overlay(self):
        # Start with RGBA texture if available
        if self.mask is not None:
            overlay = cv2.cvtColor(self.image, cv2.COLOR_BGR2BGRA)
        else:
            overlay = self.image.copy()
        for j in self.skeleton:
            x, y = j["loc"]
            cv2.circle(overlay, (x, y), 4, (0, 0, 0, 255), -1)
            if j["parent"]:
                parent = next((p for p in self.skeleton if p["name"] == j["parent"]), None)
                if parent:
                    px, py = parent["loc"]
                    cv2.line(overlay, (x, y), (px, py), (200, 200, 200, 255), 1)
        out_path = os.path.join(self.output_dir, "joint_overlay.png")
        cv2.imwrite(out_path, overlay)
        print(f"✅ Saved overlay: {out_path}")

    # ---------------------- Save YAML ----------------------
    def save_yaml(self):
        data = {"height": int(self.height), "width": int(self.width), "skeleton": self.skeleton}
        out_path = os.path.join(self.output_dir, "char_cfg.yaml")
        with open(out_path, "w") as f:
            yaml.dump(data, f, sort_keys=False)
        print(f"✅ Saved YAML: {out_path}")

    # ---------------------- Run Pipeline ----------------------
    def run(self):
        self.detect_pose()
        if self.keypoints is None:
            print("❌ Pose not detected. Aborting.")
            return
        self.create_mask_and_texture()
        self.build_skeleton()
        self.draw_overlay()
        self.save_yaml()

import os
import os.path as osp

# Import the updated CharacterExtractorYOLO class from above
# from character_extractor_yolo import CharacterExtractorYOLO  # if in separate file

# ---------------------- Configuration ----------------------
data_dir = "/Users/wenjia/Documents/GitHub/website/AutoRig/data"
output_root = "/Users/wenjia/Documents/GitHub/website/AutoRig/output"

# Create root output folder if missing
os.makedirs(output_root, exist_ok=True)

# Supported image types
valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

# ---------------------- Batch Processing ----------------------
for filename in os.listdir(data_dir):
    if not filename.lower().endswith(valid_exts):
        continue

    input_path = os.path.join(data_dir, filename)
    name_no_ext = os.path.splitext(filename)[0]

    # Each output in its own subfolder
    output_dir = os.path.join(output_root, name_no_ext)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n🔹 Processing: {filename}")
    try:
        extractor = CharacterExtractorYOLO(input_path, output_dir)
        extractor.run()
    except Exception as e:
        print(f"❌ Failed to process {filename}: {e}")

print("\n✅ All done! Results saved in:", output_root)
