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
        self.model = YOLO("yolov8n-pose.pt")  # lightweight full-body model
        self.keypoints = None
        self.skeleton = []

    # ---------------------- Pose Detection ----------------------
    def detect_pose(self):
        results = self.model(self.image, verbose=False)
        if not results or len(results[0].keypoints.data) == 0:
            print("⚠ No pose detected.")
            return None
        kp = results[0].keypoints.data[0].cpu().numpy()[:, :2]  # (17,2)
        self.keypoints = kp

    # ---------------------- Skeleton Building ----------------------
    def build_skeleton(self):
        if self.keypoints is None:
            return
        kp = self.keypoints

        # COCO format keypoints:
        # 0:nose 1:left_eye 2:right_eye 3:left_ear 4:right_ear
        # 5:left_shoulder 6:right_shoulder 7:left_elbow 8:right_elbow
        # 9:left_wrist 10:right_wrist 11:left_hip 12:right_hip
        # 13:left_knee 14:right_knee 15:left_ankle 16:right_ankle
        def xy(i): return [float(kp[i][0]), float(kp[i][1])]

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

        def mid(a, b):
            return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]

        # ----- Add 4 extra face dots -----
        eye_mid_y = (joints["left_eye"][1] + joints["right_eye"][1]) / 2
        eye_span = abs(joints["right_eye"][0] - joints["left_eye"][0])
        face_left = [joints["left_eye"][0] - 1.2 * eye_span, eye_mid_y]
        face_right = [joints["right_eye"][0] + 1.2 * eye_span, eye_mid_y]

        # Insert new points
        joints["face_left"] = face_left
        joints["face_right"] = face_right

        # ----- Skeleton definition -----
        self.skeleton = [
            {"name": "root", "loc": mid(joints["left_hip"], joints["right_hip"]), "parent": None},
            {"name": "hip", "loc": mid(joints["left_hip"], joints["right_hip"]), "parent": "root"},
            {"name": "torso", "loc": mid(joints["left_shoulder"], joints["right_shoulder"]), "parent": "hip"},
            {"name": "neck", "loc": joints["nose"], "parent": "torso"},

            # Face
            {"name": "left_eye", "loc": joints["left_eye"], "parent": "neck"},
            {"name": "right_eye", "loc": joints["right_eye"], "parent": "neck"},
            {"name": "nose", "loc": joints["nose"], "parent": "neck"},
            {"name": "face_left", "loc": joints["face_left"], "parent": "neck"},
            {"name": "face_right", "loc": joints["face_right"], "parent": "neck"},

            # Arms
            {"name": "left_shoulder_b", "loc": joints["left_shoulder"], "parent": "torso"},
            {"name": "left_elbow_b", "loc": joints["left_elbow"], "parent": "left_shoulder_b"},
            {"name": "left_hand_b", "loc": joints["left_hand"], "parent": "left_elbow_b"},

            {"name": "right_shoulder_b", "loc": joints["right_shoulder"], "parent": "torso"},
            {"name": "right_elbow_b", "loc": joints["right_elbow"], "parent": "right_shoulder_b"},
            {"name": "right_hand_b", "loc": joints["right_hand"], "parent": "right_elbow_b"},

            # Legs
            {"name": "left_hip_b", "loc": joints["left_hip"], "parent": "root"},
            {"name": "left_knee_b", "loc": joints["left_knee"], "parent": "left_hip_b"},
            {"name": "left_foot_b", "loc": joints["left_foot"], "parent": "left_knee_b"},

            {"name": "right_hip_b", "loc": joints["right_hip"], "parent": "root"},
            {"name": "right_knee_b", "loc": joints["right_knee"], "parent": "right_hip_b"},
            {"name": "right_foot_b", "loc": joints["right_foot"], "parent": "right_knee_b"},
        ]

        for j in self.skeleton:
            j["loc"] = [int(round(j["loc"][0])), int(round(j["loc"][1]))]

    # ---------------------- Drawing & Saving ----------------------
    def draw_overlay(self):
        img = self.image.copy()
        for j in self.skeleton:
            x, y = j["loc"]
            cv2.circle(img, (x, y), 8, (0, 0, 0), -1)  # large black dots
            if j["parent"]:
                parent = next((p for p in self.skeleton if p["name"] == j["parent"]), None)
                if parent:
                    px, py = parent["loc"]
                    cv2.line(img, (x, y), (px, py), (100, 100, 100), 2)  # gray connection

        out_path = os.path.join(self.output_dir, "joint_overlay.png")
        cv2.imwrite(out_path, img)
        print(f"✅ Saved overlay: {out_path}")

    def save_yaml(self):
        data = {"height": int(self.height), "width": int(self.width), "skeleton": self.skeleton}
        out_path = os.path.join(self.output_dir, "char_cfg.yaml")
        with open(out_path, "w") as f:
            yaml.dump(data, f, sort_keys=False)
        print(f"✅ Saved YAML: {out_path}")

    # ---------------------- Main ----------------------
    def run(self):
        self.detect_pose()
        if self.keypoints is None:
            print("❌ Pose not detected.")
            return
        self.build_skeleton()
        self.draw_overlay()
        self.save_yaml()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Full-body skeleton extractor using YOLO Pose + extra face dots")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()

    extractor = CharacterExtractorYOLO(args.image_path, args.output)
    extractor.run()
