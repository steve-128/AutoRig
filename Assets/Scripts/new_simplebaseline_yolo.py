import cv2
import yaml
import torch
import numpy as np
import os
import sys
from pathlib import Path
import torch.nn as nn
from torchvision.models import resnet50
import argparse

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


#########################################
# Create output folder
#########################################

OUT_DIR = "simplebaseline_outputs"
os.makedirs(OUT_DIR, exist_ok=True)


#########################################
# 1. SimpleBaseline Model
#########################################

class SimpleBaseline(nn.Module):
    def __init__(self, num_keypoints=17):
        super(SimpleBaseline, self).__init__()

        # Load ResNet50 architecture (but keep original structure)
        resnet = resnet50(weights=None)
        
        # Keep the original ResNet layers (not as Sequential)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # 3 deconv layers
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Final heatmap layer
        self.final_layer = nn.Conv2d(256, num_keypoints, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Backbone forward pass
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Deconv and final layer
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x

#########################################
# 2. Decode heatmaps → keypoint (x,y)
#########################################

def get_keypoints_from_heatmaps(heatmaps):
    heatmaps = heatmaps.squeeze(0)  # [K, H, W]
    K, H, W = heatmaps.shape

    kpts = []
    for i in range(K):
        hm = heatmaps[i]
        idx = torch.argmax(hm).item()
        y, x = divmod(idx, W)
        kpts.append((x, y))  # heatmap coords

    return kpts  # list of [x, y]


#########################################
# 3. Draw skeleton on image
#########################################

COCO_PAIRS = [
    # Face connections
    (0,1),           # nose to left_eye
    (0,2),           # nose to right_eye
    (1,3),           # left_eye to left_ear
    (2,4),           # right_eye to right_ear
    (1,2),           # left_eye to right_eye
    
    # Neck connections (face to body)
    (0,5),           # nose to left_shoulder
    (0,6),           # nose to right_shoulder
    
    # Body connections
    (5,7),(7,9),     # left arm
    (6,8),(8,10),    # right arm
    (11,13),(13,15), # left leg
    (12,14),(14,16), # right leg
    (5,6),           # shoulders
    (11,12),         # hips
    (5,11),(6,12),   # torso
]

COCO_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]


def draw_skeleton(image, kpts):
    """Draw skeleton with keypoints and connections"""
    # Draw lines (skeleton connections)
    for (i, j) in COCO_PAIRS:
        x1, y1 = kpts[i]
        x2, y2 = kpts[j]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Draw keypoints (circles)
    for idx, (x, y) in enumerate(kpts):
        # Different colors for face vs body
        if idx < 5:  # Face keypoints
            cv2.circle(image, (x, y), 6, (255, 0, 255), -1)  # Magenta
        else:  # Body keypoints
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Red

    return image


#########################################
# 4. Convert keypoints → YAML
#########################################

def save_yaml(kpts, filename, method_info=None):
    data = {"joints": {}}
    
    # Add metadata about which method was used
    if method_info:
        data["metadata"] = method_info
    
    for name, (x, y) in zip(COCO_NAMES, kpts):
        data["joints"][name] = [int(x), int(y)]

    with open(filename, "w") as f:
        yaml.dump(data, f, sort_keys=False)


#########################################
# 5. Load pretrained weights
#########################################

def load_pretrained_simplebaseline():
    """Load SimpleBaseline with pretrained COCO weights"""
    
    checkpoint_path = "pretrained_weights/pose_resnet_50_256x192.pth.tar"
    
    # Check if file exists
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] File not found: {checkpoint_path}")
        print("Please download pose_resnet_50_256x192.pth.tar and place it in pretrained_weights/")
        return None
    
    print(f"[INFO] Loading pretrained weights from {checkpoint_path}")
    
    # Create model
    model = SimpleBaseline(num_keypoints=17)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove "module." prefix if it exists (from multi-GPU training)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove "module." prefix
        else:
            new_key = key
        new_state_dict[new_key] = value
    
    # Load weights into model
    model.load_state_dict(new_state_dict)
    model.eval()
    
    print("[INFO] Successfully loaded pretrained weights!")
    return model


#########################################
# 6. SimpleBaseline face refinement
#########################################

def improve_face_simplebaseline(orig, body_kpts, sb_model):
    """
    Crop face region and re-run SimpleBaseline at higher resolution
    """
    print("[INFO] Refining face with SimpleBaseline crop...")
    
    # Get key body points
    nose = body_kpts[0]
    left_shoulder = body_kpts[5]
    right_shoulder = body_kpts[6]
    
    # Calculate face region size based on shoulder width
    shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
    
    if shoulder_width < 50:
        shoulder_width = orig.shape[1] // 3
    
    face_size = int(shoulder_width * 1.8)
    
    # Calculate crop boundaries
    x_center = nose[0]
    y_center = nose[1]
    
    x1 = max(0, x_center - face_size // 2)
    y1 = max(0, y_center - int(face_size * 0.8))
    x2 = min(orig.shape[1], x_center + face_size // 2)
    y2 = min(orig.shape[0], y_center + int(face_size * 0.4))
    
    if x2 <= x1 or y2 <= y1:
        print("[WARN] Invalid face crop")
        return None
    
    face_crop = orig[y1:y2, x1:x2]
    
    # Preprocess
    inp = cv2.resize(face_crop, (192, 256))
    inp_rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    inp_rgb = inp_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp_rgb = (inp_rgb - mean) / std
    tensor = torch.from_numpy(inp_rgb).permute(2, 0, 1).unsqueeze(0).float()
    
    # Run pose estimation on face crop
    with torch.no_grad():
        face_heatmaps = sb_model(tensor)
    
    face_kpts = get_keypoints_from_heatmaps(face_heatmaps)
    
    # Scale back to original image coordinates
    hm_H, hm_W = face_heatmaps.shape[2], face_heatmaps.shape[3]
    crop_h, crop_w = face_crop.shape[:2]
    
    refined_face_kpts = []
    for i in range(5):  # Only first 5 keypoints
        hx, hy = face_kpts[i]
        x = int(hx / hm_W * crop_w) + x1
        y = int(hy / hm_H * crop_h) + y1
        refined_face_kpts.append((x, y))
    
    return refined_face_kpts, face_crop, (x1, y1, x2, y2)


#########################################
# 7. YOLO-Pose for face keypoints
#########################################

def improve_face_yolo_pose(orig, output_folder):
    """
    Run YOLO-Pose on the FULL image and extract only face keypoints (0-4)
    """
    if not YOLO_AVAILABLE:
        print("[ERROR] YOLO not available. Install with: pip install ultralytics")
        return None
    
    print("[INFO] Running YOLOv8-Pose on full image...")
    
    # Load YOLOv8 pose model
    yolo_pose = YOLO("yolov8n-pose.pt")
    
    # Run YOLO-Pose with lower confidence threshold for better detection
    results = yolo_pose(orig, conf=0.25, verbose=False)
    
    if not results or len(results[0].keypoints.data) == 0:
        print("[WARN] YOLO-Pose detected no keypoints with conf=0.25")
        print("[INFO] Trying with even lower confidence threshold...")
        
        # Try again with very low threshold
        results = yolo_pose(orig, conf=0.1, verbose=False)
        
        if not results or len(results[0].keypoints.data) == 0:
            print("[WARN] Still no detections. YOLO may not work well for this creature.")
            return None
    
    # Get YOLO keypoints (17 keypoints from full image)
    yolo_kpts = results[0].keypoints.data[0].cpu().numpy()[:, :2]  # [17, 2]
    
    # Extract ONLY face keypoints (0-4)
    yolo_face_kpts = []
    for i in range(5):
        x, y = yolo_kpts[i]
        yolo_face_kpts.append((int(x), int(y)))
    
    print(f"[INFO] YOLO-Pose detected face keypoints:")
    for i, name in enumerate(COCO_NAMES[:5]):
        x, y = yolo_face_kpts[i]
        print(f"  {name:15s}: ({x:4d}, {y:4d})")
    
    # Save YOLO full pose visualization (all 17 keypoints)
    yolo_vis = orig.copy()
    
    # Draw all YOLO keypoints in cyan (for reference)
    for idx, (x, y) in enumerate(yolo_kpts):
        x, y = int(x), int(y)
        cv2.circle(yolo_vis, (x, y), 4, (255, 255, 0), -1)  # Cyan
        cv2.putText(yolo_vis, str(idx), (x+8, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Highlight face keypoints in magenta
    for idx, (x, y) in enumerate(yolo_face_kpts):
        cv2.circle(yolo_vis, (x, y), 7, (255, 0, 255), -1)  # Magenta (bigger)
        cv2.putText(yolo_vis, COCO_NAMES[idx], (x+10, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
    # Draw YOLO skeleton connections
    yolo_connections = [
        (5,7),(7,9),(6,8),(8,10),  # arms
        (11,13),(13,15),(12,14),(14,16),  # legs
        (5,6),(11,12),(5,11),(6,12),  # torso
        (0,1),(0,2),(1,3),(2,4)  # face
    ]
    for (i, j) in yolo_connections:
        x1, y1 = int(yolo_kpts[i][0]), int(yolo_kpts[i][1])
        x2, y2 = int(yolo_kpts[j][0]), int(yolo_kpts[j][1])
        cv2.line(yolo_vis, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow lines
    
    yolo_full_path = os.path.join(output_folder, "05_yolo_full_pose.png")
    cv2.imwrite(yolo_full_path, yolo_vis)
    print(f"[INFO] Saved YOLO full pose visualization to {yolo_full_path}")
    
    # Save YOLO face-only visualization
    yolo_face_vis = orig.copy()
    for idx, (x, y) in enumerate(yolo_face_kpts):
        cv2.circle(yolo_face_vis, (x, y), 8, (255, 0, 255), -1)
        cv2.putText(yolo_face_vis, COCO_NAMES[idx], (x+10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw face connections
    face_connections = [(0,1),(0,2),(1,3),(2,4),(1,2)]
    for (i, j) in face_connections:
        x1, y1 = yolo_face_kpts[i]
        x2, y2 = yolo_face_kpts[j]
        cv2.line(yolo_face_vis, (x1, y1), (x2, y2), (255, 0, 255), 3)
    
    yolo_face_path = os.path.join(output_folder, "06_yolo_face_only.png")
    cv2.imwrite(yolo_face_path, yolo_face_vis)
    print(f"[INFO] Saved YOLO face-only visualization to {yolo_face_path}")
    
    return yolo_face_kpts


#########################################
# 8. Full pipeline
#########################################

def run_pipeline(image_path, face_method='none'):
    """
    face_method options:
    - 'none': No face refinement
    - 'simplebaseline': Use SimpleBaseline crop for face
    - 'yolo': Use YOLO-Pose for face keypoints (full image)
    """
    
    print(f"\n{'='*60}")
    print(f"[INFO] Starting pipeline on: {image_path}")
    print(f"[INFO] Face refinement method: {face_method}")
    print(f"{'='*60}\n")
    
    #########################################
    # Create output subfolder
    #########################################
    image_name = Path(image_path).stem
    output_folder = os.path.join(OUT_DIR, image_name)
    os.makedirs(output_folder, exist_ok=True)
    print(f"[INFO] Output folder: {output_folder}")
    
    #########################################
    # Load pretrained model
    #########################################
    sb = load_pretrained_simplebaseline()
    if sb is None:
        return
    
    #########################################
    # Load image
    #########################################
    orig = cv2.imread(image_path)
    if orig is None:
        print(f"[ERROR] Could not read image")
        return
    
    print(f"[INFO] Image shape: {orig.shape}")
    orig_h, orig_w = orig.shape[:2]
    
    # Save original
    orig_save_path = os.path.join(output_folder, "01_original.png")
    cv2.imwrite(orig_save_path, orig)
    print(f"[INFO] Saved original image to {orig_save_path}")
    
    #########################################
    # Preprocess image
    #########################################
    inp = cv2.resize(orig, (192, 256))
    inp_rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    inp_rgb = inp_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp_rgb = (inp_rgb - mean) / std
    tensor = torch.from_numpy(inp_rgb).permute(2, 0, 1).unsqueeze(0).float()
    
    #########################################
    # Run SimpleBaseline (full image for body)
    #########################################
    print("[INFO] Running SimpleBaseline on full image...")
    with torch.no_grad():
        heatmaps = sb(tensor)
    
    kpts = get_keypoints_from_heatmaps(heatmaps)
    hm_H, hm_W = heatmaps.shape[2], heatmaps.shape[3]
    
    # Scale keypoints back to original
    final_kpts = []
    for (hx, hy) in kpts:
        x = int(hx / hm_W * orig_w)
        y = int(hy / hm_H * orig_h)
        final_kpts.append((x, y))
    
    print(f"[INFO] SimpleBaseline keypoints (first 5 - face): {final_kpts[:5]}")
    
    # Track which method was used for metadata
    method_info = {
        "body_keypoints": "SimpleBaseline",
        "face_keypoints": "SimpleBaseline"
    }
    
    #########################################
    # FACE REFINEMENT
    #########################################
    if face_method == 'simplebaseline':
        try:
            result = improve_face_simplebaseline(orig, final_kpts, sb)
            if result is not None:
                refined_face, face_crop, crop_coords = result
                final_kpts[:5] = refined_face
                method_info["face_keypoints"] = "SimpleBaseline (cropped & refined)"
                print("[INFO] Face refined with SimpleBaseline!")
                
                # Save visualization
                x1, y1, x2, y2 = crop_coords
                face_vis = orig.copy()
                cv2.rectangle(face_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(face_vis, "SimpleBaseline Face Region", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                face_region_path = os.path.join(output_folder, "05_simplebaseline_face_region.png")
                cv2.imwrite(face_region_path, face_vis)
                print(f"[INFO] Saved face region to {face_region_path}")
                
                face_crop_path = os.path.join(output_folder, "06_simplebaseline_face_crop.png")
                cv2.imwrite(face_crop_path, face_crop)
                print(f"[INFO] Saved face crop to {face_crop_path}")
        except Exception as e:
            print(f"[WARN] SimpleBaseline face refinement failed: {e}")
    
    elif face_method == 'yolo':
        if not YOLO_AVAILABLE:
            print("[ERROR] YOLO not installed. Use: pip install ultralytics")
        else:
            try:
                yolo_face_kpts = improve_face_yolo_pose(orig, output_folder)
                if yolo_face_kpts is not None:
                    # Override ONLY face keypoints (0-4) with YOLO results
                    final_kpts[:5] = yolo_face_kpts
                    method_info["face_keypoints"] = "YOLO-Pose"
                    print(f"[INFO] Face keypoints replaced with YOLO-Pose results!")
                    print(f"[INFO] Final face keypoints: {final_kpts[:5]}")
            except Exception as e:
                print(f"[WARN] YOLO-Pose face refinement failed: {e}")
    
    else:
        print("[INFO] No face refinement applied")
    
    print(f"\n[INFO] Final keypoints (first 5 - face): {final_kpts[:5]}")
    print(f"[INFO] Final keypoints (body sample): {final_kpts[5:8]}")
    
    #########################################
    # Draw skeleton overlay
    #########################################
    skel_img = orig.copy()
    skel_img = draw_skeleton(skel_img, final_kpts)
    
    skel_path = os.path.join(output_folder, "02_skeleton_overlay.png")
    cv2.imwrite(skel_path, skel_img)
    print(f"[INFO] Saved skeleton overlay to {skel_path}")
    
    #########################################
    # Draw keypoints with labels
    #########################################
    keypoints_img = orig.copy()
    for idx, (x, y) in enumerate(final_kpts):
        if idx < 5:  # Face keypoints
            cv2.circle(keypoints_img, (x, y), 8, (255, 0, 255), -1)  # Magenta
        else:  # Body keypoints
            cv2.circle(keypoints_img, (x, y), 6, (0, 0, 255), -1)  # Red
        
        # Add keypoint labels
        cv2.putText(keypoints_img, f"{idx}:{COCO_NAMES[idx]}", (x + 10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    keypoints_path = os.path.join(output_folder, "03_keypoints_labeled.png")
    cv2.imwrite(keypoints_path, keypoints_img)  # FIXED: correct argument order
    print(f"[INFO] Saved keypoints visualization to {keypoints_path}")
    
    #########################################
    # Save YAML
    #########################################
    yaml_path = os.path.join(output_folder, "04_pose.yaml")
    save_yaml(final_kpts, yaml_path, method_info)
    print(f"[INFO] Saved YAML to {yaml_path}")
    
    #########################################
    # Summary
    #########################################
    print(f"\n{'='*60}")
    print(f"[INFO] Pipeline complete!")
    print(f"[INFO] Method used:")
    print(f"  - Body: {method_info['body_keypoints']}")
    print(f"  - Face: {method_info['face_keypoints']}")
    print(f"[INFO] All outputs saved to: {output_folder}")
    print(f"{'='*60}\n")
    
    return final_kpts


#########################################
# Run
#########################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='SimpleBaseline + Face Refinement Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # No face refinement (fastest)
  python3 %(prog)s data/creature15.png
  
  # SimpleBaseline face crop (medium)
  python3 %(prog)s data/creature15.png --face-method simplebaseline
  
  # YOLO-Pose for face (runs on full image, extracts face keypoints only)
  python3 %(prog)s data/creature15.png --face-method yolo
        '''
    )
    
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('--face-method', type=str, 
                       choices=['none', 'simplebaseline', 'yolo'],
                       default='none',
                       help='Face refinement method (default: none)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"\n[ERROR] Image file not found: {args.image_path}")
        sys.exit(1)
    
    run_pipeline(args.image_path, face_method=args.face_method)

    # No face refinement (fastest)
    # python simplebaseline_pipeline.py data/creature15.png

    # SimpleBaseline crop for face (medium)
    # python simplebaseline_pipeline.py data/creature15.png --face-method simplebaseline

    # YOLO for face detection (most accurate if YOLO detects face well)
    # python simplebaseline_pipeline.py data/creature20.png --face-method yolo