import cv2
import yaml
import torch
import numpy as np
import os
import sys
from pathlib import Path
import torch.nn as nn
from torchvision.models import resnet50


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
    (5,7),(7,9),     # left arm
    (6,8),(8,10),    # right arm
    (11,13),(13,15), # left leg
    (12,14),(14,16), # right leg
    (5,6),           # shoulders
    (11,12),         # hips
    (5,11),(6,12)    # torso
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
    for (x, y) in kpts:
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

    return image


#########################################
# 4. Convert keypoints → YAML
#########################################

def save_yaml(kpts, filename):
    data = {"joints": {}}
    for name, (x, y) in zip(COCO_NAMES, kpts):
        data["joints"][name] = [int(x), int(y)]

    with open(filename, "w") as f:
        yaml.dump(data, f)


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
    
    print("[INFO] ✓ Successfully loaded pretrained weights!")
    return model


#########################################
# 6. Face refinement function (NEW!)
#########################################

def improve_face_detection(orig, body_kpts, sb_model):
    """
    Crop face region based on body keypoints and re-run at higher resolution
    This improves accuracy for small face keypoints
    """
    print("[INFO] Refining face keypoints with cropped region...")
    
    # Get key body points
    nose = body_kpts[0]
    left_eye = body_kpts[1]
    right_eye = body_kpts[2]
    left_shoulder = body_kpts[5]
    right_shoulder = body_kpts[6]
    
    # Calculate face region size based on shoulder width
    shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
    
    # If shoulder width is too small or zero, use image width as fallback
    if shoulder_width < 50:
        shoulder_width = orig.shape[1] // 3
    
    face_size = int(shoulder_width * 1.8)  # Face region size
    
    # Calculate crop boundaries (centered on nose, extended upward)
    x_center = nose[0]
    y_center = nose[1]
    
    x1 = max(0, x_center - face_size // 2)
    y1 = max(0, y_center - int(face_size * 0.8))  # More space above nose
    x2 = min(orig.shape[1], x_center + face_size // 2)
    y2 = min(orig.shape[0], y_center + int(face_size * 0.4))  # Less space below
    
    # Make sure crop is valid
    if x2 <= x1 or y2 <= y1:
        print("[WARN] Invalid face crop, skipping refinement")
        return None
    
    face_crop = orig[y1:y2, x1:x2]
    
    # Save face crop for debugging
    print(f"[INFO] Face crop region: ({x1}, {y1}) to ({x2}, {y2})")
    print(f"[INFO] Face crop shape: {face_crop.shape}")
    
    # Preprocess face crop
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
    
    print(f"[INFO] Face heatmaps shape: {face_heatmaps.shape}")
    
    # Extract face keypoints (0-4: nose, left_eye, right_eye, left_ear, right_ear)
    face_kpts = get_keypoints_from_heatmaps(face_heatmaps)
    
    # Scale back to original image coordinates
    hm_H, hm_W = face_heatmaps.shape[2], face_heatmaps.shape[3]
    crop_h, crop_w = face_crop.shape[:2]
    
    refined_face_kpts = []
    for i in range(5):  # Only first 5 keypoints (face keypoints)
        hx, hy = face_kpts[i]
        x = int(hx / hm_W * crop_w) + x1
        y = int(hy / hm_H * crop_h) + y1
        refined_face_kpts.append((x, y))
    
    print(f"[INFO] Refined face keypoints: {refined_face_kpts}")
    
    return refined_face_kpts, face_crop, (x1, y1, x2, y2)


#########################################
# 7. Full pipeline
#########################################

def run_pipeline(image_path):
    print(f"\n{'='*60}")
    print(f"[INFO] Starting pipeline on: {image_path}")
    print(f"{'='*60}\n")
    
    #########################################
    # Create output subfolder for this image
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
        print(f"[ERROR] Could not read image at path: {image_path}")
        return
    
    print(f"[INFO] Image shape: {orig.shape}")
    orig_h, orig_w = orig.shape[:2]
    
    # Save original image to output folder
    orig_path = os.path.join(output_folder, "01_original.png")
    cv2.imwrite(orig_path, orig)
    print(f"[INFO] ✓ Saved original image to {orig_path}")
    
    #########################################
    # Preprocess image (resize to 192x256)
    #########################################
    inp = cv2.resize(orig, (192, 256))
    inp_rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    
    # Normalize with ImageNet mean/std
    inp_rgb = inp_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp_rgb = (inp_rgb - mean) / std
    
    # Convert to tensor
    tensor = torch.from_numpy(inp_rgb).permute(2, 0, 1).unsqueeze(0).float()
    
    print(f"[INFO] Input tensor shape: {tensor.shape}")
    
    #########################################
    # Run SimpleBaseline (full image)
    #########################################
    print("[INFO] Running pose estimation on full image...")
    with torch.no_grad():
        heatmaps = sb(tensor)
    
    print(f"[INFO] Heatmaps shape: {heatmaps.shape}")
    
    #########################################
    # Extract keypoints from heatmaps
    #########################################
    kpts = get_keypoints_from_heatmaps(heatmaps)
    print(f"[INFO] Decoded {len(kpts)} keypoints")
    
    hm_H = heatmaps.shape[2]
    hm_W = heatmaps.shape[3]
    
    #########################################
    # Scale keypoints back to original image size
    #########################################
    final_kpts = []
    for (hx, hy) in kpts:
        x = int(hx / hm_W * orig_w)
        y = int(hy / hm_H * orig_h)
        final_kpts.append((x, y))
    
    print(f"[INFO] Initial keypoints (first 5): {final_kpts[:5]}")
    
    #########################################
    # FACE REFINEMENT (OPTIONAL)
    #########################################
    try:
        refinement_result = improve_face_detection(orig, final_kpts, sb)
        
        if refinement_result is not None:
            refined_face, face_crop, crop_coords = refinement_result
            
            # Replace first 5 keypoints (face) with refined versions
            final_kpts[:5] = refined_face
            print("[INFO] ✓ Face keypoints refined!")
            
            # Save face crop with bounding box
            x1, y1, x2, y2 = crop_coords
            face_region_img = orig.copy()
            cv2.rectangle(face_region_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(face_region_img, "Face Region", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            face_region_path = os.path.join(output_folder, "05_face_crop_region.png")
            cv2.imwrite(face_region_path, face_region_img)
            print(f"[INFO] ✓ Saved face crop region to {face_region_path}")
            
            # Save the actual face crop
            face_crop_path = os.path.join(output_folder, "06_face_crop.png")
            cv2.imwrite(face_crop_path, face_crop)
            print(f"[INFO] ✓ Saved face crop to {face_crop_path}")
        else:
            print("[WARN] Face refinement failed, using original face keypoints")
    except Exception as e:
        print(f"[WARN] Error during face refinement: {e}")
        print("[WARN] Using original face keypoints")
    
    print(f"[INFO] Final keypoints (first 5): {final_kpts[:5]}")
    
    #########################################
    # Draw skeleton overlay
    #########################################
    skel_img = orig.copy()
    skel_img = draw_skeleton(skel_img, final_kpts)
    
    skel_path = os.path.join(output_folder, "02_skeleton_overlay.png")
    cv2.imwrite(skel_path, skel_img)
    print(f"[INFO] ✓ Saved skeleton overlay to {skel_path}")
    
    #########################################
    # Save keypoints visualization (just dots, no lines)
    #########################################
    keypoints_img = orig.copy()
    
    # Highlight face keypoints in different color
    for idx, (x, y) in enumerate(final_kpts):
        if idx < 5:  # Face keypoints
            cv2.circle(keypoints_img, (x, y), 8, (255, 0, 255), -1)  # Magenta for face
        else:  # Body keypoints
            cv2.circle(keypoints_img, (x, y), 6, (0, 0, 255), -1)  # Red for body
        
        # Add keypoint labels
        cv2.putText(keypoints_img, str(idx), (x + 10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    keypoints_path = os.path.join(output_folder, "03_keypoints_only.png")
    cv2.imwrite(keypoints_path, keypoints_img)
    print(f"[INFO] ✓ Saved keypoints visualization to {keypoints_path}")
    
    #########################################
    # Save YAML
    #########################################
    yaml_path = os.path.join(output_folder, "04_pose.yaml")
    save_yaml(final_kpts, yaml_path)
    print(f"[INFO] ✓ Saved YAML to {yaml_path}")
    
    #########################################
    # Summary
    #########################################
    print(f"\n{'='*60}")
    print(f"[INFO] ✓ Pipeline complete!")
    print(f"[INFO] All outputs saved to: {output_folder}")
    print(f"{'='*60}\n")
    
    return final_kpts


#########################################
# Run
#########################################

if __name__ == "__main__":
    # Check if image path is provided as command-line argument
    if len(sys.argv) < 2:
        print("\n[ERROR] No image path provided!")
        print("\nUsage:")
        print(f"  python3 {sys.argv[0]} <image_path>")
        print("\nExample:")
        print(f"  python3 {sys.argv[0]} data/creature15.png")
        print(f"  python3 {sys.argv[0]} data/creature21.png")
        print()
        sys.exit(1)
    
    # Get image path from command line
    image_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"\n[ERROR] Image file not found: {image_path}")
        print("Please check the path and try again.\n")
        sys.exit(1)
    
    # Run the pipeline
    run_pipeline(image_path)

    # Usage:
    # WITH FACE REFINEMENT
    #   python3 simplebaseline_pipeline.py data/creature16.png
    # WITHOUT:
    #   python3 simplebaseline_pipeline.py data/creature16.png --no-face-refine 
