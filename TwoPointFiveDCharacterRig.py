import cv2
import numpy as np
from scipy.spatial import Delaunay
import trimesh
from pathlib import Path

class TwoPointFiveDCharacterRig:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = None
        self.mask = None
        
        # Annotation data
        self.joints = {}
        self.body_parts = {}  # Store separate meshes for each body part
        
    def load_and_preprocess(self):
        """Phase 1: Load image and create initial mask"""
        print("\n" + "="*60)
        print("PHASE 1: PREPROCESSING & ANNOTATION")
        print("="*60)
        
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from {self.image_path}")
        
        print("✓ Image loaded")
        
        # Create foreground mask using Otsu's thresholding
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, self.mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Clean mask
        kernel = np.ones((3,3), np.uint8)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        cv2.imwrite('output/mask.png', self.mask)
        print("✓ Foreground mask created")
        
    def extract_joints(self):
        """Extract 15 joint keypoints"""
        print("\nExtracting joint keypoints...")
        
        # Find contour
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            raise ValueError("No figure found in image")
        
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        cx = x + w/2
        
        # Create humanoid skeleton with proper proportions
        self.joints = {
            'head': [cx, y + h*0.08],
            'neck': [cx, y + h*0.15],
            'left_shoulder': [cx - w*0.18, y + h*0.2],
            'right_shoulder': [cx + w*0.18, y + h*0.2],
            'left_elbow': [cx - w*0.25, y + h*0.35],
            'right_elbow': [cx + w*0.25, y + h*0.35],
            'left_wrist': [cx - w*0.3, y + h*0.5],
            'right_wrist': [cx + w*0.3, y + h*0.5],
            'spine': [cx, y + h*0.35],
            'pelvis': [cx, y + h*0.5],
            'left_hip': [cx - w*0.1, y + h*0.5],
            'right_hip': [cx + w*0.1, y + h*0.5],
            'left_knee': [cx - w*0.12, y + h*0.7],
            'right_knee': [cx + w*0.12, y + h*0.7],
            'left_ankle': [cx - w*0.13, y + h*0.9],
            'right_ankle': [cx + w*0.13, y + h*0.9],
        }
        
        # Visualize skeleton
        self._visualize_skeleton()
        print(f"✓ Created {len(self.joints)} joint keypoints")
    
    def _visualize_skeleton(self):
        """Create visualization showing joints and bones"""
        vis = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)
        
        # Draw bones
        bone_pairs = [
            ('head', 'neck'),
            ('neck', 'left_shoulder'), ('neck', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
            ('neck', 'spine'), ('spine', 'pelvis'),
            ('pelvis', 'left_hip'), ('pelvis', 'right_hip'),
            ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
        ]
        
        for joint1, joint2 in bone_pairs:
            if joint1 in self.joints and joint2 in self.joints:
                p1 = tuple(map(int, self.joints[joint1]))
                p2 = tuple(map(int, self.joints[joint2]))
                cv2.line(vis, p1, p2, (0, 255, 255), 3)
        
        # Draw joints on top
        for joint_name, (x, y) in self.joints.items():
            cv2.circle(vis, (int(x), int(y)), 6, (0, 0, 255), -1)
            cv2.circle(vis, (int(x), int(y)), 7, (255, 255, 255), 1)
        
        cv2.imwrite('output/skeleton_visualization.png', vis)
        print("✓ Skeleton visualization saved")
        
    def create_body_part_meshes(self):
        """Create separate mesh for each body part"""
        print("\n" + "="*60)
        print("PHASE 2: CREATING ARTICULATED BODY PARTS")
        print("="*60)
        
        # Define body parts with their joints
        part_definitions = {
            'head': (['head', 'neck'], 15),
            'torso': (['neck', 'left_shoulder', 'right_shoulder', 'spine', 'pelvis', 
                      'left_hip', 'right_hip'], 12),
            'left_upper_arm': (['left_shoulder', 'left_elbow'], 8),
            'left_forearm': (['left_elbow', 'left_wrist'], 8),
            'right_upper_arm': (['right_shoulder', 'right_elbow'], 8),
            'right_forearm': (['right_elbow', 'right_wrist'], 8),
            'left_thigh': (['left_hip', 'left_knee'], 10),
            'left_shin': (['left_knee', 'left_ankle'], 10),
            'right_thigh': (['right_hip', 'right_knee'], 10),
            'right_shin': (['right_knee', 'right_ankle'], 10),
        }
        
        for part_name, (joint_names, radius) in part_definitions.items():
            mesh = self._create_limb_mesh(joint_names, radius)
            if mesh is not None:
                self.body_parts[part_name] = mesh
                print(f"✓ Created {part_name}")
        
        print(f"\n✓ Created {len(self.body_parts)} body part meshes")
        
    def _create_limb_mesh(self, joint_names, radius):
        """Create a cylindrical/capsule mesh for a limb"""
        if len(joint_names) < 2:
            return None
        
        # Get joint positions
        positions = [self.joints[name] for name in joint_names]
        
        if len(positions) == 2:
            # Simple limb - create capsule
            return self._create_capsule_mesh(positions[0], positions[1], radius)
        else:
            # Complex part (like torso) - create convex hull
            return self._create_hull_mesh(positions, radius)
    
    def _create_capsule_mesh(self, start, end, radius):
        """Create a capsule (cylinder with rounded ends) between two points"""
        start = np.array([start[0], start[1], 0.0])
        end = np.array([end[0], end[1], 0.0])
        
        # Calculate length and direction
        direction = end - start
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return None
        direction = direction / length
        
        # Create cylinder
        segments = 12
        vertices = []
        
        # Perpendicular vectors for cylinder
        if abs(direction[2]) < 0.9:
            perp1 = np.cross(direction, [0, 0, 1])
        else:
            perp1 = np.cross(direction, [1, 0, 0])
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(direction, perp1)
        
        # Create cylinder vertices
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            offset = radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
            vertices.append(start + offset)
            vertices.append(end + offset)
        
        # Add end caps
        vertices.append(start)
        vertices.append(end)
        
        vertices = np.array(vertices)
        
        # Create faces
        faces = []
        
        # Side faces
        for i in range(segments):
            next_i = (i + 1) % segments
            # Two triangles per segment
            faces.append([2*i, 2*i+1, 2*next_i])
            faces.append([2*i+1, 2*next_i+1, 2*next_i])
        
        # End caps
        center_start_idx = len(vertices) - 2
        center_end_idx = len(vertices) - 1
        
        for i in range(segments):
            next_i = (i + 1) % segments
            faces.append([center_start_idx, 2*next_i, 2*i])
            faces.append([center_end_idx, 2*i+1, 2*next_i+1])
        
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    def _create_hull_mesh(self, positions, radius):
        """Create a convex hull mesh for complex body parts"""
        # Expand positions with radius
        expanded = []
        for pos in positions:
            center = np.array([pos[0], pos[1], 0.0])
            # Add points around each position
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                offset = radius * np.array([np.cos(angle), np.sin(angle), 0])
                expanded.append(center + offset)
        
        expanded = np.array(expanded)
        
        try:
            hull = trimesh.convex.convex_hull(expanded)
            return hull
        except:
            return None
    
    def export_glb(self, output_path='output/character_2_5d.glb'):
        """Export articulated character as GLB"""
        print("\n" + "="*60)
        print("PHASE 3: EXPORTING TO GLB")
        print("="*60)
        
        if not self.body_parts:
            raise ValueError("No body parts created yet")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Combine all body parts into one mesh
        meshes = list(self.body_parts.values())
        combined = trimesh.util.concatenate(meshes)
        
        # Add color/texture
        height, width = self.image.shape[:2]
        vertices = np.array(combined.vertices)
        
        # Simple UV mapping
        uvs = np.zeros((len(vertices), 2))
        uvs[:, 0] = np.clip(vertices[:, 0] / width, 0, 1)
        uvs[:, 1] = np.clip(1.0 - vertices[:, 1] / height, 0, 1)
        
        # Apply texture
        texture_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        texture = trimesh.visual.TextureVisuals(uv=uvs, image=texture_image)
        combined.visual = texture
        
        # Create bone hierarchy
        bones = self._create_bones()
        
        # Store metadata
        combined.metadata.update({
            'model_type': '2.5D Articulated Character',
            'joints': {k: [float(v[0]), float(v[1])] for k, v in self.joints.items()},
            'bones': bones,
            'body_parts': list(self.body_parts.keys()),
        })
        
        combined.export(output_path, file_type='glb')
        
        print(f"✓ Exported to {output_path}")
        print(f"  Vertices: {len(combined.vertices)}")
        print(f"  Faces: {len(combined.faces)}")
        print(f"  Body parts: {len(self.body_parts)}")
        print(f"  Bones: {len(bones)}")
        
    def _create_bones(self):
        """Create bone structure for rigging"""
        bones = []
        
        bone_chains = [
            ('spine', 'pelvis', 'spine', None),
            ('spine.001', 'spine', 'neck', 'spine'),
            ('head', 'neck', 'head', 'spine.001'),
            
            ('upper_arm.L', 'left_shoulder', 'left_elbow', 'spine.001'),
            ('forearm.L', 'left_elbow', 'left_wrist', 'upper_arm.L'),
            
            ('upper_arm.R', 'right_shoulder', 'right_elbow', 'spine.001'),
            ('forearm.R', 'right_elbow', 'right_wrist', 'upper_arm.R'),
            
            ('thigh.L', 'left_hip', 'left_knee', 'spine'),
            ('shin.L', 'left_knee', 'left_ankle', 'thigh.L'),
            
            ('thigh.R', 'right_hip', 'right_knee', 'spine'),
            ('shin.R', 'right_knee', 'right_ankle', 'thigh.R'),
        ]
        
        for name, head_joint, tail_joint, parent in bone_chains:
            head_pos = self.joints.get(head_joint, [0, 0])
            tail_pos = self.joints.get(tail_joint, [0, 0])
            
            bones.append({
                'name': name,
                'head': [float(head_pos[0]), float(head_pos[1]), 0.0],
                'tail': [float(tail_pos[0]), float(tail_pos[1]), 0.0],
                'parent': parent
            })
        
        return bones
    
    def process(self):
        """Run complete pipeline"""
        print("\n" + "="*70)
        print(" 2.5D ARTICULATED CHARACTER RIG ".center(70, "="))
        print("="*70)
        
        Path('output').mkdir(exist_ok=True)
        
        self.load_and_preprocess()
        self.extract_joints()
        self.create_body_part_meshes()
        self.export_glb()
        
        print("\n" + "="*70)
        print("✓ PIPELINE COMPLETE!")
        print("="*70)
        print("\nGenerated files:")
        print("  • output/character_2_5d.glb - Articulated character")
        print("  • output/mask.png - Character mask")
        print("  • output/skeleton_visualization.png - Joint structure")
        print("\n")

if __name__ == "__main__":
    input_path = '/Users/wenjia/Documents/GitHub/website/AutoRig/data/child_drawing.jpg'
    rig = TwoPointFiveDCharacterRig(input_path)
    rig.process()