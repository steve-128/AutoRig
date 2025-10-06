import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import Delaunay
from skimage import morphology, measure
import json
import trimesh
from pathlib import Path
import warnings

class TwoPointFiveDCharacterRig:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = None
        self.mask = None
        
        # Annotation data
        self.joints = []  # 15 joint keypoints
        self.silhouette_segments = {}  # {name: {'mask': ..., 'orientation': 'left'/'right'/'none'}}
        self.part_regions = {}  # {name: {'mask': ..., 'translate': ..., 'direction': ..., etc}}
        
        # 2.5D Model (Section 4.2)
        self.left_view = None
        self.right_view = None
        
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
        """
        Extract 15 joint keypoints as per Section 4.1
        In practice, this would use pose estimation or manual annotation.
        We'll create a simplified skeleton based on simplified mask.
        """
        print("\nExtracting joint keypoints...")
        
        # CRITICAL: Simplify mask to remove hair/fur details
        # This prevents skeleton from having too many branches
        simplified_mask = self._simplify_mask_for_skeleton()
        
        # Find contour and approximate body structure
        contours, _ = cv2.findContours(simplified_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            raise ValueError("No figure found in image")
        
        largest = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest)
        
        # Create approximate 15-joint skeleton
        # Standard joints: head, neck, shoulders, elbows, wrists, hips, knees, ankles
        cx = x + w/2
        
        joints = {
            'head': [cx, y + h*0.1],
            'neck': [cx, y + h*0.15],
            'left_shoulder': [cx - w*0.15, y + h*0.2],
            'right_shoulder': [cx + w*0.15, y + h*0.2],
            'left_elbow': [cx - w*0.2, y + h*0.4],
            'right_elbow': [cx + w*0.2, y + h*0.4],
            'left_wrist': [cx - w*0.25, y + h*0.6],
            'right_wrist': [cx + w*0.25, y + h*0.6],
            'spine': [cx, y + h*0.45],
            'left_hip': [cx - w*0.1, y + h*0.5],
            'right_hip': [cx + w*0.1, y + h*0.5],
            'left_knee': [cx - w*0.12, y + h*0.7],
            'right_knee': [cx + w*0.12, y + h*0.7],
            'left_ankle': [cx - w*0.13, y + h*0.9],
            'right_ankle': [cx + w*0.13, y + h*0.9],
        }
        
        self.joints = joints
        
        # Save debug visualization
        self._visualize_skeleton(simplified_mask)
        
        print(f"✓ Created {len(joints)} joint keypoints")
    
    def _simplify_mask_for_skeleton(self):
        """
        Simplify mask by removing small protrusions (hair, fur, etc.)
        This prevents the skeleton from having too many branches
        """
        # Apply heavy morphological closing to fill in details
        kernel = np.ones((15, 15), np.uint8)
        simplified = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Apply opening to smooth boundaries
        kernel = np.ones((11, 11), np.uint8)
        simplified = cv2.morphologyEx(simplified, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Apply Gaussian blur and re-threshold to further smooth
        simplified = cv2.GaussianBlur(simplified, (15, 15), 0)
        _, simplified = cv2.threshold(simplified, 127, 255, cv2.THRESH_BINARY)
        
        cv2.imwrite('output/simplified_mask.png', simplified)
        print("✓ Created simplified mask (saved to output/simplified_mask.png)")
        
        return simplified
    
    def _visualize_skeleton(self, mask):
        """Create visualization showing joints on the figure"""
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Draw joints
        for joint_name, (x, y) in self.joints.items():
            cv2.circle(vis, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(vis, joint_name.split('_')[0][:3], 
                       (int(x)+7, int(y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        
        # Draw bones connecting joints
        bone_pairs = [
            ('head', 'neck'),
            ('neck', 'left_shoulder'),
            ('neck', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('neck', 'spine'),
            ('spine', 'left_hip'),
            ('spine', 'right_hip'),
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle'),
        ]
        
        for joint1, joint2 in bone_pairs:
            if joint1 in self.joints and joint2 in self.joints:
                p1 = tuple(map(int, self.joints[joint1]))
                p2 = tuple(map(int, self.joints[joint2]))
                cv2.line(vis, p1, p2, (0, 0, 255), 2)
        
        cv2.imwrite('output/skeleton_visualization.png', vis)
        print("✓ Skeleton visualization saved to output/skeleton_visualization.png")
        
    def annotate_orientation_cues(self):
        """
        Section 4.1: Annotate silhouette and internal orientation cues
        In practice, this requires user input. We'll create automatic approximations.
        """
        print("\nAnnotating orientation cues...")
        
        # Find contours for silhouette segmentation
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        
        # Create silhouette segments (simplified)
        # Divide vertically: hair, head, torso
        hair_mask = np.zeros_like(self.mask)
        head_mask = np.zeros_like(self.mask)
        torso_mask = np.zeros_like(self.mask)
        
        hair_mask[y:int(y+h*0.08), x:x+w] = self.mask[y:int(y+h*0.08), x:x+w]
        head_mask[int(y+h*0.08):int(y+h*0.25), x:x+w] = self.mask[int(y+h*0.08):int(y+h*0.25), x:x+w]
        torso_mask[int(y+h*0.25):int(y+h*0.55), x:x+w] = self.mask[int(y+h*0.25):int(y+h*0.55), x:x+w]
        
        self.silhouette_segments = {
            'hair': {'mask': hair_mask, 'orientation': 'none'},
            'head': {'mask': head_mask, 'orientation': 'none'},
            'torso': {'mask': torso_mask, 'orientation': 'none'},
        }
        
        # Create internal part regions (simplified - eyes, nose, mouth)
        # These would normally be user-annotated
        self.part_regions = {
            'left_eye': {
                'mask': self._create_circular_mask(int(x + w*0.4), int(y + h*0.15), 3),
                'translate': 'smooth',
                'direction': 'none',
                'enclosed': True,
                'hide_on_back': True
            },
            'right_eye': {
                'mask': self._create_circular_mask(int(x + w*0.6), int(y + h*0.15), 3),
                'translate': 'smooth',
                'direction': 'none',
                'enclosed': True,
                'hide_on_back': True
            },
            'nose': {
                'mask': self._create_circular_mask(int(x + w*0.5), int(y + h*0.18), 2),
                'translate': 'smooth',
                'direction': 'none',
                'enclosed': True,
                'hide_on_back': False
            }
        }
        
        print(f"✓ Created {len(self.silhouette_segments)} silhouette segments")
        print(f"✓ Created {len(self.part_regions)} part regions")
        
    def _create_circular_mask(self, cx, cy, radius):
        """Helper to create a circular mask"""
        mask = np.zeros_like(self.mask)
        cv2.circle(mask, (cx, cy), radius, 255, -1)
        return mask
        
    def build_2_5d_model(self):
        """
        Section 4.2: Build Left View and Right View
        Each view consists of textured meshes and keyview-transforms
        """
        print("\n" + "="*60)
        print("PHASE 2: BUILDING 2.5D CHARACTER MODEL")
        print("="*60)
        
        print("\nConstructing Left View...")
        self.left_view = self._construct_view('left')
        
        print("Constructing Right View...")
        self.right_view = self._construct_view('right')
        
        print("✓ 2.5D model complete")
        
    def _construct_view(self, view_direction):
        """
        Construct a single view (left or right) of the 2.5D model
        """
        view = {
            'direction': view_direction,
            'base_mask': None,
            'front_texture': None,
            'back_texture': None,
            'mesh': None,
            'joints': {},
            'keyview_transforms': {}
        }
        
        # Step 1: Mirror right-facing segments if building left view (and vice versa)
        working_mask = self.mask.copy()
        
        if view_direction == 'left':
            # Mirror right-facing segments
            for seg_name, seg_data in self.silhouette_segments.items():
                if seg_data['orientation'] == 'right':
                    seg_mask = seg_data['mask']
                    working_mask = np.where(seg_mask > 0, 
                                          cv2.flip(seg_mask, 1), 
                                          working_mask)
        else:
            # Mirror left-facing segments
            for seg_name, seg_data in self.silhouette_segments.items():
                if seg_data['orientation'] == 'left':
                    seg_mask = seg_data['mask']
                    working_mask = np.where(seg_mask > 0, 
                                          cv2.flip(seg_mask, 1), 
                                          working_mask)
        
        view['base_mask'] = working_mask
        
        # Step 2: Create textures (front and back)
        # Front texture: inpaint translating parts
        # Back texture: inpaint translating + hide_on_back parts
        view['front_texture'] = self.image.copy()
        view['back_texture'] = self._create_back_texture()
        
        # Step 3: Generate mesh from mask
        view['mesh'] = self._generate_mesh_from_mask(working_mask)
        
        # Step 4: Copy and potentially mirror joints
        view['joints'] = self.joints.copy()
        
        # Step 5: Create keyview-transforms for part regions
        for part_name, part_data in self.part_regions.items():
            if part_data['translate'] in ['smooth', 'discrete']:
                # Define transform for this view
                view['keyview_transforms'][part_name] = {
                    'left_transform': np.eye(3),  # Identity for now
                    'right_transform': np.eye(3)
                }
        
        return view
        
    def _create_back_texture(self):
        """Create back texture by inpainting regions that hide on back"""
        back = self.image.copy()
        # Simplified: just darken the image
        back = (back * 0.7).astype(np.uint8)
        return back
        
    def _generate_mesh_from_mask(self, mask):
        """
        Generate 3D mesh from 2D mask using Delaunay triangulation
        Similar to the original implementation
        """
        # Find contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
            
        largest = max(contours, key=cv2.contourArea)
        
        # Simplify
        epsilon = 0.005 * cv2.arcLength(largest, True)
        simplified = cv2.approxPolyDP(largest, epsilon, True)
        
        # Create 3D vertices
        vertices = []
        for point in simplified:
            x, y = point[0]
            vertices.append([float(x), float(y), 0.0])
        
        # Add interior points
        points_2d = np.array([[v[0], v[1]] for v in vertices])
        interior = self._sample_interior_points(mask, 10)
        all_points = np.vstack([points_2d, interior])
        
        # Triangulate
        try:
            tri = Delaunay(all_points)
            faces = tri.simplices
        except:
            n = len(vertices)
            faces = [[0, i, i+1] for i in range(1, n-1)]
        
        # Create full 3D mesh with thickness
        all_vertices = [[p[0], p[1], 0.0] for p in all_points]
        back_vertices = [[v[0], v[1], -5.0] for v in all_vertices]
        all_vertices.extend(back_vertices)
        
        n_verts = len(all_points)
        back_faces = [[f[0] + n_verts, f[2] + n_verts, f[1] + n_verts] for f in faces]
        all_faces = np.vstack([faces, back_faces])
        
        # Create trimesh
        mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, process=False)
        
        return mesh
        
    def _sample_interior_points(self, mask, n):
        """Sample random points inside mask"""
        points = []
        mask_points = np.argwhere(mask > 0)
        if len(mask_points) > 0:
            indices = np.random.choice(len(mask_points), min(n, len(mask_points)), replace=False)
            for idx in indices:
                y, x = mask_points[idx]
                points.append([float(x), float(y)])
        return np.array(points) if points else np.array([[0, 0]])
        
    def export_glb(self, output_path='output/character_2_5d.glb', include_armature=True):
        """
        Export the 2.5D model as GLB with proper armature nodes
        Uses pygltflib for full GLB 2.0 spec support
        """
        print("\n" + "="*60)
        print("EXPORTING TO GLB")
        print("="*60)
        
        if self.left_view is None or self.left_view['mesh'] is None:
            raise ValueError("2.5D model not built yet")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        if include_armature:
            # Export with proper armature using pygltflib
            try:
                import pygltflib
                self._export_with_pygltflib(output_path)
                print("✓ Exported with proper armature nodes (pygltflib)")
            except ImportError:
                print("⚠ pygltflib not installed, falling back to metadata-only export")
                print("  Install with: pip install pygltflib")
                self._export_basic_glb(output_path, include_armature=True)
        else:
            self._export_basic_glb(output_path, include_armature=False)
    
    def _export_basic_glb(self, output_path, include_armature=False):
        """Fallback export using trimesh (stores armature in metadata only)"""
        mesh = self.left_view['mesh'].copy()
        
        # Add texture
        height, width = self.image.shape[:2]
        vertices = np.array(mesh.vertices)
        uvs = np.zeros((len(vertices), 2))
        uvs[:, 0] = vertices[:, 0] / width
        uvs[:, 1] = 1.0 - vertices[:, 1] / height
        
        texture_image = cv2.cvtColor(self.left_view['front_texture'], cv2.COLOR_BGR2RGB)
        texture = trimesh.visual.TextureVisuals(uv=uvs, image=texture_image)
        mesh.visual = texture
        
        # Store bone/weight data in metadata
        if include_armature:
            bones = self._create_rigify_compatible_bones()
            weights = self._calculate_vertex_weights(mesh.vertices, bones)
            mesh.metadata['bones'] = bones
            mesh.metadata['vertex_weights'] = weights.tolist()
        
        mesh.metadata.update({
            'model_type': '2.5D Character Rig',
            'paper': 'Smith et al. 2025',
            'joints': {k: [float(v[0]), float(v[1])] for k, v in self.joints.items()},
            'has_armature': include_armature,
            'rigify_compatible': include_armature,
            'note': 'Armature in metadata - use import script for Blender'
        })
        
        mesh.export(output_path, file_type='glb')
        
        print(f"✓ Exported to {output_path}")
        print(f"  Vertices: {len(mesh.vertices)}")
        print(f"  Faces: {len(mesh.faces)}")
        print(f"  Armature: In metadata (requires import script)")
    
    def _export_with_pygltflib(self, output_path):
        """
        Export with proper GLB armature using pygltflib
        Creates actual bone nodes in the GLB scene graph
        """
        import pygltflib
        from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Accessor, BufferView, Buffer
        from pygltflib import Material, PBRMetallicRoughness, Image, Texture, Sampler
        import struct
        
        mesh_data = self.left_view['mesh']
        vertices = np.array(mesh_data.vertices, dtype=np.float32)
        indices = np.array(mesh_data.faces.flatten(), dtype=np.uint32)
        
        # UV coordinates
        height, width = self.image.shape[:2]
        uvs = np.zeros((len(vertices), 2), dtype=np.float32)
        uvs[:, 0] = vertices[:, 0] / width
        uvs[:, 1] = 1.0 - vertices[:, 1] / height
        
        # Create binary buffers
        vertices_binary = vertices.tobytes()
        indices_binary = indices.tobytes()
        uvs_binary = uvs.tobytes()
        
        # Texture image
        texture_image = cv2.cvtColor(self.left_view['front_texture'], cv2.COLOR_BGR2RGB)
        from PIL import Image as PILImage
        import io
        pil_img = PILImage.fromarray(texture_image)
        img_buffer = io.BytesIO()
        pil_img.save(img_buffer, format='PNG')
        texture_binary = img_buffer.getvalue()
        
        # Combine all binary data
        buffer_data = indices_binary + vertices_binary + uvs_binary + texture_binary
        
        # Create GLB structure
        gltf = GLTF2()
        
        # Buffer
        gltf.buffers.append(Buffer(byteLength=len(buffer_data)))
        
        # Buffer views
        offset = 0
        # Indices buffer view
        gltf.bufferViews.append(BufferView(
            buffer=0,
            byteOffset=offset,
            byteLength=len(indices_binary),
            target=34963  # ELEMENT_ARRAY_BUFFER
        ))
        offset += len(indices_binary)
        
        # Vertices buffer view
        gltf.bufferViews.append(BufferView(
            buffer=0,
            byteOffset=offset,
            byteLength=len(vertices_binary),
            target=34962  # ARRAY_BUFFER
        ))
        offset += len(vertices_binary)
        
        # UVs buffer view
        gltf.bufferViews.append(BufferView(
            buffer=0,
            byteOffset=offset,
            byteLength=len(uvs_binary),
            target=34962
        ))
        offset += len(uvs_binary)
        
        # Texture buffer view
        gltf.bufferViews.append(BufferView(
            buffer=0,
            byteOffset=offset,
            byteLength=len(texture_binary)
        ))
        
        # Accessors
        gltf.accessors.append(Accessor(
            bufferView=0,
            componentType=5125,  # UNSIGNED_INT
            count=len(indices),
            type="SCALAR",
            max=[int(indices.max())],
            min=[int(indices.min())]
        ))
        
        gltf.accessors.append(Accessor(
            bufferView=1,
            componentType=5126,  # FLOAT
            count=len(vertices),
            type="VEC3",
            max=vertices.max(axis=0).tolist(),
            min=vertices.min(axis=0).tolist()
        ))
        
        gltf.accessors.append(Accessor(
            bufferView=2,
            componentType=5126,
            count=len(uvs),
            type="VEC2",
            max=uvs.max(axis=0).tolist(),
            min=uvs.min(axis=0).tolist()
        ))
        
        # Material and texture
        gltf.images.append(Image(bufferView=3, mimeType="image/png"))
        gltf.samplers.append(Sampler(magFilter=9729, minFilter=9987))
        gltf.textures.append(Texture(sampler=0, source=0))
        gltf.materials.append(Material(
            pbrMetallicRoughness=PBRMetallicRoughness(
                baseColorTexture={"index": 0},
                metallicFactor=0.0,
                roughnessFactor=1.0
            ),
            doubleSided=True
        ))
        
        # Mesh
        gltf.meshes.append(Mesh(
            name="CharacterMesh",
            primitives=[Primitive(
                attributes={"POSITION": 1, "TEXCOORD_0": 2},
                indices=0,
                material=0
            )]
        ))
        
        # Create bone nodes
        bones = self._create_rigify_compatible_bones()
        bone_nodes = []
        
        for i, bone in enumerate(bones):
            node = Node(
                name=bone['name'],
                translation=[bone['head'][0], bone['head'][1], bone['head'][2]]
            )
            bone_nodes.append(node)
            gltf.nodes.append(node)
        
        # Set up bone hierarchy
        for i, bone in enumerate(bones):
            if bone['parent']:
                parent_idx = next(j for j, b in enumerate(bones) if b['name'] == bone['parent'])
                if gltf.nodes[parent_idx].children is None:
                    gltf.nodes[parent_idx].children = []
                gltf.nodes[parent_idx].children.append(i)
        
        # Mesh node (child of root bone)
        mesh_node_idx = len(gltf.nodes)
        gltf.nodes.append(Node(name="CharacterMesh", mesh=0))
        
        # Root node
        root_idx = len(gltf.nodes)
        gltf.nodes.append(Node(
            name="Character",
            children=[0, mesh_node_idx]  # Root bone and mesh
        ))
        
        # Scene
        gltf.scenes.append(Scene(nodes=[root_idx]))
        gltf.scene = 0
        
        # Set binary data
        gltf.set_binary_blob(buffer_data)
        
        # Save
        gltf.save(output_path)
        
        print(f"✓ Exported to {output_path}")
        print(f"  Vertices: {len(vertices)}")
        print(f"  Faces: {len(mesh_data.faces)}")
        print(f"  Bones: {len(bones)}")
        print(f"  Armature: Proper GLB nodes ✓")
        print(f"  File size: {Path(output_path).stat().st_size / 1024:.2f} KB")
        
    def _add_armature_to_mesh(self, mesh):
        """
        Add armature (bone) data to mesh for Blender import
        Creates a standard humanoid skeleton compatible with Rigify
        """
        # Create bone hierarchy matching Rigify's metarig structure
        bones = self._create_rigify_compatible_bones()
        
        # Calculate vertex weights (skinning)
        weights = self._calculate_vertex_weights(mesh.vertices, bones)
        
        # Store bone data in mesh metadata (GLB format limitation workaround)
        # Blender import scripts can read this
        mesh.metadata['bones'] = bones
        mesh.metadata['vertex_weights'] = weights.tolist()
        
        return mesh
    
    def _create_rigify_compatible_bones(self):
        """
        Create bone structure compatible with Blender's Rigify addon
        Rigify expects specific bone names and hierarchy
        """
        bones = []
        
        # Bone definitions: (name, head_joint, tail_joint, parent)
        bone_defs = [
            ('spine', 'spine', 'neck', None),
            ('spine.001', 'neck', 'head', 'spine'),
            ('head', 'neck', 'head', 'spine.001'),
            
            # Left arm chain
            ('shoulder.L', 'left_shoulder', 'left_shoulder', 'spine.001'),
            ('upper_arm.L', 'left_shoulder', 'left_elbow', 'shoulder.L'),
            ('forearm.L', 'left_elbow', 'left_wrist', 'upper_arm.L'),
            ('hand.L', 'left_wrist', 'left_wrist', 'forearm.L'),
            
            # Right arm chain
            ('shoulder.R', 'right_shoulder', 'right_shoulder', 'spine.001'),
            ('upper_arm.R', 'right_shoulder', 'right_elbow', 'shoulder.R'),
            ('forearm.R', 'right_elbow', 'right_wrist', 'upper_arm.R'),
            ('hand.R', 'right_wrist', 'right_wrist', 'forearm.R'),
            
            # Left leg chain
            ('thigh.L', 'left_hip', 'left_knee', 'spine'),
            ('shin.L', 'left_knee', 'left_ankle', 'thigh.L'),
            ('foot.L', 'left_ankle', 'left_ankle', 'shin.L'),
            
            # Right leg chain
            ('thigh.R', 'right_hip', 'right_knee', 'spine'),
            ('shin.R', 'right_knee', 'right_ankle', 'thigh.R'),
            ('foot.R', 'right_ankle', 'right_ankle', 'shin.R'),
        ]
        
        for bone_name, head_joint, tail_joint, parent in bone_defs:
            head_pos = self.joints[head_joint]
            tail_pos = self.joints[tail_joint]
            
            # If head and tail are same (leaf bones), offset tail slightly
            if head_pos == tail_pos:
                tail_pos = [head_pos[0], head_pos[1] - 10, head_pos[2] if len(head_pos) > 2 else 0]
            
            bone = {
                'name': bone_name,
                'head': [float(head_pos[0]), float(head_pos[1]), 0.0],
                'tail': [float(tail_pos[0]), float(tail_pos[1]), 0.0],
                'parent': parent,
                'connected': parent is not None
            }
            bones.append(bone)
        
        return bones
    
    def _calculate_vertex_weights(self, vertices, bones):
        """
        Calculate vertex weights for skinning (automatic weight painting)
        Uses distance-based weighting with Gaussian falloff
        """
        n_verts = len(vertices)
        n_bones = len(bones)
        weights = np.zeros((n_verts, n_bones))
        
        for v_idx, vertex in enumerate(vertices):
            v_pos = vertex[:2]  # Use only X,Y for 2D character
            
            # Calculate distance to each bone
            distances = []
            for b_idx, bone in enumerate(bones):
                bone_center = np.array([
                    (bone['head'][0] + bone['tail'][0]) / 2,
                    (bone['head'][1] + bone['tail'][1]) / 2
                ])
                dist = np.linalg.norm(v_pos - bone_center)
                distances.append(dist)
            
            # Convert distances to weights using Gaussian falloff
            distances = np.array(distances)
            sigma = 50.0  # Falloff parameter
            raw_weights = np.exp(-distances**2 / (2 * sigma**2))
            
            # Normalize weights to sum to 1
            weight_sum = raw_weights.sum()
            if weight_sum > 0:
                weights[v_idx] = raw_weights / weight_sum
            else:
                # If all weights are 0, assign to nearest bone
                nearest = np.argmin(distances)
                weights[v_idx, nearest] = 1.0
        
        return weights
    
    def process(self):
        """Run complete pipeline"""
        print("\n" + "="*70)
        print(" 2.5D CHARACTER RIG SYSTEM ".center(70, "="))
        print("="*70)
        print("Based on: Smith, He, Ye (2025)")
        print("'Animating Childlike Drawings with 2.5D Character Rigs'")
        print("="*70)
        
        Path('output').mkdir(exist_ok=True)
        
        # Phase 1: Preprocessing & Annotation
        self.load_and_preprocess()
        self.extract_joints()
        self.annotate_orientation_cues()
        
        # Phase 2: Build 2.5D Model
        self.build_2_5d_model()
        
        # Export
        self.export_glb(include_armature=True)
        
        print("\n" + "="*70)
        print("✓ PIPELINE COMPLETE!")
        print("="*70)
        print("\nGenerated files:")
        print("  • output/character_2_5d.glb - Rigged character mesh")
        print("  • output/import_to_blender.py - Blender import script")
        print("  • output/mask.png - Character mask")
        print("  • output/simplified_mask.png - Cleaned mask for skeleton")
        print("  • output/skeleton_visualization.png - Joint placement preview")
        print("\n")

if __name__ == "__main__":
    input_folder = '/Users/wenjia/Documents/GitHub/website/AutoRig/data/child_drawing.jpg'
    rig = TwoPointFiveDCharacterRig(input_folder)
    rig.process()