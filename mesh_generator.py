import yaml
import numpy as np
import json
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.spatial import Delaunay
from collections import defaultdict
from skimage import measure
from shapely import geometry

class CharacterAutoRigger:
    def __init__(self, config_path: str, texture_path: str, mask_path: str = None):
        """
        Initialize the auto-rigger with character configuration and assets.
        
        Args:
            config_path: Path to char_cfg.yaml
            texture_path: Path to character texture image
            mask_path: Optional path to mask image (alpha channel used if None)
        """
        config_path = Path(config_path)
        texture_path = Path(texture_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path.absolute()}")
        if not texture_path.exists():
            raise FileNotFoundError(f"Texture file not found: {texture_path.absolute()}")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.texture = Image.open(texture_path).convert('RGBA')
        
        if mask_path:
            mask_path = Path(mask_path)
            if not mask_path.exists():
                raise FileNotFoundError(f"Mask file not found: {mask_path.absolute()}")
            self.mask = Image.open(mask_path).convert('L')
        else:
            # Extract alpha channel as mask
            self.mask = self.texture.split()[3]
        
        self.width = self.config['width']
        self.height = self.config['height']
        self.skeleton = self.config['skeleton']
        
        self.bones = []
        self.bone_hierarchy = {}
        self.bone_name_to_index = {}
        self.vertex_weights = []
        
    def build_bone_hierarchy(self):
        """Build bone hierarchy and bone list."""
        bone_dict = {}
        
        for idx, joint in enumerate(self.skeleton):
            bone_dict[joint['name']] = {
                'index': idx,
                'position': joint['loc'],
                'parent': joint['parent'],
                'children': []
            }
            self.bone_name_to_index[joint['name']] = idx
        
        # Build parent-child relationships
        for name, data in bone_dict.items():
            if data['parent']:
                bone_dict[data['parent']]['children'].append(name)
        
        self.bone_hierarchy = bone_dict
        
        # Create bones - each joint is a bone
        for idx, joint in enumerate(self.skeleton):
            position = joint['loc']
            parent = joint['parent']
            
            # For root or joints without parent, bone points upward
            if parent is None:
                # Find first child to determine direction
                children = bone_dict[joint['name']]['children']
                if children:
                    child_pos = bone_dict[children[0]]['position']
                    direction = np.array(child_pos) - np.array(position)
                else:
                    direction = np.array([0, -50])  # Default upward
            else:
                parent_pos = bone_dict[parent]['position']
                direction = np.array(position) - np.array(parent_pos)
            
            # Normalize and create bone length
            length = np.linalg.norm(direction)
            if length < 1:
                length = 50  # Minimum bone length
                direction = np.array([0, -50])
            
            self.bones.append({
                'name': joint['name'],
                'index': idx,
                'position': position,
                'parent': parent,
                'length': length,
                'direction': direction
            })
    
    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate mesh vertices and triangles from mask using Delaunay triangulation.
        
        Returns:
            vertices: Nx2 array of vertex positions (normalized 0-1)
            triangles: Mx3 array of triangle indices
        """
        mask_array = np.array(self.mask)
        
        # Find contours in the mask
        try:
            contours = measure.find_contours(mask_array, 128)
        except Exception as e:
            print(f'Error finding contours: {e}')
            raise
        
        # If multiple contours, use the largest one
        if len(contours) > 1:
            print(f'Found {len(contours)} contours, using largest')
            contours.sort(key=len, reverse=True)
        
        # Get the main contour
        main_contour = contours[0]
        
        # Approximate polygon to reduce vertex count
        outside_vertices = measure.approximate_polygon(main_contour, tolerance=0.25)
        
        # Create polygon for inside/outside testing
        # Note: contour coordinates are (row, col) which is (y, x)
        character_outline = geometry.Polygon([(pt[1], pt[0]) for pt in main_contour])
        
        # Generate internal points using grid sampling
        inside_vertices_xy = []
        
        # Create a denser grid for better mesh quality
        grid_resolution = 40
        _x = np.linspace(0, self.width, grid_resolution)
        _y = np.linspace(0, self.height, grid_resolution)
        xv, yv = np.meshgrid(_x, _y)
        
        # Check which grid points are inside the character
        for x, y in zip(xv.flatten(), yv.flatten()):
            if character_outline.contains(geometry.Point(x, y)):
                inside_vertices_xy.append([x, y])
        
        # Convert outside vertices from (row, col) to (x, y)
        outside_vertices_xy = np.array([[pt[1], pt[0]] for pt in outside_vertices])
        
        # Combine boundary and internal vertices
        if len(inside_vertices_xy) > 0:
            inside_vertices = np.array(inside_vertices_xy)
            vertices = np.concatenate([outside_vertices_xy, inside_vertices]).astype(np.float32)
        else:
            vertices = outside_vertices_xy.astype(np.float32)
        
        # Perform Delaunay triangulation
        tri = Delaunay(vertices)
        
        # Filter triangles - keep only those with centroid inside character
        valid_triangles = []
        for simplex in tri.simplices:
            tri_vertices = vertices[simplex]
            tri_centroid = geometry.Point(np.mean(tri_vertices, 0))
            
            if character_outline.contains(tri_centroid):
                valid_triangles.append(simplex)
        
        # Normalize vertices to 0-1 range
        #vertices_normalized = vertices.copy()
        # vertices_normalized[:, 0] /= self.width
        # vertices_normalized[:, 1] /= self.height
        
        triangles = np.array(valid_triangles)
        
        #print(f"✓ Generated mesh: {len(vertices_normalized)} vertices, {len(triangles)} triangles")
        
        return vertices, triangles
    
    def calculate_bone_weights(self, vertices: np.ndarray) -> List[Dict]:
        """
        Calculate bone weights for each vertex using distance-based weighting.
        Vertices are in normalized 0-1 space.
        
        Args:
            vertices: Nx2 array of vertex positions (normalized 0-1)
            
        Returns:
            List of weight dictionaries for each vertex
        """
        vertex_weights = []
        
        # Convert vertices back to pixel space for distance calculations
        vertices_pixel = vertices.copy()
        vertices_pixel[:, 0] *= self.width
        vertices_pixel[:, 1] *= self.height
        
        for vertex in vertices_pixel:
            weights = {}
            
            # Calculate distance to each bone
            for bone in self.bones:
                position = np.array(bone['position'])
                
                # Calculate distance from vertex to bone position
                distance = np.linalg.norm(vertex - position)
                
                # Weight inversely proportional to distance
                if distance < 1.0:
                    distance = 1.0
                
                # Use squared distance for smoother falloff
                weight = 1.0 / (distance ** 1.5)
                weights[bone['name']] = weight
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                for bone_name in weights:
                    weights[bone_name] /= total_weight
            
            # Keep only top 4 influences (Unity standard)
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:4]
            normalized_weights = {}
            
            weight_sum = sum(w for _, w in sorted_weights)
            if weight_sum > 0:
                for bone_name, weight in sorted_weights:
                    normalized_weights[bone_name] = weight / weight_sum
            
            vertex_weights.append(normalized_weights)
        
        return vertex_weights
    
    def export_to_unity_format(self, output_dir: str):
        """
        Export rigged character in Unity-compatible format with full skinning.
        
        Args:
            output_dir: Directory to save exported files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Build bone hierarchy
        self.build_bone_hierarchy()
        
        # Generate mesh
        vertices, triangles = self.generate_mesh()
        
        # Calculate bone weights
        vertex_weights = self.calculate_bone_weights(vertices)
        
        # Export texture
        texture_path = output_path / "character_texture.png"
        self.texture.save(texture_path)
        
        # Convert vertices to Unity coordinate system (center origin, Y-up)
        unity_vertices = vertices.astype(np.float64).copy()
        unity_vertices[:, 0] -= self.width / 2  # Center X
        unity_vertices[:, 1] -= self.height / 2  # Center Y
        unity_vertices /= 100.0  # Scale down to Unity units
        
        # Convert to UV coordinates (0-1 range)
        uv_coords = vertices.astype(np.float64).copy()
        uv_coords[:, 0] /= self.width
        uv_coords[:, 1] = 1.0 - (uv_coords[:, 1] / self.height)  # Flip V
        
        # Export mesh data with bone weights in BOTH formats for compatibility
        mesh_data = {
            "vertices": unity_vertices.tolist(),
            "triangles": triangles.astype(np.int32).tolist(),
            "uvs": uv_coords.tolist(),
            "weights": vertex_weights,  # Old format (dict) - keep for compatibility
            "boneWeights": []  # New format (array) - for Unity
        }
        
        # Convert bone weights to Unity BoneWeight format
        for weights in vertex_weights:
            bone_weight = {
                "boneIndex0": 0,
                "boneIndex1": 0,
                "boneIndex2": 0,
                "boneIndex3": 0,
                "weight0": 0.0,
                "weight1": 0.0,
                "weight2": 0.0,
                "weight3": 0.0
            }
            
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            for i, (bone_name, weight) in enumerate(sorted_weights[:4]):
                bone_weight[f"boneIndex{i}"] = self.bone_name_to_index[bone_name]
                bone_weight[f"weight{i}"] = float(weight)
            
            mesh_data["boneWeights"].append(bone_weight)
        
        mesh_path = output_path / "character_mesh.json"
        with open(mesh_path, 'w') as f:
            json.dump(mesh_data, f, indent=2)
        
        # Export skeleton data with bind poses
        skeleton_data = {
            "bones": [],
            "bindPoses": []
        }
        
        for bone in self.bones:
            # Convert bone position to Unity coordinates
            bone_pos = np.array(bone['position'], dtype=np.float64)
            bone_pos[0] -= self.width / 2
            bone_pos[1] -= self.height / 2
            bone_pos /= 100.0
            
            bone_data = {
                "name": bone['name'],
                "parent": bone['parent'],
                "index": bone['index'],
                "localPosition": {
                    "x": float(bone_pos[0]),
                    "y": float(bone_pos[1]),
                    "z": 0.0
                }
            }
            skeleton_data["bones"].append(bone_data)
            
            # Create bind pose (inverse of bone's world transform)
            bind_pose = {
                "m00": 1.0, "m01": 0.0, "m02": 0.0, "m03": -float(bone_pos[0]),
                "m10": 0.0, "m11": 1.0, "m12": 0.0, "m13": -float(bone_pos[1]),
                "m20": 0.0, "m21": 0.0, "m22": 1.0, "m23": 0.0,
                "m30": 0.0, "m31": 0.0, "m32": 0.0, "m33": 1.0
            }
            skeleton_data["bindPoses"].append(bind_pose)
        
        skeleton_path = output_path / "character_skeleton.json"
        with open(skeleton_path, 'w') as f:
            json.dump(skeleton_data, f, indent=2)
        
        # Generate Unity C# import script with SkinnedMeshRenderer
        self._generate_unity_import_script(output_path)
        
        print(f"✓ Exported character to {output_path}")
        print(f"  - Texture: {texture_path.name}")
        print(f"  - Mesh: {mesh_path.name} ({len(vertices)} vertices, {len(triangles)} triangles)")
        print(f"  - Skeleton: {skeleton_path.name} ({len(self.bones)} bones)")
        print(f"  - Import script: CharacterImporter.cs")
        print(f"\n✓ Full skinned mesh with bone weights enabled!")
    
    def _generate_unity_import_script(self, output_path: Path):
        """Generate Unity C# script for importing with SkinnedMeshRenderer."""
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


if __name__ == "__main__":
    import sys
    
    # Example usage
    print("Current working directory:", Path.cwd())
    print("\nLooking for files:")
    print("  - char_cfg.yaml")
    print("  - character_texture.png")
    print("  - character_mask.png (optional)")
    print()
    
    try:
        path = "/Users/wenjia/Documents/GitHub/website/AutoRig/char7"
        rigger = CharacterAutoRigger(
            config_path= path + "/char_cfg.yaml",
            texture_path= path + "/texture.png",
            mask_path= path + "/mask.png"
        )
        
        rigger.export_to_unity_format("unitymeshoutput")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure these files are in the same directory as the script:")
        print("  - char_cfg.yaml")
        print("  - character_texture.png")
        print("  - character_mask.png (optional)")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
