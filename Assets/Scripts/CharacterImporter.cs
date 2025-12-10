using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Newtonsoft.Json;

[ExecuteInEditMode]
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

    [Header("Assets from Python export")]
    public Texture2D characterTexture;
    public TextAsset meshDataFile;
    public TextAsset skeletonDataFile;

    [Header("Runtime options")]
    public bool importOnStart = false;

    [Header("BVH / Animation options")]
    public TextAsset bvhFile;
    public string bvhFilePath;
    public bool addAnimator = true;
    public float bvhScale = 0.025f;

    void Start()
    {
        // Optional: only auto-import when playing, and only if you want it
        if (importOnStart && Application.isPlaying)
        {
            ImportCharacterRuntime();
        }
    }

    public GameObject ImportCharacterRuntime()
    {
        return ImportInternal(inEditor: false);
    }

    public GameObject ImportCharacterFromEditor()
    {
        return ImportInternal(inEditor: true);
    }

    private static GameObject currentCharacterRoot;
    private GameObject ImportInternal(bool inEditor)
    {
        if (meshDataFile == null || skeletonDataFile == null || characterTexture == null)
        {
            Debug.LogError("CharacterImporter: meshDataFile, skeletonDataFile, or characterTexture is missing.");
            return null;
        }

        currentCharacterRoot = ImportCharacter();
        return currentCharacterRoot;
    }

    private GameObject ImportCharacter()
    {
        // Parse JSON data
        MeshData meshData = JsonConvert.DeserializeObject<MeshData>(meshDataFile.text);
        SkeletonData skeletonData = JsonConvert.DeserializeObject<SkeletonData>(skeletonDataFile.text);

        if (meshData == null || skeletonData == null || skeletonData.bones == null || skeletonData.bones.Count == 0)
        {
            Debug.LogError("CharacterImporter: Failed to parse mesh or skeleton data.");
            return null;
        }

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
            boneObj.transform.SetParent(characterRoot.transform, false);
            boneObj.transform.localPosition = new Vector3(
                boneData.localPosition.x,
                boneData.localPosition.y,
                boneData.localPosition.z
            );

            boneTransforms[boneData.name] = boneObj.transform;
            boneArray[boneData.index] = boneObj.transform;
        }
        
        // Second pass: set up hierarchy
        Transform rootBone = null;
        foreach (var boneData in skeletonData.bones)
        {
            Transform bone = boneTransforms[boneData.name];
            if (!string.IsNullOrEmpty(boneData.parent) &&
                boneTransforms.TryGetValue(boneData.parent, out var parent))
            {
                bone.SetParent(parent, true); 
            }
            else
            {
                bone.SetParent(characterRoot.transform, true);
                rootBone = bone;
            }
        }
        
        if (rootBone == null)
        {
            // fallback: first bone
            rootBone = boneArray[0];
        }

        // Create mesh
        Mesh mesh = new Mesh();
        mesh.name = "CharacterMesh";
        
        // Set vertices
        Vector3[] vertices = new Vector3[meshData.vertices.Count];
        for (int i = 0; i < meshData.vertices.Count; i++)
        {
            var v = meshData.vertices[i];
            vertices[i] = new Vector3(v[0], v[1], 0f);
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

        //set bind poses
        GameObject meshObj = new GameObject("SkinnedMesh");
        meshObj.transform.SetParent(characterRoot.transform, false);
        meshObj.transform.localPosition = Vector3.zero;

        SkinnedMeshRenderer skinnedRenderer = meshObj.AddComponent<SkinnedMeshRenderer>();
        skinnedRenderer.sharedMesh = mesh;
        skinnedRenderer.bones = boneArray;
        skinnedRenderer.rootBone = rootBone;
        skinnedRenderer.updateWhenOffscreen = true;

        Matrix4x4[] bindPoses = new Matrix4x4[boneArray.Length];
        Transform rendererTransform = meshObj.transform;
        for (int i = 0; i < boneArray.Length; i++)
        {
            if (boneArray[i] != null)
            {
                bindPoses[i] = boneArray[i].worldToLocalMatrix * rendererTransform.localToWorldMatrix;
            }
            else
            {
                bindPoses[i] = Matrix4x4.identity;
            }
        }
        mesh.bindposes = bindPoses;

        // Material
        Material mat = new Material(Shader.Find("Sprites/Default"));
        mat.mainTexture = characterTexture;
        skinnedRenderer.material = mat;

        mesh.RecalculateBounds();
        mesh.RecalculateNormals();

        Debug.Log("Success: Character imported successfully with full skinned mesh!");
        Debug.Log($"  - {vertices.Length} vertices");
        Debug.Log($"  - {triangles.Length / 3} triangles");
        Debug.Log($"  - {boneArray.Length} bones");
        Debug.Log($"  - Skinning enabled: bones will deform mesh!");

        // Attach animator and pass data
        if (addAnimator) {

            string bvhContent = null;

            if (bvhFile != null) {
                bvhContent = bvhFile.text;
            }
            else if (!string.IsNullOrEmpty(bvhFilePath)) {
                StartCoroutine(LoadBVHFromStreamingAssets(characterRoot, boneTransforms, bvhFilePath, bvhScale));
                return characterRoot;
            }

            var anim = characterRoot.GetComponent<CharacterAnimator>() 
                        ?? characterRoot.AddComponent<CharacterAnimator>();
            anim.SetBoneTransforms(boneTransforms);
            anim.characterRoot = characterRoot.transform;

            if (bvhContent != null) {
                anim.ParseBVH(bvhContent, bvhScale);
                Debug.Log("Character Imported with BVH Animation!");
                anim.currentMode = CharacterAnimator.AnimationMode.BVH;
            } else {
                Debug.Log("Character Imported (No BVH - Ready for Preset Animations)");
                anim.currentMode = CharacterAnimator.AnimationMode.Squish;
            }
        }
        
        return characterRoot;
    }

    private IEnumerator LoadBVHFromStreamingAssets(GameObject characterRoot, Dictionary<string, Transform> boneTransforms, string filePath, float scale)
    {
        string fullPath = System.IO.Path.Combine(Application.streamingAssetsPath, filePath);

        #if UNITY_ANDROID && !UNITY_EDITOR
        using (var www = UnityEngine.Networking.UnityWebRequest.Get(fullPath))
        {
            yield return www.SendWebRequest();

            if (www.result == UnityEngine.Networking.UnityWebRequest.Result.Success)
            {
                var anim = characterRoot.GetComponent<CharacterAnimator>()
                        ?? characterRoot.AddComponent<CharacterAnimator>();

                anim.SetBoneTransforms(boneTransforms);
                anim.characterRoot = characterRoot.transform;
                anim.ParseBVH(www.downloadHandler.text, scale);
                anim.currentMode = CharacterAnimator.AnimationMode.Squish;

                Debug.Log("BVH Loaded from StreamingAssets (Android) - Defaulting to Squish");
            }
            else
            {
                Debug.LogError($"Failed to load BVH: {www.error}");
            }
        }
        #else
        if (System.IO.File.Exists(fullPath))
        {
            string bvhContent = System.IO.File.ReadAllText(fullPath);

            var anim = characterRoot.GetComponent<CharacterAnimator>()
                    ?? characterRoot.AddComponent<CharacterAnimator>();
            
            anim.SetBoneTransforms(boneTransforms);
            anim.characterRoot = characterRoot.transform;
            anim.ParseBVH(bvhContent, scale);
            anim.currentMode = CharacterAnimator.AnimationMode.Squish;

            Debug.Log("BVH Loaded from StreamingAssets - Defaulting to Squish");
        }
        else
        {
            Debug.LogError($"BVH file not found: {fullPath}");
        }
        yield break;
        #endif
    }
}

