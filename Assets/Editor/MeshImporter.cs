#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;
using System.IO;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;

public static class MeshImporter
{
    public static GameObject ImportCharacter(string jobRoot, string jobName)
    {
        // Look in the job-specific subfolder
        string outputFolder = Path.Combine(jobRoot, "Output");

        // Load JSON files
        string meshPath = Path.Combine(outputFolder, "mesh_data.json");
        string skeletonPath = Path.Combine(outputFolder, "skeleton_data.json");
        string texturePath = Path.Combine(outputFolder, "character_tex.png");

        if (!File.Exists(meshPath) || !File.Exists(skeletonPath))
        {
            Debug.LogError($"Missing mesh or skeleton data in {outputFolder}");
            return null;
        }

        // Parse JSON using JObject
        JObject meshJson = JObject.Parse(File.ReadAllText(meshPath));
        JObject skelJson = JObject.Parse(File.ReadAllText(skeletonPath));

        // Create root GameObject
        GameObject root = new GameObject(jobName + "_Character");

        // Create skeleton
        Dictionary<string, Transform> boneTransforms = new Dictionary<string, Transform>();
        JArray bonesArray = (JArray)skelJson["bones"];
        Transform[] bones = new Transform[bonesArray.Count];

        foreach (JToken boneData in bonesArray)
        {
            GameObject boneObj = new GameObject(boneData["name"].ToString());
            boneObj.transform.SetParent(root.transform);
            
            var localPos = boneData["localPosition"];
            boneObj.transform.localPosition = new Vector3(
                localPos["x"].Value<float>(),
                localPos["y"].Value<float>(),
                localPos["z"].Value<float>()
            );
            
            string boneName = boneData["name"].ToString();
            int boneIndex = boneData["index"].Value<int>();
            
            boneTransforms[boneName] = boneObj.transform;
            bones[boneIndex] = boneObj.transform;
        }

        // Set parent relationships
        foreach (JToken boneData in bonesArray)
        {
            string boneName = boneData["name"].ToString();
            string parentName = boneData["parent"]?.ToString();
            
            if (!string.IsNullOrEmpty(parentName) && boneTransforms.ContainsKey(parentName))
            {
                boneTransforms[boneName].SetParent(boneTransforms[parentName]);
            }
        }

        // Create mesh
        Mesh mesh = new Mesh();
        mesh.name = jobName + "_Mesh";
        
        // Parse vertices
        JArray verticesArray = (JArray)meshJson["vertices"];
        Vector3[] vertices = new Vector3[verticesArray.Count];
        for (int i = 0; i < verticesArray.Count; i++)
        {
            JArray v = (JArray)verticesArray[i];
            vertices[i] = new Vector3(v[0].Value<float>(), v[1].Value<float>(), v[2].Value<float>());
        }
        mesh.vertices = vertices;
        
        // Parse triangles
        JArray trianglesArray = (JArray)meshJson["triangles"];
        int[] triangles = new int[trianglesArray.Count * 3];
        for (int i = 0; i < trianglesArray.Count; i++)
        {
            JArray tri = (JArray)trianglesArray[i];
            triangles[i * 3] = tri[0].Value<int>();
            triangles[i * 3 + 1] = tri[1].Value<int>();
            triangles[i * 3 + 2] = tri[2].Value<int>();
        }
        mesh.triangles = triangles;
        
        // Parse UVs
        JArray uvsArray = (JArray)meshJson["uvs"];
        Vector2[] uvs = new Vector2[uvsArray.Count];
        for (int i = 0; i < uvsArray.Count; i++)
        {
            JArray uv = (JArray)uvsArray[i];
            uvs[i] = new Vector2(uv[0].Value<float>(), uv[1].Value<float>());
        }
        mesh.uv = uvs;

        // Set bone weights
        JArray boneWeightsArray = (JArray)meshJson["boneWeights"];
        BoneWeight[] boneWeights = new BoneWeight[boneWeightsArray.Count];
        for (int i = 0; i < boneWeightsArray.Count; i++)
        {
            JObject bw = (JObject)boneWeightsArray[i];
            boneWeights[i] = new BoneWeight
            {
                boneIndex0 = bw["boneIndex0"].Value<int>(),
                boneIndex1 = bw["boneIndex1"].Value<int>(),
                boneIndex2 = bw["boneIndex2"].Value<int>(),
                boneIndex3 = bw["boneIndex3"].Value<int>(),
                weight0 = bw["weight0"].Value<float>(),
                weight1 = bw["weight1"].Value<float>(),
                weight2 = bw["weight2"].Value<float>(),
                weight3 = bw["weight3"].Value<float>()
            };
        }
        mesh.boneWeights = boneWeights;

        // Set bind poses
        JArray bindPosesArray = (JArray)skelJson["bindPoses"];
        Matrix4x4[] bindPoses = new Matrix4x4[bindPosesArray.Count];
        for (int i = 0; i < bindPosesArray.Count; i++)
        {
            JObject bp = (JObject)bindPosesArray[i];
            Matrix4x4 mat = new Matrix4x4();
            mat.m00 = bp["m00"].Value<float>(); mat.m01 = bp["m01"].Value<float>(); 
            mat.m02 = bp["m02"].Value<float>(); mat.m03 = bp["m03"].Value<float>();
            mat.m10 = bp["m10"].Value<float>(); mat.m11 = bp["m11"].Value<float>(); 
            mat.m12 = bp["m12"].Value<float>(); mat.m13 = bp["m13"].Value<float>();
            mat.m20 = bp["m20"].Value<float>(); mat.m21 = bp["m21"].Value<float>(); 
            mat.m22 = bp["m22"].Value<float>(); mat.m23 = bp["m23"].Value<float>();
            mat.m30 = bp["m30"].Value<float>(); mat.m31 = bp["m31"].Value<float>(); 
            mat.m32 = bp["m32"].Value<float>(); mat.m33 = bp["m33"].Value<float>();
            bindPoses[i] = mat;
        }
        mesh.bindposes = bindPoses;

        mesh.RecalculateNormals();
        mesh.RecalculateBounds();

        // Create mesh renderer
        GameObject meshObj = new GameObject("SkinnedMesh");
        meshObj.transform.SetParent(root.transform);
        
        SkinnedMeshRenderer smr = meshObj.AddComponent<SkinnedMeshRenderer>();
        smr.sharedMesh = mesh;
        smr.bones = bones;
        smr.rootBone = bones[0];
        mesh.RecalculateBounds();
        var bounds = mesh.bounds;
        float maxSize = Mathf.Max(bounds.size.x, bounds.size.y, bounds.size.z);

        if (maxSize > 0.0001f)
        {
            // Target height/size in Unity units
            float targetSize = 2.0f;   // tweak if you want it larger/smaller
            float scaleFactor = targetSize / maxSize;

            // Scale the whole character root (mesh + bones)
            root.transform.localScale = Vector3.one * scaleFactor;

            Debug.Log($"[MeshImporter] Auto-scaled character '{jobName}' by {scaleFactor} (maxSize={maxSize})");
        }
        else
        {
            Debug.LogWarning("[MeshImporter] Mesh bounds size was ~0, skipping auto scale.");
        }


        // Load and apply texture
        if (File.Exists(texturePath))
        {
            // Convert absolute path -> "Assets/..." safely
            string relPath = FileUtil.GetProjectRelativePath(texturePath);

            Debug.Log($"[MeshImporter] Importing texture at: {texturePath}");
            Debug.Log($"[MeshImporter] Unity-relative path: {relPath}");

            AssetDatabase.ImportAsset(relPath, ImportAssetOptions.ForceUpdate);

            Texture2D tex = AssetDatabase.LoadAssetAtPath<Texture2D>(relPath);
            Debug.Log($"[MeshImporter] Loaded texture object: {tex}");

            if (tex != null)
            {
                // URP-friendly shader with fallbacks
                Shader shader = Shader.Find("Universal Render Pipeline/Lit");
                if (shader == null)
                    shader = Shader.Find("Universal Render Pipeline/Unlit");
                if (shader == null)
                    shader = Shader.Find("Sprites/Default");

                Material mat = new Material(shader);

                // URP uses _BaseMap as the albedo property
                if (mat.HasProperty("_BaseMap"))
                    mat.SetTexture("_BaseMap", tex);
                else
                    mat.mainTexture = tex;

                smr.sharedMaterial = mat;

                Debug.Log($"[MeshImporter] Applied shader '{mat.shader.name}' with texture '{relPath}'");
            }
            else
            {
                Debug.LogError($"[MeshImporter] FAILED to LoadAssetAtPath: {relPath}");
            }
        }
        else
        {
            Debug.LogWarning($"[MeshImporter] Texture file does not exist at: {texturePath}");
        }

        Debug.Log($"✅ Successfully imported 3D character: {jobName}");
        return root;
    }
}
#endif