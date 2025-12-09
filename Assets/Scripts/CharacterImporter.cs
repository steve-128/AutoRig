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

    private static void CleanupAllGeneratedCharacters()
    {
        var all = GameObject.FindObjectsByType<GameObject>(FindObjectsSortMode.None);

        foreach (var go in all)
        {
            if (!go) continue; 

            if (go.name == "Character")
            {
                if (Application.isPlaying)
                    GameObject.Destroy(go);
                else
                    GameObject.DestroyImmediate(go);
            }
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

    private GameObject currentCharacterRoot;
    private GameObject ImportInternal(bool inEditor)
    {
        // CleanupAllGeneratedCharacters();
        if (meshDataFile == null || skeletonDataFile == null || characterTexture == null)
        {
            Debug.LogError("CharacterImporter: meshDataFile, skeletonDataFile, or characterTexture is missing.");
            return null;
        }

        // Remove old character if it exists
        if (currentCharacterRoot != null)
        {
            Debug.Log($"[CharacterImporter] Destroying old character: {currentCharacterRoot.name}");

            if (inEditor || !Application.isPlaying)
                DestroyImmediate(currentCharacterRoot);
            else
                Destroy(currentCharacterRoot);
            currentCharacterRoot = null;
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

            var anim = characterRoot.AddComponent<CharacterAnimator>();
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



public class CharacterAnimator : MonoBehaviour
{
    public enum AnimationMode { BVH, Wave, Squish, Idle }

    [Header("Mode")]
    // FIXED: Default is now Squish
    public AnimationMode currentMode = AnimationMode.Squish;
        
    [Tooltip("Check this if the character is facing the wrong way. Uses Rotation instead of Scale to preserve correct bone sides.")]
    public bool flipFacing = false;

    [Header("BVH Settings")]
    public bool isPlaying = true;
    public bool loop = true;
    public float playbackSpeed = 1.0f;
    public float rotationIntensity = 1.0f;
    public float movementIntensity = 1.0f;
    public int startFrame = 0;
    public int endFrame = -1;

    [Tooltip("When true, root position channels are applied to characterRoot.transform (world/local depending).")]
    public bool applyRootMotion = true;

    [Header("Procedural Settings")]
    public float waveSpeed = 5f;
    public float squishSpeed = 4f; 
    public float squishHeight = 1.5f; 

    private bool loggedBoneNamesOnce = false;

    // INTERNAL
    private Dictionary<string, Transform> boneTransforms = new Dictionary<string, Transform>(StringComparer.OrdinalIgnoreCase);
    [System.NonSerialized] private BVHJoint rootJoint;
    [System.NonSerialized] private List<float[]> frames = new List<float[]>();
    private float frameTime = 0.0333f;
    private float currentTime = 0f;
    private float importScale = 0.025f;
    private Vector3 initialRootLocalPos = Vector3.zero;
    private Quaternion initialRootLocalRot = Quaternion.identity; 
    public Transform characterRoot;

    public bool applyRootRotationCorrection = false;
    public Vector3 rootRotationCorrection = Vector3.zero;

    private Vector3 initialRootScale = Vector3.one;

    private Dictionary<string, string> boneNameFallbackMap = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase) {
        {"Hips", "root"}, {"hip", "root"}, {"spine", "torso"}, {"Head", "neck"},
        {"LeftArm", "left_shoulder"}, {"RightArm", "right_shoulder"},
        {"LeftForeArm", "left_elbow"}, {"RightForeArm", "right_elbow"},
        {"LeftHand", "left_hand"}, {"RightHand", "right_hand"},
        {"LeftUpLeg", "left_hip"}, {"RightUpLeg", "right_hip"},
        {"LeftLeg", "left_knee"}, {"RightLeg", "right_knee"},
        {"LeftFoot", "left_foot"}, {"RightFoot", "right_foot"}
    };
    
    [System.Serializable]
    public class BVHJoint {
        public string name;
        public Vector3 offset;
        public List<string> channels = new List<string>();
        [System.NonSerialized] public int channelStartIndex;
        [System.NonSerialized] public BVHJoint parent;
        [System.NonSerialized] public List<BVHJoint> children = new List<BVHJoint>();
    }

    public void SetBoneTransforms(Dictionary<string, Transform> map)
    {
        boneTransforms = new Dictionary<string, Transform>(map, StringComparer.OrdinalIgnoreCase);
        Transform r = GetBoneTransform("root") ?? (boneTransforms.Count > 0 ? boneTransforms.Values.First() : null);
        if (r != null) {
            initialRootLocalPos = r.localPosition;
            initialRootLocalRot = r.localRotation; // Cache rotation
            initialRootScale = r.localScale;
        }
        else if (characterRoot != null)
        {
            initialRootScale = characterRoot.localScale;
        }
    }
    private void Start()
    {
        if (characterRoot == null && transform != null) characterRoot = transform;
        Transform r = GetBoneTransform("root");
        if (r != null) {
            initialRootLocalPos = r.localPosition;
            initialRootLocalRot = r.localRotation;
            initialRootScale = r.localScale;
        }
    }

    void Awake()
    {
        // If the importer already called SetBoneTransforms, do nothing.
        if (boneTransforms != null && boneTransforms.Count > 0)
            return;

        // Fallback: build bone map from children
        boneTransforms = new Dictionary<string, Transform>(StringComparer.OrdinalIgnoreCase);

        foreach (var t in GetComponentsInChildren<Transform>())
        {
            boneTransforms[t.name] = t;
        }

        // Cache initial root transforms
        Transform r = GetBoneTransform("root") ?? transform;
        if (r != null)
        {
            initialRootLocalPos = r.localPosition;
            initialRootLocalRot = r.localRotation;
            initialRootScale = r.localScale;
        }

        Debug.Log("[CharacterAnimator] Auto-built bone map from hierarchy: "
                + string.Join(", ", boneTransforms.Keys));
    }

    private void Update()
    {
        HandleFacing();

        switch (currentMode) {
            case AnimationMode.BVH:
                UpdateBVH();
                RestoreRootScaleIfNeeded();
                break;
            case AnimationMode.Wave:
                UpdateWave();
                RestoreRootScaleIfNeeded();
                RestoreRootPositionIfNeeded();
                break;
            case AnimationMode.Squish:
                UpdateSquish();
                break;
            case AnimationMode.Idle:
                RestoreRootScaleIfNeeded();
                RestoreRootPositionIfNeeded();
                break;
        }
    }

    void HandleFacing()
    {
        Transform root = GetBoneTransform("root") ?? characterRoot;
        if (root == null) return;

        float targetY = flipFacing ? 180f : 0f;
            
        if (currentMode != AnimationMode.BVH)
        {
            Quaternion baseRot = (currentMode == AnimationMode.Idle || currentMode == AnimationMode.Squish) 
                ? initialRootLocalRot 
                : root.localRotation;

            Quaternion flipRot = Quaternion.Euler(0, targetY, 0);
            
            if(currentMode == AnimationMode.Squish || currentMode == AnimationMode.Idle)
            {
                root.localRotation = initialRootLocalRot * flipRot;
            }
        }
    }

    void RestoreRootScaleIfNeeded()
    {
        Transform root = GetBoneTransform("root") ?? characterRoot;
        if (root != null && root.localScale != initialRootScale) 
            root.localScale = initialRootScale;
    }

    void RestoreRootPositionIfNeeded()
    {
         Transform root = GetBoneTransform("root");
         if (root != null && root.localPosition != initialRootLocalPos)
            root.localPosition = initialRootLocalPos;
    }

        // ------------------ PRESET ANIMATIONS ------------------
    void UpdateWave()
    {
        Transform shoulder = GetBoneTransform("left_shoulder");
        Transform elbow    = GetBoneTransform("left_elbow");

        if (shoulder == null || elbow == null)
        {
            if (!loggedBoneNamesOnce)
            {
                loggedBoneNamesOnce = true;
                Debug.LogWarning(
                    "[Wave] Could not find left_shoulder or left_elbow. " +
                    "Available bones: " + string.Join(", ", boneTransforms.Keys)
                );
            }
            return;
        }

        float wave = Mathf.Sin(Time.time * waveSpeed);

        float baseX = 0f;      // adjust if you want a default bend on X
        float baseZ = 30f;    // original static elbow bend

        float ampX = 10f;      // how much to swing on X
        float ampZ = 10f;      // how much to swing on Z

        float elbowX = baseX + wave * ampX;
        float elbowZ = baseZ + wave * ampZ;

        elbow.localRotation = Quaternion.Euler(elbowX, 0f, elbowZ);
    }
        // ------------------ SQUISH (Formerly Jump) ------------------
    void UpdateSquish()
    {
        Transform root = GetBoneTransform("root") ?? characterRoot;
        if (root == null) return;

        float cycle = Time.time * squishSpeed;
        float rawSine = Mathf.Sin(cycle); 

        float heightOffset = 0f;
        if (rawSine > 0)
        {
            heightOffset = rawSine * squishHeight; 
        }

        root.localPosition = initialRootLocalPos + new Vector3(0f, heightOffset, 0f);
        
        float squashStretch = 1.0f;
        
        if (rawSine > 0) 
        {
            squashStretch = 1.0f + (rawSine * 0.2f); 
        }
        else 
        {
            float t = Mathf.Abs(rawSine);
            squashStretch = 1.0f - (t * 0.3f); 
        }

        float widthScale = 1.0f / Mathf.Sqrt(squashStretch);
        
        Vector3 targetScale = new Vector3(
            initialRootScale.x * widthScale,
            initialRootScale.y * squashStretch,
            initialRootScale.z * widthScale
        );

        root.localScale = targetScale;
        
        if (flipFacing)
        {
             root.localRotation = initialRootLocalRot * Quaternion.Euler(0, 180, 0);
        }
        else
        {
             root.localRotation = initialRootLocalRot;
        }
    }

    // ------------------ BVH ------------------
    public void ParseBVH(string bvhContent, float scale = 0.025f)
    {
        this.importScale = scale;
        frames.Clear();
        rootJoint = null;

        string[] lines = bvhContent.Replace("\r\n", "\n").Replace('\r','\n').Split('\n');
        int li = 0;

        while (li < lines.Length && !lines[li].Trim().StartsWith("HIERARCHY", StringComparison.OrdinalIgnoreCase)) li++;
        if (li < lines.Length) li++;

        int totalChannels = 0;
        rootJoint = ParseJoint(lines, ref li, null, ref totalChannels, scale);

        while (li < lines.Length && !lines[li].Trim().StartsWith("MOTION", StringComparison.OrdinalIgnoreCase)) li++;
        if (li >= lines.Length) {
            Debug.LogWarning("BVH: MOTION section not found.");
            return;
        }
        li++;

        while (li < lines.Length && string.IsNullOrWhiteSpace(lines[li])) li++;
        if (li < lines.Length && lines[li].Trim().StartsWith("Frames", StringComparison.OrdinalIgnoreCase)) {
            var parts = lines[li].Split(':');
            if (parts.Length > 1) {
                if (!int.TryParse(parts[1].Trim(), out int fc)) fc = 0;
                else { li++; }
            } else { li++; }
        }

        while (li < lines.Length && !lines[li].Trim().ToLower().Contains("frame time")) li++;
        if (li < lines.Length && lines[li].ToLower().Contains("frame time")) {
            var parts = lines[li].Split(':');
            if (parts.Length > 1) {
                if (!float.TryParse(parts[1].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out frameTime))
                    frameTime = 0.0333f;
            }
            li++;
        }

        for (; li < lines.Length; li++)
        {
            string ln = lines[li].Trim();
            if (string.IsNullOrEmpty(ln)) continue;
            var vals = ln.Split(new char[]{' ','\t'}, StringSplitOptions.RemoveEmptyEntries);
            float[] data = new float[vals.Length];
            for (int k = 0; k < vals.Length; k++)
            {
                if (!float.TryParse(vals[k], NumberStyles.Float, CultureInfo.InvariantCulture, out data[k]))
                    data[k] = 0f;
            }
            frames.Add(data);
        }

        if (endFrame < 0) endFrame = frames.Count - 1;
        currentMode = AnimationMode.BVH;
        Debug.Log($"BVH parsed: frames={frames.Count}, frameTime={frameTime}, rootJoint={rootJoint?.name}");
    }

    private BVHJoint ParseJoint(string[] lines, ref int li, BVHJoint parent, ref int totalChannels, float scale)
    {
        if (li >= lines.Length) return null;
        string line = lines[li].Trim();
        string[] tokens = line.Split(new char[]{' ','\t'}, StringSplitOptions.RemoveEmptyEntries);
        string jtype = tokens.Length > 0 ? tokens[0] : "";
        string jname = tokens.Length > 1 ? tokens[1] : "EndSite";

        BVHJoint joint = new BVHJoint {
            name = jname,
            parent = parent,
            channelStartIndex = totalChannels
        };

        li++;
        while (li < lines.Length)
        {
            line = lines[li].Trim();
            if (line.StartsWith("}")) { li++; break; }
            else if (line.StartsWith("OFFSET", StringComparison.OrdinalIgnoreCase)) {
                var parts = line.Split(new char[]{' ','\t'}, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length >= 4) {
                    if (float.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out float ox) &&
                        float.TryParse(parts[2], NumberStyles.Float, CultureInfo.InvariantCulture, out float oy) &&
                        float.TryParse(parts[3], NumberStyles.Float, CultureInfo.InvariantCulture, out float oz)) {
                        // FIXED: Removed negation of Y (was -oy*scale)
                        joint.offset = new Vector3(ox*scale, oy*scale, oz*scale);
                    }
                }
                li++;
            }
            else if (line.StartsWith("CHANNELS", StringComparison.OrdinalIgnoreCase)) {
                var parts = line.Split(new char[]{' ','\t'}, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length >= 2) {
                    if (int.TryParse(parts[1], out int count)) {
                        for (int c = 0; c < count; c++) {
                            if (2 + c < parts.Length) joint.channels.Add(parts[2 + c]);
                            else joint.channels.Add("Xrotation");
                        }
                        totalChannels += count;
                    }
                }
                li++;
            }
            else if (line.StartsWith("JOINT", StringComparison.OrdinalIgnoreCase) || line.StartsWith("End", StringComparison.OrdinalIgnoreCase) || line.StartsWith("ROOT", StringComparison.OrdinalIgnoreCase)) {
                BVHJoint child = ParseJoint(lines, ref li, joint, ref totalChannels, scale);
                if (child != null) joint.children.Add(child);
            }
            else {
                li++;
            }
        }
        return joint;
    }

    void ApplyJointAnimation(BVHJoint joint, float[] data)
    {
        if (joint == null || data == null) return;
        Transform t = GetBoneTransform(joint.name);

        bool hasPos = false;
        Vector3 pos = Vector3.zero;
        Quaternion rot = Quaternion.identity;

        for (int i = 0; i < joint.channels.Count; i++)
        {
            int idx = joint.channelStartIndex + i;
            if (idx < 0 || idx >= data.Length) continue;
            float val = data[idx];

            string ch = joint.channels[i];
            if (ch.Equals("Xposition", StringComparison.OrdinalIgnoreCase)) { pos.x = val; hasPos = true; }
            // FIXED: Removed negation of Y (was pos.y = -val)
            else if (ch.Equals("Yposition", StringComparison.OrdinalIgnoreCase)) { pos.y = val; hasPos = true; }
            else if (ch.Equals("Zposition", StringComparison.OrdinalIgnoreCase)) { pos.z = val; hasPos = true; }
            else if (ch.Equals("Xrotation", StringComparison.OrdinalIgnoreCase)) rot = rot * Quaternion.Euler(val * rotationIntensity, 0f, 0f);
            else if (ch.Equals("Yrotation", StringComparison.OrdinalIgnoreCase)) rot = rot * Quaternion.Euler(0f, val * rotationIntensity, 0f);
            else if (ch.Equals("Zrotation", StringComparison.OrdinalIgnoreCase)) rot = rot * Quaternion.Euler(0f, 0f, val * rotationIntensity);
        }

        if (joint.parent == null && applyRootRotationCorrection)
        {
            rot = Quaternion.Euler(rootRotationCorrection) * rot;
        }

        if (hasPos)
        {
            Vector3 localPos = pos * importScale * movementIntensity;
            if (joint.parent == null && applyRootMotion && characterRoot != null)
            {
                characterRoot.localPosition = localPos;
            }
            else if (t != null)
            {
                t.localPosition = localPos;
            }
        }

        if (t != null)
        {
            if (flipFacing && joint.parent == null) {
                t.localRotation = rot * Quaternion.Euler(0, 180, 0);
            } else {
                t.localRotation = rot;
            }
        }

        foreach (var c in joint.children) ApplyJointAnimation(c, data);
    }

    void UpdateBVH()
    {
        if (!isPlaying || frames.Count == 0 || rootJoint == null) return;

        currentTime += Time.deltaTime * playbackSpeed;
        float totalDuration = (endFrame - startFrame + 1) * frameTime;
        if (totalDuration <= 0) totalDuration = frames.Count * frameTime;

        if (currentTime >= totalDuration) {
            if (loop) currentTime %= totalDuration;
            else currentTime = totalDuration;
        }

        // FIXED: Use local variable instead of class field
        int currentFrameIndex = startFrame + Mathf.FloorToInt(currentTime / frameTime);
        currentFrameIndex = Mathf.Clamp(currentFrameIndex, 0, frames.Count - 1);
        ApplyJointAnimation(rootJoint, frames[currentFrameIndex]);
    }

    // ------------------ BONE FINDING & FUZZY MATCH ------------------
    private Transform GetBoneTransform(string name)
    {
        if (string.IsNullOrEmpty(name)) return null;

        if (boneTransforms != null && boneTransforms.TryGetValue(name, out Transform t)) return t;

        if (boneNameFallbackMap.ContainsKey(name) && boneTransforms.TryGetValue(boneNameFallbackMap[name], out Transform t2))
            return t2;

        string alt = name.Replace(" ", "").Replace("-", "").Replace("_", "").ToLowerInvariant();
        foreach (var kv in boneTransforms)
        {
            string keyAlt = kv.Key.Replace(" ", "").Replace("-", "").Replace("_", "").ToLowerInvariant();
            if (keyAlt == alt) return kv.Value;
        }

        if (name.StartsWith("Left", StringComparison.OrdinalIgnoreCase)) {
            string swapped = "Right" + name.Substring(4);
            if (boneTransforms.TryGetValue(swapped, out Transform s)) return s;
        } else if (name.StartsWith("Right", StringComparison.OrdinalIgnoreCase)) {
            string swapped = "Left" + name.Substring(5);
            if (boneTransforms.TryGetValue(swapped, out Transform s2)) return s2;
        }

        foreach (var kv in boneTransforms)
            if (kv.Key.IndexOf(name, StringComparison.OrdinalIgnoreCase) >= 0) return kv.Value;

        return null;
    }

    private Transform TryFindAny(string[] names, string[] fallbackGroup)
    {
        foreach (var n in names) {
            var t = GetBoneTransform(n);
            if (t != null) return t;
        }
        foreach (var n in fallbackGroup) {
            var t = GetBoneTransform(n);
            if (t != null) return t;
        }
        return null;
    }
}