#if UNITY_EDITOR
using UnityEngine;

[CreateAssetMenu(fileName = "GenAI_PipelineAsset", menuName = "GenAI/Netflix Pipeline Asset", order = 0)]
public class GenAIPipelineAsset : ScriptableObject
{
    [Header("Basics")]
    public string imagePath;          // source 2D drawing 
    public string mode = "3d";        // defaulting to 3d bc that's our end goal
    public string outDir;            
    public string outName;

    [Header("Provider")]
    public string provider = "blackforestlabs";   
    [TextArea] public string apiKey;              
    public string providerModel = "";             
    public string providerEndpoint = ""; 

    [Header("Output")]
    public int sizeW = 1024;
    public int sizeH = 1024;
    [Range(1, 8)] public int batch = 1;

    [Header("Semantics")]
    public string objectType = "creature"; // creature|human|vehicle|building|prop|abstract

    [Header("2D Style (ignored when mode=3d)")]
    public string style2D = "photoreal";   // photoreal|cartoon|watercolor|sketch
    public string strictness = "medium";   // low|medium|high

    [Header("3D Style (used when mode=3d)")]
    public string quality3D = "standard";  // draft|standard|high
    public string style3D = "realistic";   // realistic|toy|sculpture

    [Header("Generated / Runtime")]
    public string generatedGlbPath;        // watcher fills when file arrives
    public GameObject generatedPrefab;     // created prefab
}
#endif
