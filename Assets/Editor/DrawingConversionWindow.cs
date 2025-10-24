#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;
using System;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;

public class GenAIPipelineWindow : EditorWindow
{
    private GenAIPipelineAsset asset;
    private Texture2D preview;
    private Vector2 scroll;
    private double nextPoll;
    private const double PollInterval = 1.0;

     // provider options
    private readonly string[] providerLabels = new[] {
        "BlackForestLabs (2D→3D)", "Qwen (2D→3D)"
    };
    private readonly string[] providerValues = new[] {
        "blackforestlabs", "qwen"
    };
    private int providerIdx = 0;


    [MenuItem("GenAI@Berkeley/Netflix Pipeline")]
    public static void Open()
    {
        var win = GetWindow<GenAIPipelineWindow>("GenAI Pipeline");
        win.minSize = new Vector2(540, 620);
        win.Show();
    }

    private void OnGUI()
    {
        EditorGUILayout.Space(6);
        EditorGUILayout.LabelField("<b>2D → 3D → Animate (handoff shell)</b>", richCenter());
        EditorGUILayout.Space(8);

        asset = (GenAIPipelineAsset)EditorGUILayout.ObjectField("Pipeline Asset", asset, typeof(GenAIPipelineAsset), false);
        if(asset == null)
        {
            if(GUILayout.Button("Create New Pipeline Asset…", GUILayout.Height(28)))
                CreateAssetInteractive();
            EditorGUILayout.HelpBox("Create or assign a Pipeline Asset to continue.", MessageType.Info);
            return;
        }

        scroll = EditorGUILayout.BeginScrollView(scroll);

        DrawBasics();
        EditorGUILayout.Space();
        DrawProvider();
        EditorGUILayout.Space();
        DrawOutput();
        EditorGUILayout.Space();
        DrawSemantics();
        EditorGUILayout.Space();
        DrawFoldersAndSnapshot();

        EditorGUILayout.EndScrollView();

        if(EditorApplication.timeSinceStartup >= nextPoll)
        {
            nextPoll = EditorApplication.timeSinceStartup + PollInterval;
            TryAutoIngestGlb();
        }
    }

    private void DrawBasics()
    {
        Header("1) Input 2D Drawing");
        using(new EditorGUILayout.VerticalScope("box"))
        {
            EditorGUILayout.BeginHorizontal();
            asset.imagePath = EditorGUILayout.TextField("Image Path (PNG/JPG)", asset.imagePath);
            if(GUILayout.Button("Browse", GUILayout.Width(80)))
            {
                var p = EditorUtility.OpenFilePanel("Select drawing", "", "png,jpg,jpeg");
                if(!string.IsNullOrEmpty(p)) { asset.imagePath = p; LoadPreview(p); }
            }
            EditorGUILayout.EndHorizontal();

            if(!string.IsNullOrEmpty(asset.imagePath))
            {
                if(preview != null)
                {
                    var r = GUILayoutUtility.GetAspectRect((float)preview.width / Mathf.Max(1, preview.height), GUILayout.Height(180));
                    EditorGUI.DrawPreviewTexture(r, preview, null, ScaleMode.ScaleToFit);
                    EditorGUILayout.LabelField($"Loaded {preview.width}×{preview.height}");
                }
                else
                {
                    EditorGUILayout.HelpBox("Could not preview image (ensure PNG/JPG).", MessageType.Warning);
                }
            }

            // defaulting to 3d
            asset.mode = GUILayout.Toggle(asset.mode == "3d", "3D (target)") ? "3d" : "2d";
        }
    }

    private void DrawProvider()
    {
        Header("2) Provider (model used for 2D→3D)");
        using(new EditorGUILayout.VerticalScope("box"))
        {
            int idx = Array.IndexOf(providerValues, asset.provider);
            if(idx < 0) idx = 0;
            idx = GUILayout.SelectionGrid(idx, providerLabels, providerLabels.Length);
            providerIdx = Mathf.Clamp(idx, 0, providerValues.Length - 1);
            asset.provider = providerValues[providerIdx];

            // keys
            if(asset.provider == "blackforestlabs" || asset.provider == "qwen")
            {
                asset.apiKey = EditorGUILayout.PasswordField("API Key (not saved to JSON)", asset.apiKey ?? "");
                asset.providerModel = EditorGUILayout.TextField("Model ID (e.g., flux-3d / qwen-3d)", asset.providerModel ?? "");
                asset.providerEndpoint = EditorGUILayout.TextField("Endpoint (optional URL/route)", asset.providerEndpoint ?? "");
            }

            if(asset.mode == "3d")
                EditorGUILayout.HelpBox("This provider will be recorded in the snapshot so backend can route to the right 2D→3D model.", MessageType.Info);
            else
                EditorGUILayout.HelpBox("You selected 2D mode, but these providers are primarily for 2D→3D.", MessageType.Warning);
        }
    }


    private void DrawOutput()
    {
        Header("3) Output Size & Batch");
        using(new EditorGUILayout.VerticalScope("box"))
        {
            EditorGUILayout.BeginHorizontal();
            if(GUILayout.Button("512²")) { asset.sizeW = asset.sizeH = 512; }
            if(GUILayout.Button("768²")) { asset.sizeW = asset.sizeH = 768; }
            if(GUILayout.Button("1024²")) { asset.sizeW = asset.sizeH = 1024; }
            if(GUILayout.Button("1536²")) { asset.sizeW = asset.sizeH = 1536; }
            EditorGUILayout.EndHorizontal();

            asset.sizeW = EditorGUILayout.IntField("Width", asset.sizeW);
            asset.sizeH = EditorGUILayout.IntField("Height", asset.sizeH);
            asset.batch = EditorGUILayout.IntSlider("Batch", asset.batch, 1, 8);
        }
    }

    private void DrawSemantics()
    {
        Header("4) Object Type & Style Hints");
        using(new EditorGUILayout.VerticalScope("box"))
        {
            string[] objs = { "creature","human","vehicle","building","prop","abstract" };
            int oi = Array.IndexOf(objs, asset.objectType);
            if(oi < 0) oi = 0;
            oi = GUILayout.SelectionGrid(oi, new[] { "Creature","Human","Vehicle","Building","Prop","Abstract" }, 3);
            asset.objectType = objs[Mathf.Clamp(oi, 0, objs.Length-1)];

            if(asset.mode == "2d")
            {
                string[] s2 = { "photoreal","cartoon","watercolor","sketch" };
                int si = Array.IndexOf(s2, asset.style2D); if(si < 0) si = 0;
                si = GUILayout.SelectionGrid(si, new[] { "Photoreal","Cartoon","Watercolor","Sketch" }, 4);
                asset.style2D = s2[si];

                string[] strict = { "low","medium","high" };
                int sti = Array.IndexOf(strict, asset.strictness); if(sti < 0) sti = 1;
                sti = GUILayout.SelectionGrid(sti, new[] { "Low","Medium","High" }, 3);
                asset.strictness = strict[sti];
            }
            else
            {
                string[] q = { "draft","standard","high" };
                int qi = Array.IndexOf(q, asset.quality3D); if(qi < 0) qi = 1;
                qi = GUILayout.SelectionGrid(qi, new[] { "Draft","Standard","High" }, 3);
                asset.quality3D = q[qi];

                string[] s3 = { "realistic","toy","sculpture" };
                int s3i = Array.IndexOf(s3, asset.style3D); if(s3i < 0) s3i = 0;
                s3i = GUILayout.SelectionGrid(s3i, new[] { "Realistic","Toy/Cartoon","Sculpture" }, 3);
                asset.style3D = s3[s3i];
            }
        }
    }

    private void DrawFoldersAndSnapshot()
    {
        Header("5) Standardize Folders & Save Snapshot");
        using(new EditorGUILayout.VerticalScope("box"))
        {
            string defaultRoot = $"Assets/GenAI";
            string nameStem = string.IsNullOrEmpty(asset.outName)
                ? GuessStem(asset.imagePath) + "_" + asset.mode
                : asset.outName;

            EditorGUILayout.LabelField("We'll create/use:", EditorStyles.boldLabel);
            EditorGUILayout.LabelField($"• {defaultRoot}/{nameStem}/Input/");
            EditorGUILayout.LabelField($"• {defaultRoot}/{nameStem}/Working/");
            EditorGUILayout.LabelField($"• {defaultRoot}/{nameStem}/Output/");

            EditorGUILayout.Space(4);
            asset.outName = EditorGUILayout.TextField("Out Name", nameStem);

            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.PrefixLabel("Out Dir (absolute or relative)");
            asset.outDir = EditorGUILayout.TextField(string.IsNullOrEmpty(asset.outDir) ? $"{defaultRoot}/{nameStem}/Output" : asset.outDir);
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.Space(6);
            if(GUILayout.Button("Create Folders + Save .last_session_min.json", GUILayout.Height(30)))
            {
                CreateFolders(defaultRoot, nameStem);
                SaveSnapshotJson();
                EditorUtility.DisplayDialog("Saved", "Snapshot written to project root.\nFolders ensured in Assets/GenAI/…", "OK");
            }

            EditorGUILayout.Space(4);
            if(GUILayout.Button("Open Output Folder", GUILayout.Height(24)))
            {
                var abs = ToAbsolute(asset.outDir);
                if(!Directory.Exists(abs)) Directory.CreateDirectory(abs);
                EditorUtility.RevealInFinder(abs);
            }

            EditorGUILayout.Space(8);
            EditorGUILayout.LabelField("GLB Watcher", EditorStyles.boldLabel);

            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.PrefixLabel("Detected GLB");
            EditorGUILayout.SelectableLabel(asset.generatedGlbPath ?? "(none)", GUILayout.Height(16));
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.ObjectField("Generated Prefab", asset.generatedPrefab, typeof(GameObject), false);
        }
    }


    private void CreateAssetInteractive()
    {
        string path = EditorUtility.SaveFilePanelInProject("Create Pipeline Asset", "GenAI_PipelineAsset", "asset", "Pick a save location");
        if(!string.IsNullOrEmpty(path))
        {
            var a = ScriptableObject.CreateInstance<GenAIPipelineAsset>();
            a.mode = "3d";
            a.provider = "meshy";
            a.sizeW = a.sizeH = 1024;
            AssetDatabase.CreateAsset(a, path);
            AssetDatabase.SaveAssets();
            Selection.activeObject = a;
            asset = a;
        }
    }

    private void CreateFolders(string root, string stem)
    {
        EnsureFolder("Assets", "GenAI");
        EnsureFolder("Assets/GenAI", stem);
        EnsureFolder($"Assets/GenAI/{stem}", "Input");
        EnsureFolder($"Assets/GenAI/{stem}", "Working");
        EnsureFolder($"Assets/GenAI/{stem}", "Output");
        AssetDatabase.Refresh();
    }

    private void SaveSnapshotJson()
    {
        // build json snapshot
        var root = new JObject
        {
            ["basics"] = new JObject
            {
                ["image_path"] = asset.imagePath,
                ["mode"] = asset.mode,
                ["out_dir"] = asset.outDir,
                ["out_name"] = asset.outName
            },
            ["provider"] = new JObject
            {
                ["name"] = asset.provider,
                ["model"] = string.IsNullOrEmpty(asset.providerModel) ? null : asset.providerModel,
                ["endpoint"] = string.IsNullOrEmpty(asset.providerEndpoint) ? null : asset.providerEndpoint,
                ["api_key"] = null
            },
            ["output"] = new JObject
            {
                ["size"] = new JArray(asset.sizeW, asset.sizeH),
                ["batch"] = asset.batch
            },
            ["object_type"] = asset.objectType,
            ["style2d"] = (asset.mode == "2d") ? new JObject
            {
                ["style"] = asset.style2D,
                ["strictness"] = asset.strictness
            } : null,
            ["style3d"] = (asset.mode == "3d") ? new JObject
            {
                ["quality"] = asset.quality3D,
                ["style"] = asset.style3D
            } : null
        };
        var json = root.ToString(Newtonsoft.Json.Formatting.Indented);
        var outPath = Path.Combine(Directory.GetCurrentDirectory(), ".last_session_min.json");
        File.WriteAllText(outPath, json);
        SpawnImageInScene(asset.imagePath);
        AssetDatabase.Refresh();
    }

    private void TryAutoIngestGlb()
    {
        if(asset == null) return;
        var outAbs = ToAbsolute(asset.outDir);
        if(string.IsNullOrEmpty(outAbs) || !Directory.Exists(outAbs)) return;

        string targetName = string.IsNullOrEmpty(asset.outName) ? "output" : asset.outName;
        string[] glbs = Directory.GetFiles(outAbs, "*.glb", SearchOption.TopDirectoryOnly);
        string picked = null;
        foreach(var g in glbs)
        {
            if(Path.GetFileNameWithoutExtension(g).StartsWith(targetName, StringComparison.OrdinalIgnoreCase))
            {
                picked = g; break;
            }
        }
        if(picked == null && glbs.Length > 0) picked = glbs[0];
        if(picked == null) return;

        if(asset.generatedGlbPath == picked) return; 
        asset.generatedGlbPath = picked;
        EditorUtility.SetDirty(asset);

        // copy glb into project
        string stem = string.IsNullOrEmpty(asset.outName) ? GuessStem(asset.imagePath) + "_" + asset.mode : asset.outName;
        string destFolder = $"Assets/GenAI/{stem}/Output";
        EnsureFolder($"Assets/GenAI", stem);
        EnsureFolder($"Assets/GenAI/{stem}", "Output");

        string destAssetPath = $"{destFolder}/{Path.GetFileName(picked)}";
        File.Copy(picked, destAssetPath, true);
        AssetDatabase.ImportAsset(destAssetPath, ImportAssetOptions.ForceUpdate);

        GameObject go = new GameObject(stem);
        var glbObj = AssetDatabase.LoadAssetAtPath<GameObject>(destAssetPath);
        if(glbObj != null)
        {
            var inst = (GameObject)PrefabUtility.InstantiatePrefab(glbObj);
            inst.transform.SetParent(go.transform, false);
        }
        go.AddComponent<Animator>();

    }
    
    private void LoadPreview(string p)
    {
        preview = null;
        if(string.IsNullOrEmpty(p) || !File.Exists(p)) return;
        try
        {
            var bytes = File.ReadAllBytes(p);
            var t = new Texture2D(2, 2, TextureFormat.RGBA32, false);
            if(t.LoadImage(bytes)) preview = t;
        }
        catch { }
    }

    private static string GuessStem(string path)
    {
        if(string.IsNullOrEmpty(path)) return "output";
        try { return Path.GetFileNameWithoutExtension(path); } catch { return "output"; }
    }

    private static void EnsureFolder(string parent, string child)
    {
        if(!AssetDatabase.IsValidFolder($"{parent}/{child}"))
            AssetDatabase.CreateFolder(parent, child);
    }

    private static string ToAbsolute(string maybeRel)
    {
        if(string.IsNullOrEmpty(maybeRel)) return null;
        if(Path.IsPathRooted(maybeRel)) return maybeRel;
        return Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory(), maybeRel));
    }

    private static GUIStyle richCenter()
    {
        var s = new GUIStyle(EditorStyles.label) { alignment = TextAnchor.MiddleCenter, richText = true };
        return s;
    }

    private void Header(string text)
    {
        EditorGUILayout.LabelField($"<b>{text}</b>", new GUIStyle(EditorStyles.label) { richText = true });
    }
    
    private void SpawnImageInScene(string imagePath)
    {
        // delete old preview objects
        var oldPreviews = UnityEngine.Object.FindObjectsByType<GameObject>(FindObjectsSortMode.None)
            .Where(go =>
                go.name.EndsWith("_Preview") ||
                go.name.Contains("_Preview(Clone)") ||
                go.name.Contains("_Preview "))
            .ToList();

        foreach(var old in oldPreviews)
        {
            if(old != null)
            {
                string oldName = old.name;
                UnityEngine.Object.DestroyImmediate(old);
                Debug.Log($"deleted old preview: {oldName}");
            }
        }

        // load the image
        byte[] fileData = File.ReadAllBytes(imagePath);
        Texture2D tex = new Texture2D(2, 2);
        if(!tex.LoadImage(fileData))
        {
            Debug.LogError("couldn't load image: " + imagePath);
            return;
        }

        // make a quad plane
        GameObject plane = GameObject.CreatePrimitive(PrimitiveType.Quad);
        plane.name = Path.GetFileNameWithoutExtension(imagePath) + "_Preview";

        Material mat = new Material(Shader.Find("Unlit/Texture"));
        mat.mainTexture = tex;
        plane.GetComponent<MeshRenderer>().material = mat;

        // scale based on aspect ratio
        float aspect = (float)tex.width / tex.height;
        float baseHeight = 3.5f;
        plane.transform.localScale = new Vector3(aspect * baseHeight, baseHeight, 1);

        // try to find the 3d model
        GameObject model = null;
        var allObjects = UnityEngine.Object.FindObjectsByType<GameObject>(FindObjectsSortMode.None);
        foreach(var obj in allObjects)
        {
            if(obj.name.Contains("output") || obj.name.Contains("Model") || obj.name.Contains("Generated"))
            {
                model = obj;
                break;
            }
        }

        // position the image
        if(model != null)
        {
            Vector3 modelPos = model.transform.position;
            plane.transform.position = modelPos + new Vector3(-3f, 1f, 0f);  
            Debug.Log($"spawned image beside model {model.name}");
        }
        else
        {
            plane.transform.position = new Vector3(7.2f, 4f, 0f); 
            Debug.Log("no model found, spawned image at default spot");
        }

        Selection.activeGameObject = plane;
    }


}
#endif