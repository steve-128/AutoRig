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
    private bool isImageSaved = false;
    private bool isRunning = false;
    private string runnerStatus = "";
    private System.Diagnostics.Process currentProcess;
    private System.Text.StringBuilder errorOutput = new System.Text.StringBuilder();
    private System.Text.StringBuilder standardOutput = new System.Text.StringBuilder();
    
    public string qwenRunnerPath = "Assets/Python/qwen_runner.py";

    [MenuItem("GenAI@Berkeley/Netflix Pipeline")]
    public static void Open()
    {
        var win = GetWindow<GenAIPipelineWindow>("GenAI Pipeline");
        win.minSize = new Vector2(500, 400);
        win.Show();
    }

    private void OnGUI()
    {
        EditorGUILayout.Space(6);
        EditorGUILayout.LabelField("<b>Qwen Image to 3D Pipeline</b>", RichCenter());
        EditorGUILayout.Space(8);

        asset = (GenAIPipelineAsset)EditorGUILayout.ObjectField("Pipeline Asset", asset, typeof(GenAIPipelineAsset), false);
        
        if(asset == null)
        {
            if(GUILayout.Button("Create New Pipeline Asset", GUILayout.Height(28)))
                CreateAssetInteractive();
            EditorGUILayout.HelpBox("Create or assign a Pipeline Asset to continue.", MessageType.Info);
            return;
        }

        using (var sv = new EditorGUILayout.ScrollViewScope(scroll))
        {
            scroll = sv.scrollPosition;
            DrawImageInput();
            EditorGUILayout.Space(10);
            DrawSaveSection();
            EditorGUILayout.Space(10);
            DrawRunSection();
            
            if(isRunning)
            {
                EditorGUILayout.Space(10);
                DrawStatusSection();
            }
        }
        
        if(isRunning)
        {
            Repaint();
        }
    }

    private void DrawImageInput()
    {
        Header("1) Select Input Image");
        using(new EditorGUILayout.VerticalScope("box"))
        {
            EditorGUILayout.BeginHorizontal();
            asset.imagePath = EditorGUILayout.TextField("Image Path", asset.imagePath);
            if(GUILayout.Button("Browse", GUILayout.Width(80)))
            {
                var path = EditorUtility.OpenFilePanel("Select Image", "", "png,jpg,jpeg");
                if(!string.IsNullOrEmpty(path))
                {
                    asset.imagePath = path;
                    LoadPreview(path);
                    isImageSaved = false;
                }
            }
            EditorGUILayout.EndHorizontal();

            if(!string.IsNullOrEmpty(asset.imagePath))
            {
                if(preview != null)
                {
                    var rect = GUILayoutUtility.GetAspectRect((float)preview.width / Mathf.Max(1, preview.height), GUILayout.Height(200));
                    EditorGUI.DrawPreviewTexture(rect, preview, null, ScaleMode.ScaleToFit);
                    EditorGUILayout.LabelField($"Size: {preview.width} × {preview.height}");
                }
                else
                {
                    EditorGUILayout.HelpBox("Could not preview image. Ensure file is PNG or JPG.", MessageType.Warning);
                }
            }
            else
            {
                EditorGUILayout.HelpBox("No image selected.", MessageType.Info);
            }
        }
    }

    private void DrawSaveSection()
    {
        Header("2) Save Image & Metadata");
        using(new EditorGUILayout.VerticalScope("box"))
        {
            asset.outName = EditorGUILayout.TextField("Job Name", string.IsNullOrEmpty(asset.outName) ? GuessStem(asset.imagePath) : asset.outName);
            
            EditorGUILayout.Space(4);
            EditorGUILayout.LabelField("This will create:", EditorStyles.boldLabel);
            EditorGUILayout.LabelField($"  • Assets/GenAI/{asset.outName}/Input/");
            EditorGUILayout.LabelField($"  • Copy of your image");
            EditorGUILayout.LabelField($"  • pipeline_metadata.json");
            
            EditorGUILayout.Space(6);
            
            GUI.enabled = !string.IsNullOrEmpty(asset.imagePath) && File.Exists(asset.imagePath);
            if(GUILayout.Button("Save Image & Metadata to Input Folder", GUILayout.Height(32)))
            {
                SaveImageAndMetadata();
            }
            GUI.enabled = true;

            if(isImageSaved)
            {
                EditorGUILayout.HelpBox("✓ Image and metadata saved successfully!", MessageType.Info);
            }
        }
    }

    private void DrawRunSection()
    {
        Header("3) Run Qwen Pipeline");
        using(new EditorGUILayout.VerticalScope("box"))
        {
            EditorGUILayout.LabelField("Qwen Runner Path:", EditorStyles.boldLabel);
            qwenRunnerPath = EditorGUILayout.TextField(qwenRunnerPath);
            
            EditorGUILayout.Space(4);
            EditorGUILayout.LabelField("Hugging Face Token:", EditorStyles.boldLabel);
            asset.apiKey = EditorGUILayout.PasswordField("HF_TOKEN", asset.apiKey ?? "");
            
            if(string.IsNullOrEmpty(asset.apiKey))
            {
                EditorGUILayout.HelpBox(
                    "Hugging Face token required!\n\n" +
                    "Get your token at: https://huggingface.co/settings/tokens\n" +
                    "Create a NEW token with 'read' permissions (not fine-grained).",
                    MessageType.Warning);
            }
            else
            {
                EditorGUILayout.HelpBox(
                    "Token entered. If you get 401 errors:\n" +
                    "• Make sure it's a valid READ token (not fine-grained)\n" +
                    "• Copy it fresh from HuggingFace (no extra spaces)\n" +
                    "• Try creating a brand new token",
                    MessageType.Info);
            }
            
            EditorGUILayout.Space(4);
            
            if(!isImageSaved)
            {
                EditorGUILayout.HelpBox("Please save image and metadata first before running.", MessageType.Warning);
            }

            GUI.enabled = isImageSaved && !string.IsNullOrEmpty(asset.imagePath) && !isRunning && !string.IsNullOrEmpty(asset.apiKey);
            if(GUILayout.Button(isRunning ? "Running..." : "Run Qwen Pipeline", GUILayout.Height(32)))
            {
                RunQwenPipeline();
            }
            GUI.enabled = true;
            
            if(isRunning)
            {
                if(GUILayout.Button("Stop Pipeline", GUILayout.Height(28)))
                {
                    StopPipeline();
                }
            }

            EditorGUILayout.Space(4);
            EditorGUILayout.HelpBox("This will execute qwen_runner.py with the saved image and metadata.", MessageType.Info);
        }
    }

    private void DrawStatusSection()
    {
        Header("Pipeline Status");
        using(new EditorGUILayout.VerticalScope("box"))
        {
            EditorGUILayout.LabelField("Running...", EditorStyles.boldLabel);
            
            if(!string.IsNullOrEmpty(runnerStatus))
            {
                if(runnerStatus.StartsWith("Error:") || runnerStatus.Contains("Error") || runnerStatus.Contains("Traceback"))
                {
                    EditorGUILayout.HelpBox(runnerStatus, MessageType.Error);
                    
                    if(runnerStatus.Contains("ModuleNotFoundError") || runnerStatus.Contains("No module named"))
                    {
                        EditorGUILayout.Space(4);
                        EditorGUILayout.HelpBox(
                            "Missing Python package detected!\n\n" +
                            "Fix: Open Terminal and run:\n" +
                            "/opt/anaconda3/bin/pip install [package_name]\n\n" +
                            "Common packages needed:\n" +
                            "pip install huggingface_hub transformers torch pillow requests",
                            MessageType.Info);
                    }
                    else if(runnerStatus.Contains("401") || runnerStatus.Contains("Unauthorized"))
                    {
                        EditorGUILayout.Space(4);
                        EditorGUILayout.HelpBox(
                            "Authentication failed!\n\n" +
                            "Your HuggingFace token is invalid or expired.\n\n" +
                            "Fix:\n" +
                            "1. Go to https://huggingface.co/settings/tokens\n" +
                            "2. Create a NEW token (type: Read)\n" +
                            "3. Copy it carefully (no spaces)\n" +
                            "4. Paste it in the HF_TOKEN field above\n\n" +
                            "Note: Don't use 'fine-grained' tokens, use regular Read tokens",
                            MessageType.Info);
                    }
                    else if(runnerStatus.Contains("base64") || runnerStatus.Contains("invalid_request_error"))
                    {
                        EditorGUILayout.Space(4);
                        EditorGUILayout.HelpBox(
                            "Image encoding error!\n\n" +
                            "The image file might be corrupted or in wrong format.\n\n" +
                            "Fix:\n" +
                            "1. Make sure your image is a valid PNG or JPG\n" +
                            "2. Try a different image file\n" +
                            "3. Check your qwen_runner.py image encoding logic\n" +
                            "4. The image might be too large - try resizing it",
                            MessageType.Info);
                    }
                }
                else
                {
                    EditorGUILayout.HelpBox(runnerStatus, MessageType.Info);
                }
            }
            
            string dots = new string('.', ((int)(EditorApplication.timeSinceStartup * 2) % 4));
            EditorGUILayout.LabelField($"Processing{dots}", EditorStyles.centeredGreyMiniLabel);
        }
    }

    private void SaveImageAndMetadata()
    {
        if(string.IsNullOrEmpty(asset.imagePath) || !File.Exists(asset.imagePath))
        {
            EditorUtility.DisplayDialog("Error", "Please select a valid image file first.", "OK");
            return;
        }

        string jobName = string.IsNullOrEmpty(asset.outName) ? GuessStem(asset.imagePath) : asset.outName;
        
        CreateFolders(jobName);
        CopyInputImage(jobName);
        SaveMetadata(jobName);
        
        isImageSaved = true;
        EditorUtility.DisplayDialog("Success", $"Image and metadata saved to:\nAssets/GenAI/{jobName}/Input/", "OK");
    }

    private void SpawnOutputImage(string jobRoot, string jobName)
    {
        // look for output file
        string outputFolder = Path.Combine(jobRoot, "Output");
        string[] possibleFiles = new string[]
        {
            Path.Combine(outputFolder, jobName + "_refined.png"),
            Path.Combine(outputFolder, jobName + "_refined"),
            Path.Combine(outputFolder, jobName + ".png"),
            Path.Combine(outputFolder, "output.png"),
            Path.Combine(outputFolder, "refined.png")
        };

        string outputPath = null;
        foreach(var file in possibleFiles)
        {
            if(File.Exists(file))
            {
                outputPath = file;
                break;
            }
        }

        // if we didn't find anything just grab the first png
        if(outputPath == null && Directory.Exists(outputFolder))
        {
            var pngFiles = Directory.GetFiles(outputFolder, "*.png");
            if(pngFiles.Length > 0)
                outputPath = pngFiles[0];
        }

        if(outputPath == null || !File.Exists(outputPath))
        {
            Debug.LogError("couldn't find output image in Output folder");
            return;
        }

        // find input image too
        string inputFolder = Path.Combine(jobRoot, "Input");
        string inputPath = null;
        if(Directory.Exists(inputFolder))
        {
            var inputFiles = Directory.GetFiles(inputFolder, "*.png")
                .Concat(Directory.GetFiles(inputFolder, "*.jpg"))
                .Concat(Directory.GetFiles(inputFolder, "*.jpeg"))
                .ToArray();
            if(inputFiles.Length > 0)
                inputPath = inputFiles[0];
        }

        AssetDatabase.Refresh();

        // convert paths to unity format
        string projectRoot = Application.dataPath.Substring(0, Application.dataPath.Length - "/Assets".Length);
        string outputRelativePath = outputPath.Replace(projectRoot + "/", "").Replace("\\", "/");

        AssetDatabase.ImportAsset(outputRelativePath, ImportAssetOptions.ForceUpdate);
        
        // set npot scale to none so aspect ratio is preserved
        TextureImporter outputImporter = AssetImporter.GetAtPath(outputRelativePath) as TextureImporter;
        if(outputImporter != null)
        {
            outputImporter.npotScale = TextureImporterNPOTScale.None;
            outputImporter.SaveAndReimport();
        }
        
        Texture2D outputTex = AssetDatabase.LoadAssetAtPath<Texture2D>(outputRelativePath);

        if(outputTex == null)
        {
            Debug.LogError($"failed to load texture from {outputRelativePath}");
            return;
        }

        // load input texture
        Texture2D inputTex = null;
        if(inputPath != null)
        {
            string inputRelativePath = inputPath.Replace(projectRoot + "/", "").Replace("\\", "/");
            AssetDatabase.ImportAsset(inputRelativePath, ImportAssetOptions.ForceUpdate);
            
            TextureImporter inputImporter = AssetImporter.GetAtPath(inputRelativePath) as TextureImporter;
            if(inputImporter != null)
            {
                inputImporter.npotScale = TextureImporterNPOTScale.None;
                inputImporter.SaveAndReimport();
            }
            
            inputTex = AssetDatabase.LoadAssetAtPath<Texture2D>(inputRelativePath);
        }

        // clean up old preview quads
        var oldPreviews = FindObjectsOfType<GameObject>()
            .Where(go => go.name.Contains("_RefinedPreview") || go.name.Contains("_Preview") || go.name.Contains("_InputPreview"))
            .ToArray();
        
        foreach(var old in oldPreviews)
            DestroyImmediate(old);

        // spawn main output quad
        GameObject outputQuad = GameObject.CreatePrimitive(PrimitiveType.Quad);
        outputQuad.name = jobName + "_RefinedPreview";

        Material outputMat = new Material(Shader.Find("Unlit/Texture"));
        outputMat.mainTexture = outputTex;
        outputQuad.GetComponent<MeshRenderer>().material = outputMat;

        Camera mainCam = Camera.main;
        if(mainCam != null)
        {
            float distance = 10f;
            float height = 2f * distance * Mathf.Tan(mainCam.fieldOfView * 0.5f * Mathf.Deg2Rad);
            float width = height * mainCam.aspect;

            outputQuad.transform.localScale = new Vector3(width, height, 1f);
            outputQuad.transform.position = mainCam.transform.position + mainCam.transform.forward * distance;
            outputQuad.transform.rotation = mainCam.transform.rotation;

            // spawn input preview in corner
            if(inputTex != null)
            {
                GameObject inputQuad = GameObject.CreatePrimitive(PrimitiveType.Quad);
                inputQuad.name = jobName + "_InputPreview";

                Material inputMat = new Material(Shader.Find("Unlit/Texture"));
                inputMat.mainTexture = inputTex;
                inputQuad.GetComponent<MeshRenderer>().material = inputMat;

                float inputAspect = (float)inputTex.width / inputTex.height;
                float inputWidth = width * 0.2f;
                float inputHeight = inputWidth / inputAspect;
                
                inputQuad.transform.localScale = new Vector3(inputWidth, inputHeight, 1f);
                
                // position in top right
                float xOffset = (width * 0.5f) - (inputWidth * 0.5f) - 0.3f;
                float yOffset = (height * 0.5f) - (inputHeight * 0.5f) - 0.3f;
                
                Vector3 topRightOffset = mainCam.transform.right * xOffset + mainCam.transform.up * yOffset;
                
                inputQuad.transform.position = outputQuad.transform.position + topRightOffset;
                inputQuad.transform.position -= mainCam.transform.forward * 0.1f;
                inputQuad.transform.rotation = mainCam.transform.rotation;
            }
        }
        else
        {
            // no camera found, just place it somewhere
            float height = 10f;
            float aspect = (float)outputTex.width / outputTex.height;
            outputQuad.transform.localScale = new Vector3(aspect * height, height, 1f);
            outputQuad.transform.position = new Vector3(0f, 0f, 10f);
            outputQuad.transform.rotation = Quaternion.Euler(0f, 180f, 0f);
        }
        
        Selection.activeGameObject = outputQuad;
        SceneView.lastActiveSceneView?.FrameSelected();
        
        Debug.Log($"spawned output image preview");
    }

    private void CreateFolders(string jobName)
    {
        EnsureFolder("Assets", "GenAI");
        EnsureFolder("Assets/GenAI", jobName);
        EnsureFolder($"Assets/GenAI/{jobName}", "Input");
        EnsureFolder($"Assets/GenAI/{jobName}", "Output");
        AssetDatabase.Refresh();
    }

    private void CopyInputImage(string jobName)
    {
        string inputFolder = $"Assets/GenAI/{jobName}/Input";
        string fileName = Path.GetFileName(asset.imagePath);
        string destPath = Path.Combine(inputFolder, fileName);
        
        try
        {
            File.Copy(asset.imagePath, destPath, true);
            AssetDatabase.ImportAsset(destPath, ImportAssetOptions.ForceUpdate);
            
            // set texture settings to preserve aspect
            TextureImporter importer = AssetImporter.GetAtPath(destPath) as TextureImporter;
            if(importer != null)
            {
                importer.npotScale = TextureImporterNPOTScale.None;
                importer.SaveAndReimport();
            }
        }
        catch(Exception e)
        {
            Debug.LogError($"couldn't copy image: {e.Message}");
        }
    }

    private void SaveMetadata(string jobName)
    {
        var metadata = new JObject
        {
            ["job_name"] = jobName,
            ["image_path"] = asset.imagePath,
            ["timestamp"] = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss")
        };
        
        string inputFolder = $"Assets/GenAI/{jobName}/Input";
        string jsonPath = Path.Combine(inputFolder, "pipeline_metadata.json");
        
        try
        {
            File.WriteAllText(jsonPath, metadata.ToString(Newtonsoft.Json.Formatting.Indented));
            AssetDatabase.Refresh();
        }
        catch(Exception e)
        {
            Debug.LogError($"couldn't save metadata: {e.Message}");
        }
    }

    private void RunQwenPipeline()
    {
        string jobName = asset.outName;
        string rootPath = Path.GetFullPath($"Assets/GenAI/{jobName}");
        string imagePath = Path.GetFullPath($"Assets/GenAI/{jobName}/Input/{Path.GetFileName(asset.imagePath)}");
        
        if(!File.Exists(qwenRunnerPath))
        {
            EditorUtility.DisplayDialog("Error", $"Qwen runner not found at:\n{qwenRunnerPath}", "OK");
            return;
        }

        if(!File.Exists(imagePath))
        {
            EditorUtility.DisplayDialog("Error", "Image not found in Input folder. Please save first.", "OK");
            return;
        }

        string pythonPath = "/opt/anaconda3/bin/python3";
        
        if(!File.Exists(pythonPath))
        {
            Debug.LogWarning($"python not found at {pythonPath}, trying system python3");
            pythonPath = "python3";
        }
        
        errorOutput.Clear();
        standardOutput.Clear();
        
        string absRoot = Path.GetFullPath(rootPath);
        string absImage = Path.GetFullPath(imagePath);

        var psi = new System.Diagnostics.ProcessStartInfo
        {
            FileName = pythonPath,
            Arguments = $"-u \"{qwenRunnerPath}\" --root \"{absRoot}\" --image \"{absImage}\" --job \"{jobName}\"",
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
            WorkingDirectory = Directory.GetParent(Application.dataPath).FullName
        };

        psi.EnvironmentVariables["PYTHONUNBUFFERED"] = "1";
        
        if(!string.IsNullOrEmpty(asset.apiKey))
        {
            psi.EnvironmentVariables["HF_TOKEN"] = asset.apiKey;
            psi.EnvironmentVariables["GROQ_API_KEY"] = asset.apiKey;
            psi.EnvironmentVariables["FAL_KEY"] = asset.apiKey;
        }

        try
        {
            Debug.Log($"starting qwen pipeline for {jobName}");
            
            currentProcess = System.Diagnostics.Process.Start(psi);
            if(currentProcess == null)
            {
                Debug.LogError("process failed to start");
                return;
            }

            isRunning = true;
            runnerStatus = "Pipeline started...";
            
            currentProcess.OutputDataReceived += (sender, e) => 
            { 
                if(!string.IsNullOrEmpty(e.Data))
                {
                    standardOutput.AppendLine(e.Data);
                    Debug.Log($"[qwen] {e.Data}");
                    runnerStatus = e.Data;
                }
            };
            
            currentProcess.ErrorDataReceived += (sender, e) => 
            { 
                if(!string.IsNullOrEmpty(e.Data))
                {
                    errorOutput.AppendLine(e.Data);
                    Debug.LogError($"[qwen error] {e.Data}");
                    runnerStatus = $"Error: {e.Data}";
                }
            };
            
            currentProcess.EnableRaisingEvents = true;
            currentProcess.Exited += (sender, e) =>
            {
                string rootCopy = absRoot;
                string jobCopy = jobName;

                System.Threading.Thread.Sleep(500);

                int exitCode = currentProcess.ExitCode;

                EditorApplication.delayCall += () =>
                {
                    isRunning = false;

                    if(exitCode == 0)
                    {
                        runnerStatus = "Pipeline completed!";
                        Debug.Log($"qwen pipeline finished for {jobCopy}");
                        SpawnOutputImage(rootCopy, jobCopy);
                    }
                    else
                    {
                        runnerStatus = $"Pipeline failed with code {exitCode}";
                        Debug.LogError($"qwen pipeline failed with code {exitCode}");
                        
                        if(errorOutput.Length > 0)
                        {
                            Debug.LogError("errors:");
                            Debug.LogError(errorOutput.ToString());
                        }
                    }

                    currentProcess = null;
                };
            };

            currentProcess.BeginOutputReadLine();
            currentProcess.BeginErrorReadLine();
        }
        catch(Exception e)
        {
            isRunning = false;
            Debug.LogError($"exception starting pipeline: {e.Message}");
            EditorUtility.DisplayDialog("Error", $"Failed to start pipeline:\n{e.Message}", "OK");
        }
    }

    private void StopPipeline()
    {
        if(currentProcess != null && !currentProcess.HasExited)
        {
            try
            {
                currentProcess.Kill();
                Debug.Log("stopped pipeline");
                runnerStatus = "Stopped by user";
            }
            catch(Exception e)
            {
                Debug.LogError($"couldn't stop pipeline: {e.Message}");
            }
        }
        
        isRunning = false;
        currentProcess = null;
    }

    private void CreateAssetInteractive()
    {
        string path = EditorUtility.SaveFilePanelInProject("Create Pipeline Asset", "QwenPipelineAsset", "asset", "Choose save location");
        if(!string.IsNullOrEmpty(path))
        {
            var newAsset = ScriptableObject.CreateInstance<GenAIPipelineAsset>();
            newAsset.outName = "qwen_job";
            AssetDatabase.CreateAsset(newAsset, path);
            AssetDatabase.SaveAssets();
            Selection.activeObject = newAsset;
            asset = newAsset;
        }
    }

    private void LoadPreview(string path)
    {
        preview = null;
        if(string.IsNullOrEmpty(path) || !File.Exists(path)) return;
        
        try
        {
            byte[] bytes = File.ReadAllBytes(path);
            Texture2D tex = new Texture2D(2, 2, TextureFormat.RGBA32, false);
            if(tex.LoadImage(bytes))
            {
                preview = tex;
            }
        }
        catch(Exception e)
        {
            Debug.LogWarning($"couldn't load preview: {e.Message}");
        }
    }

    private static string GuessStem(string path)
    {
        if(string.IsNullOrEmpty(path)) return "qwen_output";
        try { return Path.GetFileNameWithoutExtension(path); }
        catch { return "qwen_output"; }
    }

    private static void EnsureFolder(string parent, string child)
    {
        if(!AssetDatabase.IsValidFolder($"{parent}/{child}"))
        {
            AssetDatabase.CreateFolder(parent, child);
        }
    }

    private static GUIStyle RichCenter()
    {
        return new GUIStyle(EditorStyles.label)
        {
            alignment = TextAnchor.MiddleCenter,
            richText = true
        };
    }

    private void Header(string text)
    {
        EditorGUILayout.LabelField($"<b>{text}</b>", new GUIStyle(EditorStyles.label) { richText = true });
    }
}
#endif