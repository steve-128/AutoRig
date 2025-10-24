using UnityEngine;
using UnityEditor;
using System.IO;
using System.Diagnostics;
using Debug = UnityEngine.Debug;
using System.Linq;

public class AutomateVideoToSprite : AssetPostprocessor
{
    static void OnPostprocessAllAssets(string[] importedAssets, string[] deletedAssets,
                                       string[] movedAssets, string[] movedFromAssetPaths)
    {
        foreach (string assetPath in importedAssets)
        {
            if (assetPath.EndsWith(".mp4"))
            {
                ConvertVideoToSprites(assetPath);
            }
        }
    }

    static void ConvertVideoToSprites(string videoPath)
    {
        Debug.Log("Converting " + videoPath);
        string fullPath = Path.GetFullPath(videoPath);
        string projectPath = Application.dataPath;
        string folder = Path.GetDirectoryName(fullPath);
        string fileName = Path.GetFileNameWithoutExtension(videoPath);
        string framesDir = Path.Combine(folder, fileName + "_frames");

        if (!Directory.Exists(framesDir))
            Directory.CreateDirectory(framesDir);

        string ffmpegPath = "/opt/homebrew/bin/ffmpeg";
        string ffmpegArgs = $"-i \"{fullPath}\" -vf fps=15 \"{framesDir}/frame_%04d.png\"";
        RunFFmpeg(ffmpegPath, ffmpegArgs);

        AssetDatabase.Refresh();

        string[] framePaths = Directory.GetFiles(framesDir, "*.png").OrderBy(f => f).ToArray();
        if (framePaths.Length == 0)
        {
            Debug.LogError($"No frames found for {videoPath}");
            return;
        }

        foreach (var frame in framePaths)
        {
            string relPath = "Assets" + frame.Replace(projectPath, "").Replace("\\", "/");
            TextureImporter importer = (TextureImporter)TextureImporter.GetAtPath(relPath);
            if (importer != null)
            {
                importer.textureType = TextureImporterType.Sprite;
                importer.spriteImportMode = SpriteImportMode.Single;
                importer.SaveAndReimport();
            }
        }

        AssetDatabase.Refresh();

        var sprites = framePaths.Select(p =>
            (Sprite)AssetDatabase.LoadAssetAtPath(
                "Assets" + p.Replace(projectPath, "").Replace("\\", "/"), typeof(Sprite))
        ).Where(s => s != null).ToArray();

        if (sprites.Length == 0)
        {
            Debug.LogError($"Failed to load sprites for {fileName}");
            return;
        }

        AnimationClip clip = new AnimationClip();
        clip.frameRate = 30f;

        EditorCurveBinding binding = new EditorCurveBinding
        {
            type = typeof(SpriteRenderer),
            path = "",
            propertyName = "m_Sprite"
        };

        ObjectReferenceKeyframe[] keyframes = new ObjectReferenceKeyframe[sprites.Length];
        for (int i = 0; i < sprites.Length; i++)
        {
            keyframes[i] = new ObjectReferenceKeyframe
            {
                time = i / clip.frameRate,
                value = sprites[i]
            };
        }

        AnimationUtility.SetObjectReferenceCurve(clip, binding, keyframes);

        string animPath = Path.Combine(folder, fileName + ".anim");
        string unityAnimPath = "Assets" + animPath.Replace(projectPath, "").Replace("\\", "/");
        AssetDatabase.CreateAsset(clip, unityAnimPath);

        string controllerPath = Path.Combine(folder, fileName + "_controller.controller");
        string unityControllerPath = "Assets" + controllerPath.Replace(projectPath, "").Replace("\\", "/");
        var controller = UnityEditor.Animations.AnimatorController.CreateAnimatorControllerAtPath(unityControllerPath);
        controller.AddMotion(clip);

        AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();

        GameObject obj = new GameObject(fileName);
        var renderer = obj.AddComponent<SpriteRenderer>();
        renderer.sprite = sprites[0];
        var animator = obj.AddComponent<Animator>();
        animator.runtimeAnimatorController = controller;
        obj.transform.position = Vector3.zero;
        obj.transform.localScale = Vector3.one * 2f;

        Camera cam = Camera.main;
        if (cam != null)
        {
            cam.transform.position = new Vector3(0, 0, -10);
            cam.orthographic = true;
        }

        Debug.Log("Spawned GameObject in scene for " + fileName);
    }

    static void RunFFmpeg(string ffmpegPath, string args)
    {
        try
        {
            ProcessStartInfo startInfo = new ProcessStartInfo
            {
                FileName = ffmpegPath,
                Arguments = args,
                UseShellExecute = false,
                RedirectStandardError = true,
                RedirectStandardOutput = true,
                CreateNoWindow = true
            };

            using (Process proc = Process.Start(startInfo))
            {
                string output = proc.StandardError.ReadToEnd();
                string stdout = proc.StandardOutput.ReadToEnd();
                proc.WaitForExit();

                Debug.Log("ffmpeg output:\n" + stdout);
                Debug.Log("ffmpeg error:\n" + output);
            }
        }
        catch (System.Exception ex)
        {
            Debug.LogError("FFmpeg failed: " + ex.Message);
        }
    }

}