using System;
using System.IO;
using UnityEngine;

public static class PythonLocator
{
    /// <summary>
    /// Try to find a Conda prefix folder such as ~/miniconda3 or ~/anaconda3.
    /// Returns null if nothing is found.
    /// </summary>
    public static string DetectCondaPrefix()
    {
        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

#if UNITY_EDITOR_WIN
        string[] candidates =
        {
            Path.Combine(home, "miniconda3"),
            Path.Combine(home, "anaconda3"),
            @"C:\ProgramData\miniconda3",
            @"C:\ProgramData\anaconda3"
        };
#elif UNITY_EDITOR_OSX
        string[] candidates =
        {
            Path.Combine(home, "miniconda3"),
            Path.Combine(home, "anaconda3"),
            "/opt/miniconda3",
            "/opt/anaconda3"
        };
#else // Linux or others
        string[] candidates =
        {
            Path.Combine(home, "miniconda3"),
            Path.Combine(home, "anaconda3"),
            "/opt/miniconda3",
            "/opt/anaconda3"
        };
#endif

        foreach (var c in candidates)
        {
            if (!string.IsNullOrEmpty(c) && Directory.Exists(c))
            {
                Debug.Log($"[PythonLocator] Found Conda prefix at: {c}");
                return c;
            }
        }

        Debug.LogWarning("[PythonLocator] No miniconda3/anaconda3 folder found in common locations.");
        return null;
    }

    /// <summary>
    /// Try to build the full python path for a given env name under the detected Conda prefix.
    /// Returns null if detection fails.
    /// </summary>
    public static string DetectPythonForEnv(string envName)
    {
        string prefix = DetectCondaPrefix();
        if (string.IsNullOrEmpty(prefix))
            return null;

#if UNITY_EDITOR_WIN
        string envPath   = Path.Combine(prefix, "envs", envName);
        string pythonExe = Path.Combine(envPath, "python.exe");
#else
        string envPath   = Path.Combine(prefix, "envs", envName);
        string pythonExe = Path.Combine(envPath, "bin/python");
#endif

        if (File.Exists(pythonExe))
        {
            Debug.Log($"[PythonLocator] Using python at: {pythonExe}");
            return pythonExe;
        }

        Debug.LogWarning($"[PythonLocator] Env '{envName}' not found under {prefix} (looked for {pythonExe}).");
        return null;
    }
}
