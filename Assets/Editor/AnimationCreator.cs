#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;

public class AnimationCreator : EditorWindow
{
    private GameObject targetModel; 
    private string animationName = "Walk";
    private float duration = 1f; // duration of animation
    private float stepHeight = 0.1f; // how high the steps are, will adjust based on output
    private float bodyBob = 0.01f; // bob of the steps, will adjust based on output
    
    [MenuItem("GenAI@Berkeley/Create Walk Animation")]
    public static void ShowWindow()
    {
        GetWindow<AnimationCreator>("Walk Animation Creator");
    }
    private void CreateWalkAnimation(){
        // TODO: use Transform __body part__ to assign bones, then write methods adjusting position of bones
    }
}

#endif