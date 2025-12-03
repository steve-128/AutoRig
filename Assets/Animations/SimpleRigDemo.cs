using UnityEngine;

public class SimpleRigDemo : MonoBehaviour
{
    public Transform leftShoulder;
    public Transform rightShoulder;
    public Transform leftElbow;
    public Transform rightElbow;
    public Transform leftKnee;
    public Transform rightKnee;
    public Transform neck;

    public float scaleMultiplier = 1.0f;

    private float time;
    private Vector3 neckBasePos;
    private Quaternion leftShoulderBaseRot, rightShoulderBaseRot;
    private Quaternion leftElbowBaseRot, rightElbowBaseRot;
    private Quaternion leftKneeBaseRot, rightKneeBaseRot;
    private Vector3 originalScale;

    void Start()
    {
        originalScale = transform.localScale;
        // transform.localScale = originalScale * scaleMultiplier;

        if (neck != null) neckBasePos = neck.localPosition;

        if (leftShoulder != null) leftShoulderBaseRot = leftShoulder.localRotation;
        if (rightShoulder != null) rightShoulderBaseRot = rightShoulder.localRotation;

        if (leftElbow != null) leftElbowBaseRot = leftElbow.localRotation;
        if (rightElbow != null) rightElbowBaseRot = rightElbow.localRotation;

        if (leftKnee != null) leftKneeBaseRot = leftKnee.localRotation;
        if (rightKnee != null) rightKneeBaseRot = rightKnee.localRotation;
    }

    void Update()
    {
        time += Time.deltaTime;

        if (neck != null)
        {
            float breathe = Mathf.Sin(time * 1.0f) * 0.3f;
            neck.localPosition = neckBasePos + new Vector3(0f, breathe, 0f);
        }

        float shoulderWave = Mathf.Sin(time * 4f) * 18f;

        if (leftShoulder != null)
            leftShoulder.localRotation = leftShoulderBaseRot * Quaternion.Euler(0, 0, shoulderWave);

        if (rightShoulder != null)
            rightShoulder.localRotation = rightShoulderBaseRot * Quaternion.Euler(0, 0, -shoulderWave);

        float elbowBend = Mathf.Sin(time * 4f) * 12f - 12f;

        if (leftElbow != null)
            leftElbow.localRotation = leftElbowBaseRot * Quaternion.Euler(0, 0, elbowBend);

        if (rightElbow != null)
            rightElbow.localRotation = rightElbowBaseRot * Quaternion.Euler(0, 0, -elbowBend);

        float kneeBend = Mathf.Sin(time * 1.5f) * 2f - 2f;

        if (leftKnee != null)
            leftKnee.localRotation = leftKneeBaseRot * Quaternion.Euler(0, 0, kneeBend);

        if (rightKnee != null)
            rightKnee.localRotation = rightKneeBaseRot * Quaternion.Euler(0, 0, -kneeBend);
    }
}
