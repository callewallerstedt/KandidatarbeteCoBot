using System.Collections;
using UnityEngine;

public class DatasetCaptureController : MonoBehaviour
{
    public ObjectBoxRandomizer randomizer;
    public GripPoseExporter exporter;

    [Header("Capture settings")]
    public int framesToCapture = 2000;
    public float settleSeconds = 0.02f;
    public bool autoStart = false;

    private bool running = false;

    private void Start()
    {
        if (autoStart) StartCapture();
    }

    [ContextMenu("Start Capture")]
    public void StartCapture()
    {
        if (running) return;
        StartCoroutine(CaptureRoutine());
    }

    private IEnumerator CaptureRoutine()
    {
        running = true;
        for (int i = 1; i <= Mathf.Max(1, framesToCapture); i++)
        {
            if (randomizer != null) randomizer.RandomizeOnce();
            if (settleSeconds > 0f) yield return new WaitForSeconds(settleSeconds);

            if (exporter != null) exporter.CaptureFrame(i);

            if (i % 50 == 0)
                Debug.Log($"[DatasetCapture] Captured {i}/{framesToCapture}");

            yield return null;
        }
        running = false;
        Debug.Log("[DatasetCapture] Done");
    }
}
