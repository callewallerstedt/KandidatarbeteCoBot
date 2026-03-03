using System;
using System.Collections;
using System.IO;
using UnityEngine;

[Serializable]
public class UnityCaptureCommand
{
    public string command;
    public int count = 1;
    public string run_name;
    public long timestamp;
}

public class UnityCommandBridge : MonoBehaviour
{
    [Header("Wire existing components")]
    public ObjectBoxRandomizer randomizer;
    public GripPoseExporter exporter;

    [Header("Command folder")]
    public string unityExportRoot = "D:/unity_export";
    public float pollSeconds = 0.5f;

    private bool busy = false;

    private void Start()
    {
        StartCoroutine(PollLoop());
    }

    private IEnumerator PollLoop()
    {
        while (true)
        {
            if (!busy)
            {
                TryProcessCommand();
            }
            yield return new WaitForSeconds(Mathf.Max(0.1f, pollSeconds));
        }
    }

    private void TryProcessCommand()
    {
        string cmdPath = Path.Combine(unityExportRoot, "_commands", "next_command.json");
        if (!File.Exists(cmdPath)) return;

        string txt = File.ReadAllText(cmdPath);
        UnityCaptureCommand cmd = JsonUtility.FromJson<UnityCaptureCommand>(txt);
        if (cmd == null || string.IsNullOrWhiteSpace(cmd.command))
        {
            File.Delete(cmdPath);
            return;
        }

        File.Delete(cmdPath);

        if (cmd.command == "randomize_once")
        {
            randomizer?.RandomizeOnce();
            Debug.Log("[UnityCommandBridge] randomize_once done");
            return;
        }

        if (cmd.command == "capture")
        {
            int n = Mathf.Max(1, cmd.count);
            StartCoroutine(CaptureN(n));
        }
    }

    private IEnumerator CaptureN(int n)
    {
        busy = true;

        string rgbDir = Path.Combine(unityExportRoot, "RGB");
        int startIdx = 1;
        if (Directory.Exists(rgbDir))
        {
            startIdx = Directory.GetFiles(rgbDir, "frame_*.png").Length + 1;
        }

        for (int i = 0; i < n; i++)
        {
            randomizer?.RandomizeOnce();
            yield return null;
            exporter?.CaptureFrame(startIdx + i);
            if ((i + 1) % 25 == 0)
            {
                Debug.Log($"[UnityCommandBridge] captured {i + 1}/{n}");
            }
            yield return null;
        }

        busy = false;
        Debug.Log($"[UnityCommandBridge] capture done: {n}");
    }
}
