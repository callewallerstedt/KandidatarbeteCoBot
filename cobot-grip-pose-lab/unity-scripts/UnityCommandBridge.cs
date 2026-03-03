using System;
using System.Collections;
using System.IO;
using UnityEngine;

[Serializable]
public class UnityCaptureCommand
{
    public string command;
    public int count = 1;
    public string profile = "cobot"; // cobot|roof|all
    public string run_name;
    public long timestamp;
}

public class UnityCommandBridge : MonoBehaviour
{
    [Header("Wire existing components")]
    public ObjectBoxRandomizer randomizer;

    [Header("Camera exporters")]
    public GripPoseExporter cobotExporter;
    public GripPoseExporter[] roofExporters;

    [Header("Command folder")]
    public string unityExportRoot = "D:/unity_export";
    public float pollSeconds = 0.5f;

    private bool busy = false;

    private void Start()
    {
        Application.runInBackground = true;
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
            StartCoroutine(CaptureN(n, (cmd.profile ?? "cobot").ToLowerInvariant()));
        }
    }

    private GripPoseExporter[] SelectExporters(string profile)
    {
        if (profile == "all")
        {
            var list = new System.Collections.Generic.List<GripPoseExporter>();
            if (cobotExporter != null) list.Add(cobotExporter);
            if (roofExporters != null)
            {
                foreach (var r in roofExporters) if (r != null) list.Add(r);
            }
            return list.ToArray();
        }
        if (profile == "roof")
        {
            return roofExporters ?? new GripPoseExporter[0];
        }
        return new GripPoseExporter[] { cobotExporter };
    }

    private int GetNextIndex(GripPoseExporter ex)
    {
        if (ex == null) return 1;
        string dir = Path.Combine(unityExportRoot, "RGB", ex.cameraTag);
        if (!Directory.Exists(dir)) return 1;
        return Directory.GetFiles(dir, "frame_*.png").Length + 1;
    }

    private IEnumerator CaptureN(int n, string profile)
    {
        busy = true;

        var exporters = SelectExporters(profile);
        if (exporters == null || exporters.Length == 0)
        {
            busy = false;
            Debug.LogWarning("[UnityCommandBridge] no exporters assigned for selected profile");
            yield break;
        }

        var starts = new int[exporters.Length];
        for (int e = 0; e < exporters.Length; e++) starts[e] = GetNextIndex(exporters[e]);

        for (int i = 0; i < n; i++)
        {
            randomizer?.RandomizeOnce();
            yield return null;

            for (int e = 0; e < exporters.Length; e++)
            {
                if (exporters[e] == null) continue;
                exporters[e].CaptureFrame(starts[e] + i);
            }

            if ((i + 1) % 25 == 0)
            {
                Debug.Log($"[UnityCommandBridge] captured {i + 1}/{n} profile={profile}");
            }
            yield return null;
        }

        busy = false;
        Debug.Log($"[UnityCommandBridge] capture done: {n} profile={profile}");
    }
}
