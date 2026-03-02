using System;
using System.Net.Sockets;
using UnityEngine;

[RequireComponent(typeof(Camera))]
public class CameraTcpStreamer : MonoBehaviour
{
    [Header("Network")]
    public string host = "127.0.0.1";
    public int port = 5000;

    [Header("Capture (lower defaults to reduce Unity lag)")]
    public int width = 320;
    public int height = 180;
    [Range(1, 30)] public int fps = 8;
    [Range(10, 100)] public int jpegQuality = 55;

    private Camera cam;
    private RenderTexture rt;
    private Texture2D tex;
    private TcpClient client;
    private NetworkStream stream;
    private float nextTime;

    void Start()
    {
        cam = GetComponent<Camera>();

        // Keep this camera from rendering to screen if it's only a stream camera.
        // (You can still override in inspector.)
        if (cam.targetDisplay == 0)
            cam.targetDisplay = 1;

        rt = new RenderTexture(width, height, 24);
        tex = new Texture2D(width, height, TextureFormat.RGB24, false);
        cam.targetTexture = rt;

        TryConnect();
    }

    void TryConnect()
    {
        try
        {
            client = new TcpClient();
            client.NoDelay = true;
            client.Connect(host, port);
            stream = client.GetStream();
            Debug.Log($"[{name}] Connected to {host}:{port}");
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[{name}] Connect failed: {e.Message}");
        }
    }

    void Update()
    {
        if (Time.time < nextTime) return;
        nextTime = Time.time + (1f / fps);

        if (client == null || !client.Connected || stream == null)
        {
            TryConnect();
            return;
        }

        try
        {
            // NOTE: Do NOT call cam.Render() here. It causes extra rendering and heavy lag.
            RenderTexture current = RenderTexture.active;
            RenderTexture.active = rt;

            tex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
            tex.Apply(false, false);

            RenderTexture.active = current;

            byte[] jpg = tex.EncodeToJPG(jpegQuality);
            byte[] len = BitConverter.GetBytes(jpg.Length); // little-endian int32

            stream.Write(len, 0, 4);
            stream.Write(jpg, 0, jpg.Length);
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[{name}] Send failed: {e.Message}");
            SafeClose();
        }
    }

    void OnDestroy()
    {
        SafeClose();
        if (cam != null) cam.targetTexture = null;
        if (rt != null) rt.Release();
    }

    void SafeClose()
    {
        try { stream?.Close(); } catch { }
        try { client?.Close(); } catch { }
        stream = null;
        client = null;
    }
}
