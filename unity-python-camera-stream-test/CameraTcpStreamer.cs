using System;
using System.Collections.Concurrent;
using System.Net.Sockets;
using System.Threading;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;

[RequireComponent(typeof(Camera))]
public class CameraTcpStreamer : MonoBehaviour
{
    public enum EncodeMode
    {
        JPG,
        PNG
    }

    [Header("Network")]
    public string host = "127.0.0.1";
    public int port = 5000;

    [Header("Capture")]
    public int width = 1280;
    public int height = 720;
    [Range(1, 60)] public int fps = 24;
    public EncodeMode encodeMode = EncodeMode.JPG;
    [Range(10, 100)] public int jpegQuality = 80;

    [Header("Performance")]
    [Tooltip("Max pending frames in queue. 1 = always latest frame (lowest latency).")]
    [Range(1, 6)] public int maxQueueSize = 2;

    private Camera cam;
    private RenderTexture rt;
    private Texture2D encodeTex;

    private TcpClient client;
    private NetworkStream stream;

    private float nextCaptureTime;
    private bool readbackInFlight;

    private readonly ConcurrentQueue<byte[]> frameQueue = new ConcurrentQueue<byte[]>();
    private readonly AutoResetEvent queueSignal = new AutoResetEvent(false);
    private Thread senderThread;
    private volatile bool senderRunning;

    void Start()
    {
        cam = GetComponent<Camera>();

        rt = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32);
        rt.Create();

        encodeTex = new Texture2D(width, height, TextureFormat.RGBA32, false);
        cam.targetTexture = rt;

        StartSenderThread();
    }

    void StartSenderThread()
    {
        senderRunning = true;
        senderThread = new Thread(SenderLoop)
        {
            IsBackground = true,
            Name = $"CameraTcpStreamerSender-{name}-{port}"
        };
        senderThread.Start();
    }

    void TryConnect()
    {
        try
        {
            client = new TcpClient();
            client.NoDelay = true;
            client.SendTimeout = 2000;
            client.Connect(host, port);
            stream = client.GetStream();
            Debug.Log($"[{name}] Connected to {host}:{port}");
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[{name}] Connect failed: {e.Message}");
            SafeClose();
        }
    }

    void Update()
    {
        if (Time.time < nextCaptureTime) return;
        nextCaptureTime = Time.time + (1f / fps);

        if (readbackInFlight) return;
        if (!rt.IsCreated()) return;

        readbackInFlight = true;
        AsyncGPUReadback.Request(rt, 0, TextureFormat.RGBA32, OnReadbackComplete);
    }

    void OnReadbackComplete(AsyncGPUReadbackRequest req)
    {
        readbackInFlight = false;

        if (!senderRunning) return;
        if (req.hasError) return;

        try
        {
            NativeArray<byte> data = req.GetData<byte>();
            encodeTex.LoadRawTextureData(data);
            encodeTex.Apply(false, false);

            byte[] encoded = encodeMode == EncodeMode.PNG
                ? encodeTex.EncodeToPNG()
                : encodeTex.EncodeToJPG(jpegQuality);

            if (encoded == null || encoded.Length == 0) return;

            while (frameQueue.Count >= maxQueueSize && frameQueue.TryDequeue(out _))
            {
                // Drop oldest frame to keep latency down.
            }

            frameQueue.Enqueue(encoded);
            queueSignal.Set();
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[{name}] Encode queue failed: {e.Message}");
        }
    }

    void SenderLoop()
    {
        while (senderRunning)
        {
            try
            {
                if (client == null || !client.Connected || stream == null)
                {
                    TryConnect();
                    Thread.Sleep(200);
                    continue;
                }

                if (!frameQueue.TryDequeue(out byte[] encoded))
                {
                    queueSignal.WaitOne(50);
                    continue;
                }

                byte[] len = BitConverter.GetBytes(encoded.Length); // little-endian int32
                stream.Write(len, 0, 4);
                stream.Write(encoded, 0, encoded.Length);
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[{name}] Send failed: {e.Message}");
                SafeClose();
                Thread.Sleep(300);
            }
        }
    }

    void OnDestroy()
    {
        senderRunning = false;
        queueSignal.Set();

        try
        {
            if (senderThread != null && senderThread.IsAlive)
                senderThread.Join(800);
        }
        catch { }

        SafeClose();

        if (cam != null) cam.targetTexture = null;
        if (rt != null)
        {
            rt.Release();
            Destroy(rt);
        }
        if (encodeTex != null) Destroy(encodeTex);
    }

    void SafeClose()
    {
        try { stream?.Close(); } catch { }
        try { client?.Close(); } catch { }
        stream = null;
        client = null;
    }
}
