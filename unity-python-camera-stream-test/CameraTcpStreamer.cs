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
    public enum EncodeMode { JPG, PNG }

    [Header("RGB stream")]
    public string host = "127.0.0.1";
    public int port = 5000;
    public int width = 1280;
    public int height = 720;
    [Range(1, 60)] public int fps = 24;
    public EncodeMode encodeMode = EncodeMode.JPG;
    [Range(10, 100)] public int jpegQuality = 80;
    [Range(1, 6)] public int maxQueueSize = 2;

    [Header("Mask stream (same script)")]
    public bool enableMaskStream = false;
    public int maskPort = 6000;
    public string maskLayerName = "SegmentationMask";
    public Shader maskShader;

    [Header("Internal")]
    public bool maskOnlyMode = false;

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

        SetupCameraForMode();

        rt = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32);
        rt.Create();
        cam.targetTexture = rt;

        encodeTex = new Texture2D(width, height, TextureFormat.RGBA32, false);

        StartSenderThread();

        if (enableMaskStream && !maskOnlyMode)
        {
            SpawnMaskChildStreamer();
        }
    }

    void SetupCameraForMode()
    {
        if (!maskOnlyMode) return;

        cam.clearFlags = CameraClearFlags.SolidColor;
        cam.backgroundColor = Color.black;
        cam.allowHDR = false;
        cam.allowMSAA = false;

        int layer = LayerMask.NameToLayer(maskLayerName);
        if (layer < 0)
        {
            Debug.LogError($"[{name}] Layer '{maskLayerName}' not found.");
            layer = 0;
        }
        cam.cullingMask = 1 << layer;

        if (maskShader == null)
            maskShader = Shader.Find("Hidden/ObjectMaskRed");

        if (maskShader != null)
            cam.SetReplacementShader(maskShader, "");
        else
            Debug.LogError($"[{name}] Mask shader missing (Hidden/ObjectMaskRed).");

        encodeMode = EncodeMode.PNG;
    }

    void SpawnMaskChildStreamer()
    {
        var child = new GameObject($"{name}_MaskStream");
        child.transform.SetParent(transform, false);

        var childCam = child.AddComponent<Camera>();
        childCam.CopyFrom(cam);
        childCam.depth = cam.depth - 1;

        var childStreamer = child.AddComponent<CameraTcpStreamer>();
        childStreamer.host = host;
        childStreamer.port = maskPort;
        childStreamer.width = width;
        childStreamer.height = height;
        childStreamer.fps = fps;
        childStreamer.encodeMode = EncodeMode.PNG;
        childStreamer.jpegQuality = jpegQuality;
        childStreamer.maxQueueSize = maxQueueSize;

        childStreamer.enableMaskStream = false;
        childStreamer.maskOnlyMode = true;
        childStreamer.maskLayerName = maskLayerName;
        childStreamer.maskShader = maskShader;

        Debug.Log($"[{name}] Spawned mask stream on port {maskPort} (red-on-black PNG).");
    }

    void StartSenderThread()
    {
        senderRunning = true;
        senderThread = new Thread(SenderLoop) { IsBackground = true, Name = $"CamSender-{name}-{port}" };
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
        }
        catch
        {
            SafeClose();
        }
    }

    void Update()
    {
        if (Time.time < nextCaptureTime) return;
        nextCaptureTime = Time.time + (1f / fps);

        if (readbackInFlight || !rt.IsCreated()) return;

        readbackInFlight = true;
        AsyncGPUReadback.Request(rt, 0, TextureFormat.RGBA32, OnReadbackComplete);
    }

    void OnReadbackComplete(AsyncGPUReadbackRequest req)
    {
        readbackInFlight = false;
        if (!senderRunning || req.hasError) return;

        try
        {
            NativeArray<byte> data = req.GetData<byte>();
            encodeTex.LoadRawTextureData(data);
            encodeTex.Apply(false, false);

            byte[] encoded = encodeMode == EncodeMode.PNG ? encodeTex.EncodeToPNG() : encodeTex.EncodeToJPG(jpegQuality);
            if (encoded == null || encoded.Length == 0) return;

            while (frameQueue.Count >= maxQueueSize && frameQueue.TryDequeue(out _)) { }
            frameQueue.Enqueue(encoded);
            queueSignal.Set();
        }
        catch { }
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

                byte[] len = BitConverter.GetBytes(encoded.Length);
                stream.Write(len, 0, 4);
                stream.Write(encoded, 0, encoded.Length);
            }
            catch
            {
                SafeClose();
                Thread.Sleep(300);
            }
        }
    }

    void OnDestroy()
    {
        senderRunning = false;
        queueSignal.Set();

        try { if (senderThread != null && senderThread.IsAlive) senderThread.Join(800); } catch { }

        SafeClose();

        if (cam != null) cam.targetTexture = null;
        if (rt != null) { rt.Release(); Destroy(rt); }
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
