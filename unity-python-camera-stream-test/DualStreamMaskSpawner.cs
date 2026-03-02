using UnityEngine;

[RequireComponent(typeof(Camera))]
public class DualStreamMaskSpawner : MonoBehaviour
{
    [Header("Mask stream")]
    public string host = "127.0.0.1";
    public int maskPort = 6000;
    public string maskLayerName = "SegmentationMask";

    [Header("Mask stream quality")]
    public int width = 1280;
    public int height = 720;
    [Range(1, 60)] public int fps = 24;
    [Range(1, 6)] public int maxQueueSize = 2;

    private Camera sourceCam;
    private Camera maskCam;

    void Start()
    {
        sourceCam = GetComponent<Camera>();
        CreateMaskCamera();
    }

    void LateUpdate()
    {
        if (sourceCam == null || maskCam == null) return;

        maskCam.transform.SetPositionAndRotation(sourceCam.transform.position, sourceCam.transform.rotation);

        maskCam.orthographic = sourceCam.orthographic;
        maskCam.fieldOfView = sourceCam.fieldOfView;
        maskCam.orthographicSize = sourceCam.orthographicSize;
        maskCam.nearClipPlane = sourceCam.nearClipPlane;
        maskCam.farClipPlane = sourceCam.farClipPlane;
        maskCam.aspect = sourceCam.aspect;
    }

    void CreateMaskCamera()
    {
        var go = new GameObject($"{name}_MaskCamera");
        go.transform.SetParent(transform, false);

        maskCam = go.AddComponent<Camera>();
        maskCam.enabled = true;
        maskCam.clearFlags = CameraClearFlags.SolidColor;
        maskCam.backgroundColor = Color.black;
        maskCam.allowHDR = false;
        maskCam.allowMSAA = false;
        maskCam.depth = sourceCam.depth - 1;

        int layer = LayerMask.NameToLayer(maskLayerName);
        if (layer < 0)
        {
            Debug.LogError($"[{name}] Layer '{maskLayerName}' does not exist. Create it in Unity and assign mask objects to it.");
            layer = 0;
        }
        maskCam.cullingMask = 1 << layer;

        Shader maskShader = Shader.Find("Hidden/ObjectMaskRed");
        if (maskShader == null)
        {
            Debug.LogError($"[{name}] Shader 'Hidden/ObjectMaskRed' not found. Import ObjectMaskRed.shader.");
        }
        else
        {
            maskCam.SetReplacementShader(maskShader, "");
        }

        var streamer = go.AddComponent<CameraTcpStreamer>();
        streamer.host = host;
        streamer.port = maskPort;
        streamer.width = width;
        streamer.height = height;
        streamer.fps = fps;
        streamer.maxQueueSize = maxQueueSize;
        streamer.encodeMode = CameraTcpStreamer.EncodeMode.PNG; // lossless mask edges/colors

        Debug.Log($"[{name}] Mask camera ready on port {maskPort}. Objects on layer '{maskLayerName}' will render red on black.");
    }
}
