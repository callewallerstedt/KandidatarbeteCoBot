using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

public class GripPoseExporter : MonoBehaviour
{
    public Camera renderCamera;
    [Header("Optional mask camera (renders red object on black)")]
    public Camera maskCamera;
    public string maskLayerName = "SegmentationMask";
    public Shader maskShader;
    public bool forceMaskReplacementShader = true;
    public bool forceFlatRedMaterialOverride = true;

    public string outputRoot = "unity_export";
    public string cameraTag = "cobot"; // cobot, roof1, roof2 ...
    public int width = 1920;
    public int height = 1080;
    public int pngQuality = 100;

    [Header("Annotated objects")]
    public List<GripAnnotatable> objects = new List<GripAnnotatable>();

    [Serializable]
    public class ObjAnn
    {
        public int class_id;
        public float[] bbox_xyxy;
        public float[] center;
        public float[] grip_a;
        public float[] grip_b;
    }

    [Serializable]
    public class FrameAnn
    {
        public string image;
        public int width;
        public int height;
        public List<ObjAnn> objects = new List<ObjAnn>();
    }

    private void EnsureDirs()
    {
        Directory.CreateDirectory(Path.Combine(outputRoot, "RGB", cameraTag));
        Directory.CreateDirectory(Path.Combine(outputRoot, "MASK", cameraTag));
        Directory.CreateDirectory(Path.Combine(outputRoot, "annotations", cameraTag));
    }

    private void EnsureObjectList()
    {
        if (objects != null && objects.Count > 0) return;
        objects = FindObjectsByType<GripAnnotatable>(FindObjectsSortMode.None).ToList();
        Debug.Log($"[GripPoseExporter] Auto-found GripAnnotatable objects: {objects.Count}");
    }

    private void ConfigureMaskCameraIfNeeded()
    {
        Camera cam = maskCamera != null ? maskCamera : renderCamera;
        if (cam == null || !forceMaskReplacementShader) return;

        int layer = LayerMask.NameToLayer(maskLayerName);
        if (layer >= 0)
            cam.cullingMask = 1 << layer;

        cam.clearFlags = CameraClearFlags.SolidColor;
        cam.backgroundColor = Color.black;
        cam.allowHDR = false;
        cam.allowMSAA = false;

        if (maskShader == null)
            maskShader = Shader.Find("Hidden/ObjectMaskRed");

        if (maskShader != null)
            cam.SetReplacementShader(maskShader, "");
        else
            Debug.LogWarning("[GripPoseExporter] maskShader missing. Assign ObjectMaskRed shader.");
    }

    public void CaptureFrame(int frameIndex)
    {
        if (renderCamera == null)
        {
            Debug.LogError("GripPoseExporter: renderCamera is null");
            return;
        }

        EnsureDirs();
        EnsureObjectList();

        string imageName = $"frame_{frameIndex:D6}.png";
        string rgbPath = Path.Combine(outputRoot, "RGB", cameraTag, imageName);
        string maskPath = Path.Combine(outputRoot, "MASK", cameraTag, imageName);

        // 1) Always save normal RGB first
        SaveCameraPng(renderCamera, rgbPath);

        // 2) Save mask either from dedicated mask camera or by temporarily switching render camera
        if (maskCamera != null)
        {
            ConfigureMaskCameraIfNeeded();
            SaveCameraPng(maskCamera, maskPath);
        }
        else
        {
            SaveMaskFromRenderCamera(maskPath);
        }

        FrameAnn ann = new FrameAnn { image = imageName, width = width, height = height };

        int missingKp = 0;
        int behindCam = 0;
        int badBox = 0;

        foreach (var obj in objects)
        {
            if (obj == null || obj.centerPoint == null || obj.gripPointA == null || obj.gripPointB == null)
            {
                missingKp++;
                continue;
            }

            Vector3 c = renderCamera.WorldToViewportPoint(obj.centerPoint.position);
            Vector3 a = renderCamera.WorldToViewportPoint(obj.gripPointA.position);
            Vector3 b = renderCamera.WorldToViewportPoint(obj.gripPointB.position);

            // Skip if keypoints behind camera
            if (c.z <= 0f || a.z <= 0f || b.z <= 0f)
            {
                behindCam++;
                continue;
            }

            if (!TryProjectBounds(obj.GetWorldBounds(), out float x1, out float y1, out float x2, out float y2))
            {
                badBox++;
                continue;
            }

            // Viewport -> image pixel coordinates (top-left origin for exported labels)
            float cx = Mathf.Clamp(c.x * width, 0, width - 1);
            float cy = Mathf.Clamp((1f - c.y) * height, 0, height - 1);
            float ax = Mathf.Clamp(a.x * width, 0, width - 1);
            float ay = Mathf.Clamp((1f - a.y) * height, 0, height - 1);
            float bx = Mathf.Clamp(b.x * width, 0, width - 1);
            float by = Mathf.Clamp((1f - b.y) * height, 0, height - 1);

            ann.objects.Add(new ObjAnn
            {
                class_id = obj.classId,
                bbox_xyxy = new[] { x1, y1, x2, y2 },
                center = new[] { cx, cy, 2f },
                grip_a = new[] { ax, ay, 2f },
                grip_b = new[] { bx, by, 2f },
            });
        }

        string jsonPath = Path.Combine(outputRoot, "annotations", cameraTag, $"frame_{frameIndex:D6}.json");
        File.WriteAllText(jsonPath, JsonUtility.ToJson(ann, true));
        Debug.Log($"[GripPoseExporter] frame {frameIndex}: objects={ann.objects.Count} missingKp={missingKp} behindCam={behindCam} badBox={badBox} rgb={rgbPath} mask={maskPath} ann={jsonPath} mode=" + (maskCamera != null ? "dual-camera" : "single-camera"));
    }

    private Shader FindAnyUnlitShader()
    {
        string[] candidates = {
            "Unlit/Color",
            "Universal Render Pipeline/Unlit",
            "HDRP/Unlit",
            "Sprites/Default"
        };
        foreach (var s in candidates)
        {
            var sh = Shader.Find(s);
            if (sh != null) return sh;
        }
        return null;
    }

    private void SaveMaskFromRenderCamera(string outPath)
    {
        if (renderCamera == null) return;

        var go = new GameObject("__TempMaskCam");
        var tempCam = go.AddComponent<Camera>();
        tempCam.CopyFrom(renderCamera);

        int layer = LayerMask.NameToLayer(maskLayerName);
        if (layer >= 0)
            tempCam.cullingMask = 1 << layer;

        tempCam.clearFlags = CameraClearFlags.SolidColor;
        tempCam.backgroundColor = Color.black;
        tempCam.allowHDR = false;
        tempCam.allowMSAA = false;

        bool usedMaterialOverride = false;
        List<Renderer> layerRenderers = null;
        List<Material[]> oldMats = null;
        Material redMat = null;

        if (forceFlatRedMaterialOverride)
        {
            layerRenderers = FindObjectsByType<Renderer>(FindObjectsSortMode.None)
                .Where(r => r != null && r.gameObject.layer == layer)
                .ToList();
            oldMats = new List<Material[]>();

            var unlit = FindAnyUnlitShader();
            if (unlit != null)
            {
                redMat = new Material(unlit);
                if (redMat.HasProperty("_BaseColor")) redMat.SetColor("_BaseColor", Color.red);
                if (redMat.HasProperty("_Color")) redMat.SetColor("_Color", Color.red);

                foreach (var r in layerRenderers)
                {
                    var old = r.sharedMaterials;
                    oldMats.Add(old);
                    var rep = new Material[old.Length];
                    for (int i = 0; i < rep.Length; i++) rep[i] = redMat;
                    r.sharedMaterials = rep;
                }
                usedMaterialOverride = true;
            }
        }

        if (!usedMaterialOverride)
        {
            if (maskShader == null)
                maskShader = Shader.Find("Hidden/ObjectMaskRed");
            if (maskShader != null)
                tempCam.SetReplacementShader(maskShader, "");
        }

        SaveCameraPng(tempCam, outPath);

        if (!usedMaterialOverride)
        {
            tempCam.ResetReplacementShader();
        }
        else if (layerRenderers != null && oldMats != null)
        {
            for (int i = 0; i < layerRenderers.Count && i < oldMats.Count; i++)
                layerRenderers[i].sharedMaterials = oldMats[i];
            if (redMat != null) Destroy(redMat);
        }

        Destroy(go);
    }

    private void SaveCameraPng(Camera cam, string outPath)
    {
        var rt = new RenderTexture(width, height, 24);
        cam.targetTexture = rt;
        var tex = new Texture2D(width, height, TextureFormat.RGB24, false);

        cam.Render();
        RenderTexture.active = rt;
        tex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        tex.Apply();

        File.WriteAllBytes(outPath, tex.EncodeToPNG());

        cam.targetTexture = null;
        RenderTexture.active = null;
        Destroy(rt);
        Destroy(tex);
    }

    private bool TryProjectBounds(Bounds b, out float x1, out float y1, out float x2, out float y2)
    {
        x1 = float.MaxValue; y1 = float.MaxValue;
        x2 = float.MinValue; y2 = float.MinValue;

        Vector3 c = b.center;
        Vector3 e = b.extents;
        Vector3[] pts = new Vector3[8]
        {
            c + new Vector3(-e.x,-e.y,-e.z),
            c + new Vector3( e.x,-e.y,-e.z),
            c + new Vector3(-e.x, e.y,-e.z),
            c + new Vector3( e.x, e.y,-e.z),
            c + new Vector3(-e.x,-e.y, e.z),
            c + new Vector3( e.x,-e.y, e.z),
            c + new Vector3(-e.x, e.y, e.z),
            c + new Vector3( e.x, e.y, e.z),
        };

        bool anyFront = false;
        foreach (var p in pts)
        {
            Vector3 sp = renderCamera.WorldToViewportPoint(p);
            if (sp.z <= 0f) continue;
            anyFront = true;
            float px = sp.x * width;
            float py = (1f - sp.y) * height;
            x1 = Mathf.Min(x1, px);
            x2 = Mathf.Max(x2, px);
            y1 = Mathf.Min(y1, py);
            y2 = Mathf.Max(y2, py);
        }

        if (!anyFront) return false;

        x1 = Mathf.Clamp(x1, 0, width - 1);
        x2 = Mathf.Clamp(x2, 0, width - 1);
        y1 = Mathf.Clamp(y1, 0, height - 1);
        y2 = Mathf.Clamp(y2, 0, height - 1);

        if (x2 - x1 < 3 || y2 - y1 < 3) return false;
        return true;
    }
}
