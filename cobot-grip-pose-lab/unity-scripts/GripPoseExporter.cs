// Unity helper template for exporting RGB + grip keypoints JSON
// Attach to a camera object and wire object providers.
using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class GripPoseExporter : MonoBehaviour
{
    public Camera renderCamera;
    public string outputRoot = "unity_export";
    public int width = 1920;
    public int height = 1080;

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

    // TODO: Replace with your own object source
    public List<Transform> objectRoots = new List<Transform>();

    void Start()
    {
        Directory.CreateDirectory(Path.Combine(outputRoot, "RGB"));
        Directory.CreateDirectory(Path.Combine(outputRoot, "annotations"));
    }

    public void CaptureFrame(int frameIndex)
    {
        string imageName = $"frame_{frameIndex:D6}.png";
        string rgbPath = Path.Combine(outputRoot, "RGB", imageName);

        var rt = new RenderTexture(width, height, 24);
        renderCamera.targetTexture = rt;
        var tex = new Texture2D(width, height, TextureFormat.RGB24, false);
        renderCamera.Render();
        RenderTexture.active = rt;
        tex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        tex.Apply();
        File.WriteAllBytes(rgbPath, tex.EncodeToPNG());
        renderCamera.targetTexture = null;
        RenderTexture.active = null;
        Destroy(rt);
        Destroy(tex);

        FrameAnn ann = new FrameAnn { image = imageName, width = width, height = height };

        foreach (var t in objectRoots)
        {
            // TODO replace these with real points from your object definition
            Vector3 cW = t.position;
            Vector3 aW = t.position + t.right * 0.03f;
            Vector3 bW = t.position - t.right * 0.03f;

            Vector3 c = renderCamera.WorldToScreenPoint(cW);
            Vector3 a = renderCamera.WorldToScreenPoint(aW);
            Vector3 b = renderCamera.WorldToScreenPoint(bW);

            if (c.z <= 0) continue;

            float x1 = Mathf.Min(a.x, b.x) - 20f;
            float y1 = Mathf.Min(a.y, b.y) - 20f;
            float x2 = Mathf.Max(a.x, b.x) + 20f;
            float y2 = Mathf.Max(a.y, b.y) + 20f;

            ann.objects.Add(new ObjAnn
            {
                class_id = 0,
                bbox_xyxy = new[] { x1, height - y2, x2, height - y1 },
                center = new[] { c.x, height - c.y, 2f },
                grip_a = new[] { a.x, height - a.y, 2f },
                grip_b = new[] { b.x, height - b.y, 2f },
            });
        }

        string json = JsonUtility.ToJson(ann, true);
        File.WriteAllText(Path.Combine(outputRoot, "annotations", $"frame_{frameIndex:D6}.json"), json);
    }
}
