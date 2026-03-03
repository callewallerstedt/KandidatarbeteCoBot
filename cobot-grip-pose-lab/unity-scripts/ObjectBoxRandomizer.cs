using System.Collections.Generic;
using UnityEngine;

public class ObjectBoxRandomizer : MonoBehaviour
{
    [Header("Objects to randomize")]
    public List<Transform> objects = new List<Transform>();

    [Header("Sampling box (local to this transform)")]
    public Vector3 boxCenter = new Vector3(0f, 0.2f, 0f);
    public Vector3 boxSize = new Vector3(0.4f, 0.2f, 0.3f);

    [Header("Rotation")]
    public bool randomYaw = true;
    public bool randomPitch = true;
    public bool randomRoll = true;

    [Header("Scale jitter")]
    public bool scaleJitter = false;
    public float scaleMin = 0.95f;
    public float scaleMax = 1.05f;

    [Header("Collision attempts")]
    public int maxPlacementTries = 30;
    public float minSeparation = 0.04f;

    private List<Vector3> placed = new List<Vector3>();

    public void RandomizeOnce()
    {
        placed.Clear();

        foreach (var t in objects)
        {
            if (t == null) continue;

            Vector3 p = SamplePositionNonOverlap();
            t.position = p;

            float rx = randomPitch ? Random.Range(0f, 360f) : t.eulerAngles.x;
            float ry = randomYaw ? Random.Range(0f, 360f) : t.eulerAngles.y;
            float rz = randomRoll ? Random.Range(0f, 360f) : t.eulerAngles.z;
            t.rotation = Quaternion.Euler(rx, ry, rz);

            if (scaleJitter)
            {
                float s = Random.Range(scaleMin, scaleMax);
                t.localScale = Vector3.one * s;
            }

            placed.Add(p);
        }
    }

    private Vector3 SamplePositionNonOverlap()
    {
        for (int i = 0; i < Mathf.Max(1, maxPlacementTries); i++)
        {
            Vector3 local = boxCenter + new Vector3(
                Random.Range(-boxSize.x * 0.5f, boxSize.x * 0.5f),
                Random.Range(-boxSize.y * 0.5f, boxSize.y * 0.5f),
                Random.Range(-boxSize.z * 0.5f, boxSize.z * 0.5f)
            );
            Vector3 world = transform.TransformPoint(local);

            bool ok = true;
            foreach (var p in placed)
            {
                if (Vector3.Distance(world, p) < minSeparation) { ok = false; break; }
            }
            if (ok) return world;
        }

        // fallback
        return transform.TransformPoint(boxCenter);
    }

    private void OnDrawGizmosSelected()
    {
        Gizmos.color = new Color(0f, 1f, 1f, 0.25f);
        Matrix4x4 old = Gizmos.matrix;
        Gizmos.matrix = transform.localToWorldMatrix;
        Gizmos.DrawCube(boxCenter, boxSize);
        Gizmos.color = Color.cyan;
        Gizmos.DrawWireCube(boxCenter, boxSize);
        Gizmos.matrix = old;
    }
}
