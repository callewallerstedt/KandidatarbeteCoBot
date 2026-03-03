using UnityEngine;

public class GripAnnotatable : MonoBehaviour
{
    [Header("Class")]
    public int classId = 0;

    [Header("Keypoints")]
    public Transform centerPoint;
    public Transform gripPointA;
    public Transform gripPointB;

    [Header("Bounds source (optional)")]
    public Renderer[] renderers;

    public Bounds GetWorldBounds()
    {
        if (renderers != null && renderers.Length > 0)
        {
            bool has = false;
            Bounds b = new Bounds(transform.position, Vector3.zero);
            foreach (var r in renderers)
            {
                if (r == null) continue;
                if (!has) { b = r.bounds; has = true; }
                else b.Encapsulate(r.bounds);
            }
            if (has) return b;
        }

        var fallback = GetComponentInChildren<Renderer>();
        if (fallback != null) return fallback.bounds;
        return new Bounds(transform.position, Vector3.one * 0.05f);
    }
}
