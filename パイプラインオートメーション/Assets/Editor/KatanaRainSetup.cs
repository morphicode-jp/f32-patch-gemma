using UnityEngine;
using UnityEditor;

/// <summary>
/// AETHER Demo: Katana Rain Physics Setup
/// Adds Rigidbody components to all imported katanas for physics simulation.
/// </summary>
public class KatanaRainSetup : EditorWindow
{
    [MenuItem("AETHER/Setup Katana Rain Physics")]
    public static void ShowWindow()
    {
        GetWindow<KatanaRainSetup>("Katana Rain Setup");
    }

    private void OnGUI()
    {
        GUILayout.Label("Katana Rain Physics Setup", EditorStyles.boldLabel);
        GUILayout.Space(10);
        
        GUILayout.Label("This will add Rigidbody components to all katanas.");
        GUILayout.Label("Make sure 'KatanaRain_50' is in the scene.");
        GUILayout.Space(20);
        
        if (GUILayout.Button("Add Rigidbody to All Katanas", GUILayout.Height(40)))
        {
            SetupPhysics();
        }
        
        GUILayout.Space(10);
        
        if (GUILayout.Button("Add Ground Plane", GUILayout.Height(30)))
        {
            AddGroundPlane();
        }
        
        GUILayout.Space(10);
        
        if (GUILayout.Button("Setup Camera", GUILayout.Height(30)))
        {
            SetupCamera();
        }
    }

    private void SetupPhysics()
    {
        // Find all katana objects in the scene
        var allObjects = FindObjectsByType<Transform>(FindObjectsSortMode.None);
        int count = 0;
        
        foreach (var t in allObjects)
        {
            if (t.name.Contains("Katana") && t.GetComponent<MeshFilter>() != null)
            {
                // Add Rigidbody if not present
                if (t.GetComponent<Rigidbody>() == null)
                {
                    var rb = t.gameObject.AddComponent<Rigidbody>();
                    rb.mass = 1.5f;  // Katana weight ~1.5kg
                    rb.useGravity = true;
                    count++;
                }
                
                // Add MeshCollider if not present
                if (t.GetComponent<Collider>() == null)
                {
                    var col = t.gameObject.AddComponent<MeshCollider>();
                    col.convex = true;
                }
            }
        }
        
        Debug.Log($"[AETHER] Added Rigidbody to {count} katanas!");
        EditorUtility.DisplayDialog("Katana Rain Setup", 
            $"Added Rigidbody to {count} katanas.\nPress Play to start simulation!", 
            "OK");
    }

    private void AddGroundPlane()
    {
        // Check if ground already exists
        if (GameObject.Find("Ground") != null)
        {
            Debug.Log("[AETHER] Ground already exists.");
            return;
        }
        
        // Create ground plane
        var ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
        ground.name = "Ground";
        ground.transform.position = Vector3.zero;
        ground.transform.localScale = new Vector3(5, 1, 5);  // 50x50 units
        
        // Dark material
        var renderer = ground.GetComponent<Renderer>();
        var mat = new Material(Shader.Find("Standard"));
        mat.color = new Color(0.1f, 0.1f, 0.15f);
        renderer.material = mat;
        
        Debug.Log("[AETHER] Ground plane added!");
    }

    private void SetupCamera()
    {
        var cam = Camera.main;
        if (cam == null)
        {
            Debug.LogWarning("[AETHER] No main camera found!");
            return;
        }
        
        // Position camera to see the falling katanas
        cam.transform.position = new Vector3(8, 6, -10);
        cam.transform.LookAt(Vector3.up * 5);
        
        Debug.Log("[AETHER] Camera positioned for katana rain view!");
    }
}
