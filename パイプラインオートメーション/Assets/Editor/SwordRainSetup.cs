using UnityEngine;
using UnityEditor;

public class SwordRainSetup : EditorWindow
{
    [MenuItem("AETHER/Setup Sword Rain Demo")]
    public static void SetupSwordRain()
    {
        string[] swordNames = new string[] {
            "Sword_Fire", "Sword_Ice", "Sword_Nature", "Sword_Thunder", "Sword_Shadow",
            "Sword_Light", "Sword_Blood", "Sword_Ocean", "Sword_Earth", "Sword_Void"
        };
        
        // Refresh assets first
        AssetDatabase.Refresh();
        
        // Create parent object
        GameObject parent = new GameObject("SwordRain");
        parent.transform.position = Vector3.zero;
        
        int count = 0;
        for (int i = 0; i < swordNames.Length; i++)
        {
            string path = $"Assets/Generated/{swordNames[i]}.fbx";
            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(path);
            
            if (prefab != null)
            {
                GameObject sword = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
                sword.name = swordNames[i];
                sword.transform.SetParent(parent.transform);
                
                // Add physics
                Rigidbody rb = sword.AddComponent<Rigidbody>();
                rb.mass = 1f;
                
                BoxCollider col = sword.AddComponent<BoxCollider>();
                col.size = new Vector3(0.4f, 2.5f, 0.1f);
                col.center = new Vector3(0, -0.8f, 0);
                
                count++;
                Debug.Log($"[AETHER] Added {swordNames[i]} with Rigidbody");
            }
            else
            {
                Debug.LogWarning($"[AETHER] Sword not found: {path}");
            }
        }
        
        // Create ground plane
        GameObject ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
        ground.name = "Ground";
        ground.transform.position = new Vector3(0, -3, 0);
        ground.transform.localScale = new Vector3(5, 1, 5);
        
        Debug.Log($"[AETHER] Sword Rain Setup Complete! {count} swords with Rigidbody. Press Play!");
        
        EditorUtility.DisplayDialog("AETHER Demo", 
            $"セットアップ完了！\n\n{count} 本の剣 + 地面を配置しました。\n\n再生ボタン(▶)を押してください！", 
            "OK");
    }
}
