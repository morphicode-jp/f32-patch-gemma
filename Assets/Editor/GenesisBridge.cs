using UnityEngine;
using UnityEditor;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Text.RegularExpressions;

[InitializeOnLoad]
public class GenesisBridge
{
    private static TcpListener listener;
    private static bool isRunning = false;
    private static StringBuilder errorLogBuffer = new StringBuilder();

    static GenesisBridge()
    {
        Application.logMessageReceived += (log, stack, type) => {
            if (type == LogType.Error || type == LogType.Warning) 
                errorLogBuffer.AppendLine($"[{type}] {log}");
        };
        
        EditorApplication.delayCall += () => {
            Debug.Log("[Genesis] Initializing Bridge (delayed)...");
            StartServerSafe();
        };
    }

    private static void StartServerSafe()
    {
        try
        {
            if (isRunning) return;
            
            try
            {
                listener = new TcpListener(IPAddress.Any, 8091);
                listener.Start();
            }
            catch (SocketException ex)
            {
                Debug.LogWarning($"[Genesis] Port 8091 already in use: {ex.Message}");
                return;
            }
            
            isRunning = true;
            Debug.Log("[Genesis] MCP Bridge Listening on 8091");
            AcceptClientsAsync();
        }
        catch (System.Exception ex)
        {
            Debug.LogError($"[Genesis] Failed to start server: {ex.Message}");
        }
    }
    
    private static async void AcceptClientsAsync()
    {
        while (isRunning)
        {
            try
            {
                var client = await listener.AcceptTcpClientAsync();
                _ = HandleClient(client);
            }
            catch (System.ObjectDisposedException) { break; }
            catch (System.Exception ex)
            {
                Debug.LogWarning($"[Genesis] Client accept error: {ex.Message}");
            }
        }
    }
    
    [MenuItem("Genesis/Restart Bridge")]
    public static void RestartBridge()
    {
        StopServer();
        StartServerSafe();
    }
    
    [MenuItem("Genesis/Stop Bridge")]
    public static void StopServer()
    {
        isRunning = false;
        if (listener != null)
        {
            try { listener.Stop(); listener = null; }
            catch { }
        }
    }

    private static async Task HandleClient(TcpClient client)
    {
        using (var stream = client.GetStream())
        using (var reader = new StreamReader(stream, Encoding.UTF8))
        using (var writer = new StreamWriter(stream, Encoding.UTF8) { AutoFlush = true })
        {
            try {
                string jsonCmd = await reader.ReadLineAsync();
                string response = await ExecuteOnMainThread(jsonCmd);
                await writer.WriteLineAsync(response);
            } catch (System.Exception e) {
                Debug.LogError(e);
            }
        }
        client.Close();
    }

    private static Task<string> ExecuteOnMainThread(string cmd)
    {
        var tcs = new TaskCompletionSource<string>();
        
        if (cmd == null || cmd.Length == 0) {
            tcs.SetResult("Empty Command");
            return tcs.Task;
        }

        EditorApplication.CallbackFunction updateHandler = null;
        updateHandler = () => {
            try {
                EditorApplication.update -= updateHandler;
                
                if (cmd.Contains("wait_for_import")) {
                    string path = ExtractParam(cmd, "path");
                    string fullPath = Path.Combine(Application.dataPath.Replace("/Assets", ""), path);
                    string metaPath = fullPath + ".meta";
                    tcs.SetResult(File.Exists(fullPath) && File.Exists(metaPath) ? "Import Verified" : "Import Pending");
                }
                else if (cmd.Contains("get_errors")) {
                    string logs = errorLogBuffer.ToString();
                    errorLogBuffer.Clear();
                    tcs.SetResult(string.IsNullOrEmpty(logs) ? "Clean" : logs);
                }
                else if (cmd.Contains("refresh_assets")) {
                    AssetDatabase.Refresh();
                    tcs.SetResult("Refreshed");
                }
                else if (cmd.Contains("create_material")) {
                    tcs.SetResult(CreateMaterialFromTexture(cmd));
                }
                else if (cmd.Contains("instantiate_prefab")) {
                    tcs.SetResult(InstantiatePrefab(cmd));
                }
                else if (cmd.Contains("add_component")) {
                    tcs.SetResult(AddComponent(cmd));
                }
                else if (cmd.Contains("set_rotation")) {
                    tcs.SetResult(SetRotation(cmd));
                }
                else if (cmd.Contains("create_primitive")) {
                    tcs.SetResult(CreatePrimitive(cmd));
                }
                else if (cmd.Contains("delete_object")) {
                    tcs.SetResult(DeleteObject(cmd));
                }
                else {
                    tcs.SetResult("OK");
                }
            } catch (System.Exception e) {
                tcs.SetResult("Error: " + e.Message);
            }
        };
        EditorApplication.update += updateHandler;
        
        return tcs.Task;
    }

    private static string CreateMaterialFromTexture(string cmd)
    {
        string texPath = ExtractParam(cmd, "texturePath");
        string modelPath = ExtractParam(cmd, "modelPath");
        if (string.IsNullOrEmpty(texPath) || string.IsNullOrEmpty(modelPath)) return "Error: Missing params";
        
        Texture2D tex = AssetDatabase.LoadAssetAtPath<Texture2D>(texPath);
        if (tex == null) return $"Error: Texture not found at {texPath}";
        
        string matPath = texPath.Replace(".png", "_Mat.mat").Replace(".jpg", "_Mat.mat");
        Material mat = new Material(Shader.Find("Standard"));
        mat.mainTexture = tex;
        AssetDatabase.CreateAsset(mat, matPath);
        AssetDatabase.SaveAssets();
        
        return $"Material Created: {matPath}";
    }
    
    private static string InstantiatePrefab(string cmd)
    {
        string modelPath = ExtractParam(cmd, "modelPath");
        string instanceName = ExtractParam(cmd, "instanceName");
        string positionStr = ExtractParam(cmd, "position") ?? "0,0,0";
        
        if (string.IsNullOrEmpty(modelPath)) return "Error: Missing modelPath";
        
        GameObject modelAsset = AssetDatabase.LoadAssetAtPath<GameObject>(modelPath);
        if (modelAsset == null) return $"Error: Model not found at {modelPath}";
        
        string prefabPath = modelPath.Replace(".fbx", ".prefab").Replace(".FBX", ".prefab");
        GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath);
        if (prefab == null) {
            prefab = PrefabUtility.SaveAsPrefabAsset(modelAsset, prefabPath);
        }
        
        Vector3 position = ParseVector3(positionStr);
        GameObject instance = PrefabUtility.InstantiatePrefab(prefab) as GameObject;
        if (instance != null) {
            instance.transform.position = position;
            instance.name = string.IsNullOrEmpty(instanceName) ? 
                $"{Path.GetFileNameWithoutExtension(modelPath)}_{System.Guid.NewGuid().ToString().Substring(0, 6)}" : 
                instanceName;
            Selection.activeGameObject = instance;
            return $"Prefab Instantiated: {instance.name} at ({position.x},{position.y},{position.z})";
        }
        return "Error: Failed to instantiate prefab";
    }
    
    private static string AddComponent(string cmd)
    {
        string objectName = ExtractParam(cmd, "objectName");
        string componentType = ExtractParam(cmd, "componentType");
        if (string.IsNullOrEmpty(objectName) || string.IsNullOrEmpty(componentType)) return "Error: Missing params";
        
        GameObject obj = GameObject.Find(objectName);
        if (obj == null) return $"Error: Object '{objectName}' not found";
        
        Component added = null;
        switch (componentType.ToLower()) {
            case "rigidbody": added = obj.AddComponent<Rigidbody>(); break;
            case "boxcollider": added = obj.AddComponent<BoxCollider>(); break;
            case "spherecollider": added = obj.AddComponent<SphereCollider>(); break;
            case "capsulecollider": added = obj.AddComponent<CapsuleCollider>(); break;
            case "meshcollider": 
                var mc = obj.AddComponent<MeshCollider>(); 
                mc.convex = true; 
                added = mc; 
                break;
            default:
                var type = System.Type.GetType($"UnityEngine.{componentType}, UnityEngine");
                if (type != null) added = obj.AddComponent(type);
                else return $"Error: Unknown component '{componentType}'";
                break;
        }
        return added != null ? $"Component Added: {componentType} -> {objectName}" : "Error: Failed to add component";
    }

    private static string SetRotation(string cmd)
    {
        string objectName = ExtractParam(cmd, "objectName");
        string rotationStr = ExtractParam(cmd, "rotation") ?? "0,0,0";
        if (string.IsNullOrEmpty(objectName)) return "Error: Missing objectName";
        
        GameObject obj = GameObject.Find(objectName);
        if (obj == null) return $"Error: Object '{objectName}' not found";
        
        obj.transform.rotation = Quaternion.Euler(ParseVector3(rotationStr));
        return $"Rotation Set: {objectName} -> ({rotationStr})";
    }

    private static string CreatePrimitive(string cmd)
    {
        string primitiveType = ExtractParam(cmd, "primitiveType");
        string objectName = ExtractParam(cmd, "objectName");
        string positionStr = ExtractParam(cmd, "position") ?? "0,0,0";
        string scaleStr = ExtractParam(cmd, "scale") ?? "1,1,1";

        PrimitiveType type;
        switch (primitiveType?.ToLower()) {
            case "cube": type = PrimitiveType.Cube; break;
            case "sphere": type = PrimitiveType.Sphere; break;
            case "capsule": type = PrimitiveType.Capsule; break;
            case "cylinder": type = PrimitiveType.Cylinder; break;
            case "plane": type = PrimitiveType.Plane; break;
            case "quad": type = PrimitiveType.Quad; break;
            default: return $"Error: Unknown primitive type '{primitiveType}'";
        }

        GameObject obj = GameObject.CreatePrimitive(type);
        if (!string.IsNullOrEmpty(objectName)) obj.name = objectName;
        obj.transform.position = ParseVector3(positionStr);
        obj.transform.localScale = ParseVector3(scaleStr);

        return $"Created Primitive: {obj.name}";
    }

    private static string DeleteObject(string cmd)
    {
        string objectName = ExtractParam(cmd, "objectName");
        if (string.IsNullOrEmpty(objectName)) return "Error: Missing objectName";

        GameObject obj = GameObject.Find(objectName);
        if (obj == null) return $"Error: Object '{objectName}' not found";

        GameObject.DestroyImmediate(obj);
        return $"Deleted: {objectName}";
    }

    private static Vector3 ParseVector3(string str)
    {
        Vector3 v = Vector3.zero;
        string[] parts = str.Split(',');
        if (parts.Length >= 3) {
            float.TryParse(parts[0], out v.x);
            float.TryParse(parts[1], out v.y);
            float.TryParse(parts[2], out v.z);
        }
        return v;
    }

    private static string ExtractParam(string json, string paramName)
    {
        var match = Regex.Match(json, $"\"{paramName}\"\\s*:\\s*\"([^\"]+)\"");
        return match.Success ? match.Groups[1].Value : null;
    }
}
