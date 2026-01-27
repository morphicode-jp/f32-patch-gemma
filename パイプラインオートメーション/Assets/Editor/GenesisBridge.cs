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
        // ログ監視のフック
        Application.logMessageReceived += (log, stack, type) => {
            if (type == LogType.Error || type == LogType.Warning) 
                errorLogBuffer.AppendLine($"[{type}] {log}");
        };
        
        // 遅延起動（Unityの初期化完了を待つ）
        EditorApplication.delayCall += () => {
            Debug.Log("[Genesis] Initializing Bridge (delayed)...");
            StartServerSafe();
        };
    }

    /// <summary>
    /// 安全なサーバー起動（try-catch付き）
    /// </summary>
    private static void StartServerSafe()
    {
        try
        {
            if (isRunning) return;
            
            // ポートが既に使用中かチェック
            try
            {
                listener = new TcpListener(IPAddress.Any, 8091);
                listener.Start();
            }
            catch (SocketException ex)
            {
                Debug.LogWarning($"[Genesis] Port 8091 already in use or unavailable: {ex.Message}");
                Debug.LogWarning("[Genesis] Bridge NOT started. Use menu: Genesis/Restart Bridge");
                return;
            }
            
            isRunning = true;
            Debug.Log("[Genesis] MCP Bridge Listening on 8091");
            
            // 非同期でクライアント受付開始（メインスレッドをブロックしない）
            AcceptClientsAsync();
        }
        catch (System.Exception ex)
        {
            Debug.LogError($"[Genesis] Failed to start server: {ex.Message}");
        }
    }
    
    /// <summary>
    /// 非同期でクライアントを受け付ける（ブロッキングなし）
    /// </summary>
    private static async void AcceptClientsAsync()
    {
        while (isRunning)
        {
            try
            {
                var client = await listener.AcceptTcpClientAsync();
                _ = HandleClient(client);
            }
            catch (System.ObjectDisposedException)
            {
                // リスナーが停止された
                break;
            }
            catch (System.Exception ex)
            {
                Debug.LogWarning($"[Genesis] Client accept error: {ex.Message}");
            }
        }
    }
    
    /// <summary>
    /// メニューからの手動再起動
    /// </summary>
    [MenuItem("Genesis/Restart Bridge")]
    public static void RestartBridge()
    {
        StopServer();
        StartServerSafe();
    }
    
    /// <summary>
    /// サーバーを停止
    /// </summary>
    [MenuItem("Genesis/Stop Bridge")]
    public static void StopServer()
    {
        isRunning = false;
        if (listener != null)
        {
            try
            {
                listener.Stop();
                listener = null;
                Debug.Log("[Genesis] Bridge stopped");
            }
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
                // メインスレッドで実行
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
        
        // For simple commands, respond immediately without main thread dispatch
        if (cmd == null || cmd.Length == 0) {
            tcs.SetResult("Empty Command");
            return tcs.Task;
        }
        
        // Ping/simple commands - immediate response
        if (!cmd.Contains("wait_for_import") && !cmd.Contains("get_errors") 
            && !cmd.Contains("refresh_assets") && !cmd.Contains("create_material")
            && !cmd.Contains("instantiate_prefab") && !cmd.Contains("add_component")
            && !cmd.Contains("set_rotation")) {
            tcs.SetResult("OK");
            return tcs.Task;
        }
        
        // For commands needing main thread, use update callback
        EditorApplication.CallbackFunction updateHandler = null;
        updateHandler = () => {
            try {
                EditorApplication.update -= updateHandler;
                
                if (cmd.Contains("wait_for_import")) {
                    string path = ExtractParam(cmd, "path");
                    string fullPath = Path.Combine(Application.dataPath.Replace("/Assets", ""), path);
                    string metaPath = fullPath + ".meta";
                    if (File.Exists(fullPath) && File.Exists(metaPath)) {
                        tcs.SetResult("Import Verified");
                    } else {
                        tcs.SetResult("Import Pending");
                    }
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
                    string result = CreateMaterialFromTexture(cmd);
                    tcs.SetResult(result);
                }
                else if (cmd.Contains("instantiate_prefab")) {
                    string result = InstantiatePrefab(cmd);
                    tcs.SetResult(result);
                }
                else if (cmd.Contains("add_component")) {
                    string result = AddComponent(cmd);
                    tcs.SetResult(result);
                }
                else if (cmd.Contains("set_rotation")) {
                    string result = SetRotation(cmd);
                    tcs.SetResult(result);
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

    /// <summary>
    /// テクスチャからマテリアルを生成し、対象モデルに割り当てる
    /// コマンド例: {"command":"create_material","texturePath":"Assets/Generated/SwordTexture.png","modelPath":"Assets/Generated/GenesisSword.fbx"}
    /// </summary>
    private static string CreateMaterialFromTexture(string cmd)
    {
        string texturePath = ExtractParam(cmd, "texturePath");
        string modelPath = ExtractParam(cmd, "modelPath");
        
        if (string.IsNullOrEmpty(texturePath) || string.IsNullOrEmpty(modelPath)) {
            return "Error: Missing texturePath or modelPath";
        }
        
        // 1. テクスチャをロード
        Texture2D texture = AssetDatabase.LoadAssetAtPath<Texture2D>(texturePath);
        if (texture == null) {
            return $"Error: Texture not found at {texturePath}";
        }
        
        // 2. マテリアルを作成
        string materialPath = texturePath.Replace(".png", "_Mat.mat").Replace(".jpg", "_Mat.mat");
        Material material = new Material(GetDefaultShader());
        material.mainTexture = texture;
        material.SetFloat("_Metallic", 0.5f);
        material.SetFloat("_Smoothness", 0.5f);
        
        // マテリアルをアセットとして保存
        AssetDatabase.CreateAsset(material, materialPath);
        Debug.Log($"[Genesis] Created material at: {materialPath}");
        
        // 3. モデルをロードしてマテリアルを割り当て
        GameObject modelPrefab = AssetDatabase.LoadAssetAtPath<GameObject>(modelPath);
        if (modelPrefab == null) {
            return $"Error: Model not found at {modelPath}";
        }
        
        // モデルのインポート設定を変更してマテリアルをリマップ
        ModelImporter importer = AssetImporter.GetAtPath(modelPath) as ModelImporter;
        if (importer != null) {
            // マテリアルリマッピング設定
            importer.materialImportMode = ModelImporterMaterialImportMode.ImportViaMaterialDescription;
            
            // 外部マテリアルを使用するように設定
            var externalObjects = importer.GetExternalObjectMap();
            
            // 既存のすべてのマテリアルスロットに新しいマテリアルを割り当て
            SerializedObject serializedImporter = new SerializedObject(importer);
            SerializedProperty materials = serializedImporter.FindProperty("m_Materials");
            
            if (materials != null && materials.isArray) {
                for (int i = 0; i < materials.arraySize; i++) {
                    SerializedProperty matProp = materials.GetArrayElementAtIndex(i);
                    SerializedProperty nameProp = matProp.FindPropertyRelative("name");
                    if (nameProp != null) {
                        string matName = nameProp.stringValue;
                        importer.AddRemap(new AssetImporter.SourceAssetIdentifier(typeof(Material), matName), material);
                    }
                }
            }
            
            serializedImporter.ApplyModifiedPropertiesWithoutUndo();
            importer.SaveAndReimport();
        }
        
        AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();
        
        return $"Material Created: {materialPath} -> Applied to {modelPath}";
    }
    
    /// <summary>
    /// FBXからPrefabを生成し、シーンにインスタンス化する
    /// コマンド例: {"command":"instantiate_prefab","modelPath":"Assets/Generated/GenesisSword.fbx","instanceName":"Sword_001","position":"0,0,0"}
    /// instanceNameを指定しない場合は自動でユニークなIDが付与される
    /// </summary>
    private static string InstantiatePrefab(string cmd)
    {
        string modelPath = ExtractParam(cmd, "modelPath");
        string instanceName = ExtractParam(cmd, "instanceName");  // NEW: カスタム名
        string positionStr = ExtractParam(cmd, "position") ?? "0,0,0";
        
        if (string.IsNullOrEmpty(modelPath)) {
            return "Error: Missing modelPath";
        }
        
        // 1. モデルをロード
        GameObject modelAsset = AssetDatabase.LoadAssetAtPath<GameObject>(modelPath);
        if (modelAsset == null) {
            return $"Error: Model not found at {modelPath}";
        }
        
        // 2. Prefabパスを生成
        string prefabPath = modelPath.Replace(".fbx", ".prefab").Replace(".FBX", ".prefab");
        
        // 3. Prefabが存在しなければ作成
        GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath);
        if (prefab == null) {
            // FBXからPrefabを生成
            prefab = PrefabUtility.SaveAsPrefabAsset(modelAsset, prefabPath);
            Debug.Log($"[Genesis] Created Prefab: {prefabPath}");
        }
        
        // 4. 位置をパース
        Vector3 position = Vector3.zero;
        string[] posParts = positionStr.Split(',');
        if (posParts.Length >= 3) {
            float.TryParse(posParts[0], out position.x);
            float.TryParse(posParts[1], out position.y);
            float.TryParse(posParts[2], out position.z);
        }
        
        // 5. シーンにインスタンス化
        GameObject instance = PrefabUtility.InstantiatePrefab(prefab) as GameObject;
        if (instance != null) {
            instance.transform.position = position;
            
            // 6. 名前を設定（重要！一意な識別のため）
            if (!string.IsNullOrEmpty(instanceName)) {
                // カスタム名が指定された場合はそれを使用
                instance.name = instanceName;
            } else {
                // 指定がない場合は自動でユニークなサフィックスを付与
                string baseName = Path.GetFileNameWithoutExtension(modelPath);
                string uniqueId = System.Guid.NewGuid().ToString().Substring(0, 6);
                instance.name = $"{baseName}_{uniqueId}";
            }
            
            Selection.activeGameObject = instance;
            Debug.Log($"[Genesis] Instantiated: {instance.name} at {position}");
            
            return $"Prefab Instantiated: {instance.name} at ({position.x},{position.y},{position.z})";
        }
        
        return "Error: Failed to instantiate prefab";
    }
    
    /// <summary>
    /// GameObjectにコンポーネントを追加する
    /// コマンド例: {"command":"add_component","objectName":"GenesisSword","componentType":"BoxCollider"}
    /// サポート: Rigidbody, BoxCollider, SphereCollider, MeshCollider, CapsuleCollider
    /// </summary>
    private static string AddComponent(string cmd)
    {
        string objectName = ExtractParam(cmd, "objectName");
        string componentType = ExtractParam(cmd, "componentType");
        
        if (string.IsNullOrEmpty(objectName) || string.IsNullOrEmpty(componentType)) {
            return "Error: Missing objectName or componentType";
        }
        
        // シーン内のオブジェクトを検索
        GameObject targetObject = GameObject.Find(objectName);
        if (targetObject == null) {
            return $"Error: Object '{objectName}' not found in scene";
        }
        
        // コンポーネントを追加
        Component addedComponent = null;
        switch (componentType.ToLower()) {
            case "rigidbody":
                addedComponent = targetObject.AddComponent<Rigidbody>();
                break;
            case "boxcollider":
                addedComponent = targetObject.AddComponent<BoxCollider>();
                break;
            case "spherecollider":
                addedComponent = targetObject.AddComponent<SphereCollider>();
                break;
            case "capsulecollider":
                addedComponent = targetObject.AddComponent<CapsuleCollider>();
                break;
            case "meshcollider":
                var meshCol = targetObject.AddComponent<MeshCollider>();
                meshCol.convex = true; // Rigidbody互換性のため
                addedComponent = meshCol;
                break;
            default:
                // 汎用コンポーネント追加（リフレクション）
                System.Type type = System.Type.GetType($"UnityEngine.{componentType}, UnityEngine");
                if (type != null) {
                    addedComponent = targetObject.AddComponent(type);
                } else {
                    return $"Error: Unknown component type '{componentType}'";
                }
                break;
        }
        
        if (addedComponent != null) {
            Debug.Log($"[Genesis] Added {componentType} to {objectName}");
            return $"Component Added: {componentType} -> {objectName}";
        }
        
        return $"Error: Failed to add {componentType}";
    }

    /// <summary>
    /// GameObjectの回転を設定する
    /// コマンド例: {"command":"set_rotation","objectName":"Sword_001","rotation":"180,0,0"}
    /// </summary>
    private static string SetRotation(string cmd)
    {
        string objectName = ExtractParam(cmd, "objectName");
        string rotationStr = ExtractParam(cmd, "rotation") ?? "0,0,0";
        
        if (string.IsNullOrEmpty(objectName)) {
            return "Error: Missing objectName";
        }
        
        // シーン内のオブジェクトを検索
        GameObject targetObject = GameObject.Find(objectName);
        if (targetObject == null) {
            return $"Error: Object '{objectName}' not found in scene";
        }
        
        // 回転をパース
        Vector3 rotation = Vector3.zero;
        string[] rotParts = rotationStr.Split(',');
        if (rotParts.Length >= 3) {
            float.TryParse(rotParts[0], out rotation.x);
            float.TryParse(rotParts[1], out rotation.y);
            float.TryParse(rotParts[2], out rotation.z);
        }
        
        targetObject.transform.rotation = Quaternion.Euler(rotation);
        Debug.Log($"[Genesis] Rotated {objectName} to ({rotation.x},{rotation.y},{rotation.z})");
        
        return $"Rotation Set: {objectName} -> ({rotation.x},{rotation.y},{rotation.z})";
    }

    /// <summary>
    /// プロジェクトのレンダーパイプラインに適したデフォルトシェーダーを取得
    /// </summary>
    private static Shader GetDefaultShader()
    {
        // URPを優先的に検索
        Shader urpLit = Shader.Find("Universal Render Pipeline/Lit");
        if (urpLit != null) return urpLit;
        
        // Standardにフォールバック
        Shader standard = Shader.Find("Standard");
        if (standard != null) return standard;
        
        // 最終手段
        return Shader.Find("Diffuse");
    }
    
    /// <summary>
    /// JSONコマンドからパラメータを抽出
    /// </summary>
    private static string ExtractParam(string json, string paramName)
    {
        // Simple regex to extract "paramName":"value"
        var match = Regex.Match(json, $"\"{paramName}\"\\s*:\\s*\"([^\"]+)\"");
        return match.Success ? match.Groups[1].Value : null;
    }

    private static async Task WaitForImport(string path)
    {
        int timeout = 10000;
        while (timeout > 0) {
            if (File.Exists(path) && File.Exists(path + ".meta")) return;
            await Task.Delay(500);
            timeout -= 500;
        }
        throw new System.Exception("Import Timeout");
    }

    private static string ExtractPath(string json) {
        int start = json.IndexOf("Assets/");
        if (start == -1) return "Assets/Generated/default.fbx";
        int end = json.IndexOf("\"", start);
        return json.Substring(start, end - start);
    }
}
