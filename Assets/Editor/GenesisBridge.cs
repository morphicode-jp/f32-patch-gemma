using UnityEngine;
using UnityEditor;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using System.IO;

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
        Debug.Log("[Genesis] Initializing Bridge...");
        StartServer();
    }

    private static async void StartServer()
    {
        if (isRunning) return;
        listener = new TcpListener(IPAddress.Any, 8091);
        listener.Start();
        isRunning = true;
        Debug.Log("[Genesis] MCP Bridge Listening on 8091");

        while (isRunning)
        {
            var client = await listener.AcceptTcpClientAsync();
            _ = HandleClient(client);
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
        EditorApplication.delayCall += async () => {
            try {
                // ハンドシェイク・プロトコル
                if (cmd.Contains("wait_for_import")) {
                    string path = ExtractPath(cmd); // 簡易抽出
                    await WaitForImport(path);
                    tcs.SetResult("Import Verified");
                }
                // エラーログ取得（自己修復用）
                else if (cmd.Contains("get_errors")) {
                    string logs = errorLogBuffer.ToString();
                    errorLogBuffer.Clear();
                    tcs.SetResult(string.IsNullOrEmpty(logs) ? "Clean" : logs);
                }
                else {
                    tcs.SetResult("OK");
                }
            } catch (System.Exception e) {
                tcs.SetResult("Error: " + e.Message);
            }
        };
        return tcs.Task;
    }

    private static async Task WaitForImport(string path)
    {
        int timeout = 10000;
        while (timeout > 0) {
            // .metaファイルの生成を確認
            if (File.Exists(path) && File.Exists(path + ".meta")) return;
            await Task.Delay(500);
            timeout -= 500;
        }
        throw new System.Exception("Import Timeout");
    }

    private static string ExtractPath(string json) {
        // ※実際はJSONパーサーを使用すること
        int start = json.IndexOf("Assets/");
        if (start == -1) return "Assets/Generated/default.fbx";
        int end = json.IndexOf("\"", start);
        return json.Substring(start, end - start);
    }
}
