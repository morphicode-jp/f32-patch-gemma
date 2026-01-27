#!/usr/bin/env node
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";
import * as net from "net";
import { exec } from "child_process";
import { promisify } from "util";

const execAsync = promisify(exec);

const UNITY_HOST = process.env.UNITY_HOST || "127.0.0.1";
const UNITY_PORT = parseInt(process.env.UNITY_PORT || "8091");
const PROJECT_ROOT = process.env.PROJECT_ROOT || process.cwd();

const server = new Server(
    { name: "unity-genesis-bridge", version: "1.1.0" },
    { capabilities: { tools: {} } }
);

// UnityへTCPコマンド送信
function sendToUnity(command: object): Promise<string> {
    return new Promise((resolve, reject) => {
        const client = new net.Socket();
        client.connect(UNITY_PORT, UNITY_HOST, () => {
            client.write(JSON.stringify(command) + "\n");
        });
        client.on("data", (data) => {
            resolve(data.toString().trim());
            client.destroy();
        });
        client.on("error", (err) => reject(err));
    });
}

// Git操作: コミット
async function gitCommit(message: string): Promise<string> {
    try {
        // ステージング
        await execAsync("git add .", { cwd: PROJECT_ROOT });

        // 変更があるか確認
        const { stdout: status } = await execAsync("git status --porcelain", { cwd: PROJECT_ROOT });

        if (!status.trim()) {
            return "No changes to commit";
        }

        // コミット実行
        const commitMessage = `[Genesis Auto] ${message}`;
        const { stdout } = await execAsync(`git commit -m "${commitMessage}"`, { cwd: PROJECT_ROOT });

        return `Committed: ${commitMessage}\n${stdout}`;
    } catch (error: any) {
        // "nothing to commit" はエラーではない
        if (error.message.includes("nothing to commit")) {
            return "No changes to commit";
        }
        throw error;
    }
}

// Git操作: Diff確認（自己診断用）
async function gitDiff(): Promise<string> {
    try {
        const { stdout } = await execAsync("git diff --stat", { cwd: PROJECT_ROOT });
        if (!stdout.trim()) {
            return "No uncommitted changes";
        }
        // 詳細なdiffも取得（最初の500文字まで）
        const { stdout: detailed } = await execAsync("git diff", { cwd: PROJECT_ROOT });
        const truncated = detailed.length > 500 ? detailed.substring(0, 500) + "\n... (truncated)" : detailed;
        return `Changes Summary:\n${stdout}\n\nDetails:\n${truncated}`;
    } catch (error: any) {
        return `Diff Error: ${error.message}`;
    }
}

// Git操作: ステータス確認
async function gitStatus(): Promise<string> {
    try {
        const { stdout } = await execAsync("git status --short", { cwd: PROJECT_ROOT });
        if (!stdout.trim()) {
            return "Working tree clean - no changes";
        }
        return `Modified files:\n${stdout}`;
    } catch (error: any) {
        return `Status Error: ${error.message}`;
    }
}

// ツール定義
server.setRequestHandler(ListToolsRequestSchema, async () => {
    return {
        tools: [
            {
                name: "wait_for_import",
                description: "Waits until the specified asset is fully imported into Unity (prevents race conditions).",
                inputSchema: {
                    type: "object",
                    properties: {
                        path: { type: "string", description: "Relative path in Unity (e.g., Assets/Generated/model.fbx)" }
                    },
                    required: ["path"]
                }
            },
            {
                name: "get_console_errors",
                description: "Retrieves recent error logs from Unity console for self-healing.",
                inputSchema: { type: "object", properties: {} }
            },
            {
                name: "git_commit",
                description: "Stages all changes and commits with a timestamped message. Acts as an automatic save point (Time Machine).",
                inputSchema: {
                    type: "object",
                    properties: {
                        message: { type: "string", description: "Commit message describing the changes (e.g., 'Created GenesisSword.fbx')" }
                    },
                    required: ["message"]
                }
            },
            {
                name: "git_diff",
                description: "Shows uncommitted changes in the working directory. Use this for self-verification after making changes - helps catch accidental deletions or modifications.",
                inputSchema: { type: "object", properties: {} }
            },
            {
                name: "git_status",
                description: "Shows which files have been modified, added, or deleted. Quick overview before committing.",
                inputSchema: { type: "object", properties: {} }
            }
        ]
    };
});

// ツール実行
server.setRequestHandler(CallToolRequestSchema, async (request) => {
    try {
        // Git操作はUnityを経由しない
        if (request.params.name === "git_commit") {
            const args = request.params.arguments as { message: string };
            const result = await gitCommit(args.message);
            return { content: [{ type: "text", text: result }] };
        }
        if (request.params.name === "git_diff") {
            const result = await gitDiff();
            return { content: [{ type: "text", text: result }] };
        }
        if (request.params.name === "git_status") {
            const result = await gitStatus();
            return { content: [{ type: "text", text: result }] };
        }

        // Unity系コマンド
        const response = await sendToUnity({
            command: request.params.name,
            ...request.params.arguments
        });
        return { content: [{ type: "text", text: response }] };
    } catch (error) {
        return { content: [{ type: "text", text: `Error: ${error}` }], isError: true };
    }
});

const transport = new StdioServerTransport();
await server.connect(transport);
