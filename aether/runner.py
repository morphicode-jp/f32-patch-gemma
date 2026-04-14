"""
AETHER Runner - AI出力JSONを処理してBlenderで実行
CREATIONモードの場合、new_library_codeをaether/parts/custom.pyに保存

🛡️ 安全装置:
1. 文法チェック (ast.parse)
2. 重複チェック (クラス名/関数名)
3. テキスト解析によるパーツ一覧取得（キャッシュ問題回避）
"""

import json
import os
import subprocess
import sys
import tempfile
import ast
import re

# === パス設定 ===
AETHER_ROOT = r"C:\Users\mukic\OneDrive\デスクトップ\パイプラインオートメーション"
CUSTOM_PARTS_PATH = os.path.join(AETHER_ROOT, "aether", "parts", "custom.py")
BLENDER_PATH = r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"


def process_ai_response(ai_json: dict) -> str:
    """
    AI出力JSONを処理
    
    Args:
        ai_json: AIが出力したJSON（mode, thought, new_library_code, execution_code）
    
    Returns:
        実行結果メッセージ
    """
    mode = ai_json.get("mode", "ASSEMBLY")
    thought = ai_json.get("thought", "")
    new_library_code = ai_json.get("new_library_code", "")
    execution_code = ai_json.get("execution_code", "")
    
    print(f"=== AETHER Runner ===")
    print(f"Mode: {mode}")
    print(f"Thought: {thought}")
    
    # === CREATION モード: 新規ライブラリコードを保存 ===
    if mode == "CREATION" and new_library_code.strip():
        success = save_library_code(new_library_code)
        if not success:
            return "❌ Library code save failed. Aborting execution."
    
    # === 実行コードをBlenderで実行 ===
    if execution_code.strip():
        result = execute_in_blender(execution_code)
        return result
    else:
        return "No execution code provided."


def save_library_code(code: str) -> bool:
    """
    新規ライブラリコードをcustom.pyに追記保存
    
    🛡️ 安全装置:
    1. 文法チェック（壊れたコードは保存しない）
    2. 重複チェック（同じ定義は追記しない）
    
    Args:
        code: 保存するPythonコード
    
    Returns:
        成功: True, 失敗: False
    """
    
    # ===========================================
    # 🛡️ 安全装置1: 文法チェック (Syntax Check)
    # ===========================================
    try:
        ast.parse(code)
    except SyntaxError as e:
        print(f"❌ AI生成コードに文法エラーがあります。保存をスキップします。")
        print(f"   Error: {e}")
        return False
    
    print("✅ Syntax check passed")
    
    # ファイルが存在しない場合は初期化
    if not os.path.exists(CUSTOM_PARTS_PATH):
        with open(CUSTOM_PARTS_PATH, "w", encoding="utf-8") as f:
            f.write('"""\nAETHER Custom Parts - AI生成の新規パーツ\n"""\n\nimport bpy\nimport bmesh\nimport math\n\n')
        print(f"✅ Created: {CUSTOM_PARTS_PATH}")
    
    # ===========================================
    # 🛡️ 安全装置2: 重複チェック (Duplicate Check)
    # ===========================================
    with open(CUSTOM_PARTS_PATH, "r", encoding="utf-8") as f:
        existing_content = f.read()
    
    # クラス名・関数名を抽出
    new_definitions = re.findall(r'^(?:class|def)\s+(\w+)', code, re.MULTILINE)
    existing_definitions = re.findall(r'^(?:class|def)\s+(\w+)', existing_content, re.MULTILINE)
    
    duplicates = set(new_definitions) & set(existing_definitions)
    if duplicates:
        print(f"⚠️ Already exists: {duplicates} - Skipping save")
        return True  # 既に存在するのでOK扱い
    
    # 1行目チェック（class Trident: などの完全一致）
    first_line = code.strip().split('\n')[0]
    if first_line in existing_content:
        print(f"⚠️ '{first_line}' は既に存在するため、追記をスキップします。")
        return True
    
    # ===========================================
    # 追記保存
    # ===========================================
    with open(CUSTOM_PARTS_PATH, "a", encoding="utf-8") as f:
        f.write("\n\n# === AI Generated ===\n")
        f.write(code)
        f.write("\n")
    
    print(f"✅ Saved to library: {CUSTOM_PARTS_PATH}")
    print(f"   Added: {new_definitions}")
    return True


def execute_in_blender(code: str, rollback_on_error: bool = True) -> str:
    """
    BlenderでPythonコードを実行
    
    🛡️ Lv.1 エラーチェックによる自動破棄:
    実行時エラーが発生した場合、直前に追記したライブラリコードを自動削除
    
    Args:
        code: 実行するPythonコード
        rollback_on_error: エラー時にライブラリコードをロールバックするか
    
    Returns:
        実行結果
    """
    # ロールバック用に現在のcustom.pyの内容を保存
    backup_content = None
    if rollback_on_error and os.path.exists(CUSTOM_PARTS_PATH):
        with open(CUSTOM_PARTS_PATH, "r", encoding="utf-8") as f:
            backup_content = f.read()
    
    # sys.pathにAETHER_ROOTを追加するプリアンブル
    preamble = f'''
import sys
if r"{AETHER_ROOT}" not in sys.path:
    sys.path.insert(0, r"{AETHER_ROOT}")

'''
    
    full_code = preamble + code
    
    # 一時ファイルに保存
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(full_code)
        temp_path = f.name
    
    try:
        # Blender実行
        result = subprocess.run(
            [BLENDER_PATH, "--background", "--python", temp_path],
            capture_output=True,
            text=True,
            cwd=AETHER_ROOT
        )
        
        output = result.stdout + result.stderr
        
        # ===========================================
        # 🛡️ Lv.1 エラーチェックによる自動破棄
        # ===========================================
        error_indicators = [
            "Traceback (most recent call last):",
            "Error:",
            "SyntaxError:",
            "NameError:",
            "AttributeError:",
            "TypeError:",
            "ImportError:",
            "ModuleNotFoundError:",
        ]
        
        has_error = any(indicator in output for indicator in error_indicators)
        
        if result.returncode != 0 or has_error:
            print(f"❌ Blender execution failed (code: {result.returncode})")
            
            # エラー内容を抽出して表示
            error_lines = []
            in_traceback = False
            for line in output.split('\n'):
                if "Traceback" in line:
                    in_traceback = True
                if in_traceback:
                    error_lines.append(line)
                    if line.strip() and not line.startswith(" ") and "Error" in line:
                        break
            
            if error_lines:
                print("📋 Error details:")
                for line in error_lines[-5:]:  # 最後の5行だけ表示
                    print(f"   {line}")
            
            # 🛡️ ロールバック実行
            if rollback_on_error and backup_content is not None:
                with open(CUSTOM_PARTS_PATH, "w", encoding="utf-8") as f:
                    f.write(backup_content)
                print("🔄 Auto-rollback: Library code reverted to previous state")
        else:
            print("✅ Blender execution successful")
        
        return output
        
    finally:
        # 一時ファイル削除
        os.unlink(temp_path)


def get_available_parts() -> list:
    """
    現在使用可能なパーツ一覧を取得（テキスト解析版）
    
    🛡️ キャッシュ問題回避:
    ファイルを直接読んで解析するので、
    custom.pyに追記された直後でも最新リストが取得できる
    
    Returns:
        利用可能なクラス・関数のリスト
    """
    parts = []
    parts_dir = os.path.join(AETHER_ROOT, "aether", "parts")
    
    # 対象ファイル
    target_files = [
        os.path.join(parts_dir, "blades.py"),
        os.path.join(parts_dir, "custom.py"),
    ]
    
    for file_path in target_files:
        if not os.path.exists(file_path):
            continue
        
        module_name = os.path.basename(file_path).replace(".py", "")
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                original_line = line  # インデント確認用
                line_stripped = line.strip()
                
                # 🛡️ インデントされた行はスキップ（クラスメソッドは除外）
                if original_line.startswith("    ") or original_line.startswith("\t"):
                    continue
                
                # クラス定義: "class Katana:" → "aether.parts.blades.Katana"
                if line_stripped.startswith("class ") and ":" in line_stripped:
                    class_name = line_stripped.split("(")[0].split(":")[0].replace("class ", "").strip()
                    parts.append(f"aether.parts.{module_name}.{class_name}")
                
                # トップレベル関数のみ: "def create_sword():"
                # ※ "def create(self):" はクラスメソッドなので上で除外済み
                elif line_stripped.startswith("def ") and not line_stripped.startswith("def _"):
                    func_match = re.match(r'def\s+(\w+)\s*\(', line_stripped)
                    if func_match:
                        func_name = func_match.group(1)
                        parts.append(f"aether.parts.{module_name}.{func_name}()")
    
    return parts


def get_context_injection() -> str:
    """
    SKILL.md用のCONTEXT INJECTION文字列を生成
    """
    parts = get_available_parts()
    return "\n".join(f"- {p}" for p in parts)


# === CLI実行 ===
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # コマンドライン引数からJSON読み込み
        json_str = sys.argv[1]
        try:
            ai_json = json.loads(json_str)
            result = process_ai_response(ai_json)
            print(result)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON: {e}")
    else:
        # パーツ一覧を表示
        print("=== AETHER Available Parts ===")
        print(get_context_injection())
