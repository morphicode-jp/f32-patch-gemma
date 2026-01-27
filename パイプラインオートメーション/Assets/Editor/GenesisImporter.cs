using UnityEngine;
using UnityEditor;

// [cite: 10] AssetPostprocessorによるインポート自動化戦略
public class GenesisImporter : AssetPostprocessor
{
    void OnPreprocessModel()
    {
        ModelImporter importer = assetImporter as ModelImporter;
        if (importer == null) return;

        // Genesis Pipeline Standard: 座標系の不一致を解決
        // Blender(Z-up) -> Unity(Y-up) の回転をデータに焼き付ける
        importer.bakeAxisConversion = true;
        importer.globalScale = 1.0f;
        
        // オプション: マテリアル設定の自動化（エラー回避）
        importer.materialImportMode = ModelImporterMaterialImportMode.ImportStandard;
        
        Debug.Log($"[Genesis] Auto-configured import settings for: {assetPath}");
    }
}
