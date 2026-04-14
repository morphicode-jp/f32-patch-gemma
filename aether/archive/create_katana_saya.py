"""
Dragon Katana Scabbard (鞘/Saya) Generator
Creates a scabbard that fits the Dragon Katana v6 (72cm blade)

Specifications based on research:
- Length: Blade length + 5cm (72cm + 5cm = 77cm)
- Koiguchi (鯉口): Opening at the mouth
- Kurikata (栗形): Cord loop on the side
- Kojiri (鐺): End cap
"""

import bpy
import bmesh
import math
import os

# === SHARED BOUNDARY VARIABLES (SBV) ===
BLADE_LENGTH = 0.72  # Match katana v6 blade
SAYA_LENGTH = BLADE_LENGTH + 0.05  # 77cm total
SAYA_WIDTH = 0.04  # 4cm width
SAYA_THICKNESS = 0.018  # 1.8cm thickness

KOIGUCHI_HEIGHT = 0.025  # Mouth piece
KOJIRI_LENGTH = 0.02  # End cap

OUTPUT_DIR = r"C:\Users\mukic\OneDrive\デスクトップ\パイプラインオートメーション\パイプラインオートメーション\Assets\Generated"
PREVIEW_PATH = r"C:\Users\mukic\OneDrive\デスクトップ\パイプラインオートメーション\saya_preview.png"

# === CLEANUP ===
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

for mesh in bpy.data.meshes:
    bpy.data.meshes.remove(mesh)
for mat in bpy.data.materials:
    bpy.data.materials.remove(mat)

print("=== Creating Dragon Katana Scabbard (Saya) ===")

# === MATERIALS ===
def create_mat(name, color, metal, rough):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    output = nodes.new(type='ShaderNodeOutputMaterial')
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    bsdf.inputs['Base Color'].default_value = color
    bsdf.inputs['Metallic'].default_value = metal
    bsdf.inputs['Roughness'].default_value = rough
    return mat

# Black lacquer finish (拵鞘)
lacquer_mat = create_mat("BlackLacquer", (0.02, 0.02, 0.03, 1.0), 0.0, 0.2)
# Bronze fittings
bronze_mat = create_mat("Bronze", (0.55, 0.35, 0.15, 1.0), 0.9, 0.4)
# Horn (koiguchi/kojiri)
horn_mat = create_mat("Horn", (0.1, 0.08, 0.06, 1.0), 0.0, 0.5)

# === MAIN SCABBARD BODY ===
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, SAYA_LENGTH / 2))
saya_body = bpy.context.active_object
saya_body.name = "Saya_Body"

# Scale to scabbard shape (slightly tapered would be ideal, but keeping simple)
saya_body.scale = (SAYA_THICKNESS, SAYA_WIDTH, SAYA_LENGTH)
bpy.ops.object.transform_apply(scale=True)

# Apply lacquer material
saya_body.data.materials.append(lacquer_mat)

# === KOIGUCHI (鯉口) - Mouth piece ===
# Position at the top (where blade enters)
koiguchi_z = SAYA_LENGTH + KOIGUCHI_HEIGHT / 2
bpy.ops.mesh.primitive_cylinder_add(
    radius=max(SAYA_THICKNESS, SAYA_WIDTH) * 0.6,
    depth=KOIGUCHI_HEIGHT,
    location=(0, 0, koiguchi_z)
)
koiguchi = bpy.context.active_object
koiguchi.name = "Koiguchi"
koiguchi.scale = (0.8, 1.2, 1.0)  # Oval shape
bpy.ops.object.transform_apply(scale=True)
koiguchi.data.materials.append(horn_mat)

# === KURIKATA (栗形) - Cord loop ===
# Position on the side, about 1/4 from top
kurikata_z = SAYA_LENGTH * 0.75
bpy.ops.mesh.primitive_cube_add(size=1, location=(SAYA_THICKNESS / 2 + 0.008, 0, kurikata_z))
kurikata = bpy.context.active_object
kurikata.name = "Kurikata"
kurikata.scale = (0.012, 0.025, 0.015)
bpy.ops.object.transform_apply(scale=True)
kurikata.data.materials.append(bronze_mat)

# === KOJIRI (鐺) - End cap ===
bpy.ops.mesh.primitive_uv_sphere_add(
    radius=max(SAYA_THICKNESS, SAYA_WIDTH) * 0.4,
    location=(0, 0, 0)
)
kojiri = bpy.context.active_object
kojiri.name = "Kojiri"
kojiri.scale = (0.8, 1.0, 0.5)  # Flattened oval
bpy.ops.object.transform_apply(scale=True)
kojiri.data.materials.append(horn_mat)

# === JOIN ALL PARTS ===
bpy.ops.object.select_all(action='DESELECT')
saya_body.select_set(True)
koiguchi.select_set(True)
kurikata.select_set(True)
kojiri.select_set(True)
bpy.context.view_layer.objects.active = saya_body
bpy.ops.object.join()

# Finalize
bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# UV unwrap
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.smart_project()
bpy.ops.object.mode_set(mode='OBJECT')

saya = bpy.context.active_object
saya.name = "DragonKatana_Saya"

print(f"✅ Saya created: {SAYA_LENGTH * 100:.1f}cm length")

# === CAMERA (適切な配置) ===
# オブジェクトサイズに基づいてカメラ距離を計算
obj_dims = saya.dimensions
max_dim = max(obj_dims)
cam_distance = max_dim * 1.2  # 全体が見える距離

bpy.ops.object.camera_add(location=(cam_distance * 0.8, -cam_distance * 0.6, cam_distance * 0.4))
cam = bpy.context.active_object
cam.rotation_euler = (math.radians(70), 0, math.radians(50))
cam.data.lens = 35  # 広めに
bpy.context.scene.camera = cam

# === LIGHTING (明るく！) ===
# メインライト
bpy.ops.object.light_add(type='SUN', location=(5, -5, 10))
sun = bpy.context.active_object
sun.data.energy = 7.0  # 強め
sun.rotation_euler = (math.radians(45), 0, math.radians(30))

# フィルライト
bpy.ops.object.light_add(type='AREA', location=(-3, 3, 5))
fill = bpy.context.active_object
fill.data.energy = 300

# 背景 (明るめのグレー)
world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
bpy.context.scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs['Color'].default_value = (0.3, 0.3, 0.35, 1.0)

# === RENDER (縦長) ===
bpy.context.scene.render.resolution_x = 600
bpy.context.scene.render.resolution_y = 1000  # 縦長
bpy.context.scene.render.filepath = PREVIEW_PATH
bpy.ops.render.render(write_still=True)
print(f"📷 Preview: {PREVIEW_PATH}")

# === EXPORT FBX ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
fbx_path = os.path.join(OUTPUT_DIR, "DragonKatana_Saya.fbx")

bpy.ops.object.select_all(action='DESELECT')
saya.select_set(True)
bpy.context.view_layer.objects.active = saya

bpy.ops.export_scene.fbx(
    filepath=fbx_path,
    use_selection=True,
    axis_forward='-Z',
    axis_up='Y',
    bake_space_transform=True
)
print(f"✅ Exported: {fbx_path}")
print("=== DONE ===")
