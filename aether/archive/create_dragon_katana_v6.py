"""
AETHER: Authentic Katana v6
刃の形状をより明確に - 視覚的に分かりやすいバージョン
"""
import bpy
import bmesh
import os
import math

# === PATHS ===
output_dir = r"C:\Users\mukic\OneDrive\デスクトップ\パイプラインオートメーション\パイプラインオートメーション\Assets\Generated"
preview_path = r"C:\Users\mukic\OneDrive\デスクトップ\パイプラインオートメーション\sword_preview.png"
os.makedirs(output_dir, exist_ok=True)

# === CLEAR ===
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

print("=== Creating Authentic Katana v6 ===")

# ============================================
# 寸法（視覚的に分かりやすいスケール）
# 刀の典型的な比率を維持しつつ、幅を視認できるように調整
# ============================================
BLADE_LENGTH = 0.72      # 刃長: 72cm
BLADE_WIDTH = 0.06       # 身幅: 6cm（視覚用に強調）
BLADE_THICK = 0.012      # 厚み: 1.2cm
CURVE_DEPTH = 0.025      # 反り: 2.5cm

TSUBA_RADIUS = 0.055     # 鍔半径: 5.5cm
TSUBA_THICK = 0.008      # 鍔厚み

HANDLE_LENGTH = 0.26     # 柄長
HANDLE_RADIUS = 0.018    # 柄半径

# ============================================
# マテリアル
# ============================================
def make_mat(name, color, metallic=0.0, roughness=0.5):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    out = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    mat.node_tree.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    bsdf.inputs['Base Color'].default_value = color
    bsdf.inputs['Metallic'].default_value = metallic
    bsdf.inputs['Roughness'].default_value = roughness
    return mat

mat_blade = make_mat("Blade_Steel", (0.85, 0.87, 0.9, 1.0), 1.0, 0.1)
mat_tsuba = make_mat("Tsuba_Bronze", (0.12, 0.08, 0.04, 1.0), 0.9, 0.35)
mat_handle = make_mat("Handle_Cloth", (0.015, 0.015, 0.015, 1.0), 0.0, 0.95)

# ============================================
# 1. 刃（BLADE）- 日本刀らしい形状
# ============================================
mesh = bpy.data.meshes.new("Blade_Mesh")
bm = bmesh.new()

# 刃の断面を定義（日本刀は片刃で、背が厚く刃先が薄い）
# Z軸に沿って押し出す

segments = 32
for i in range(segments + 1):
    t = i / segments
    z = TSUBA_THICK + t * BLADE_LENGTH
    
    # 反り（中央が最大）
    sori = CURVE_DEPTH * math.sin(t * math.pi)
    
    # テーパー（先端に向かって細くなる）
    taper = 1.0 - t * 0.4  # 先端で60%の幅
    w = BLADE_WIDTH * taper
    th = BLADE_THICK * taper
    
    # 日本刀の断面（鎬造り風）
    #       背(Mune)
    #      /  |  \
    #     /   |   \  <- 鎬筋
    #    -----+-----  <- 平地
    #     \       /
    #      \     /   <- 刃先(Ha)
    #       \   /
    #        \_/
    
    # 5つの頂点で断面を作成
    verts = []
    
    # 1. 背（中央上）
    verts.append(bm.verts.new((0, sori + th/2, z)))
    
    # 2. 右鎬（右上斜め）
    verts.append(bm.verts.new((w * 0.3, sori + th/4, z)))
    
    # 3. 右平地（右端）
    verts.append(bm.verts.new((w * 0.5, sori, z)))
    
    # 4. 刃先（中央下）- 片刃なので右寄り
    verts.append(bm.verts.new((w * 0.3, sori - th/2, z)))
    
    # 5. 左平地（左端）- 片刃なので狭い
    verts.append(bm.verts.new((-w * 0.15, sori, z)))
    
    # 6. 左鎬（左上斜め）
    verts.append(bm.verts.new((-w * 0.1, sori + th/4, z)))
    
    # 面を作成（前のセグメントと接続）
    if i > 0:
        prev = bm.verts[-12:-6]  # 前の6頂点
        curr = bm.verts[-6:]      # 現在の6頂点
        
        for j in range(6):
            j2 = (j + 1) % 6
            try:
                bm.faces.new([prev[j], prev[j2], curr[j2], curr[j]])
            except:
                pass

# 切っ先を閉じる
tip_verts = bm.verts[-6:]
tip_center = bm.verts.new((0, CURVE_DEPTH * 0.5, TSUBA_THICK + BLADE_LENGTH + 0.03))
for j in range(6):
    j2 = (j + 1) % 6
    try:
        bm.faces.new([tip_verts[j], tip_verts[j2], tip_center])
    except:
        pass

# 根元を閉じる
base_verts = list(bm.verts)[:6]
try:
    bm.faces.new(base_verts[::-1])
except:
    pass

bm.to_mesh(mesh)
bm.free()

blade = bpy.data.objects.new("Blade", mesh)
bpy.context.collection.objects.link(blade)
blade.data.materials.append(mat_blade)

bpy.context.view_layer.objects.active = blade
blade.select_set(True)
bpy.ops.object.shade_smooth()

# ============================================
# 2. 鍔（TSUBA）- 龍形
# ============================================
bpy.ops.mesh.primitive_cylinder_add(
    radius=TSUBA_RADIUS,
    depth=TSUBA_THICK,
    location=(0, 0, TSUBA_THICK/2),
    vertices=48
)
tsuba = bpy.context.active_object
tsuba.name = "Tsuba"

bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(tsuba.data)

for v in bm.verts:
    r = math.sqrt(v.co.x**2 + v.co.y**2)
    if r < 0.01:
        continue
    ang = math.atan2(v.co.y, v.co.x)
    # 龍の形
    dragon = 1.0 + 0.2 * math.exp(-((ang - 0.2)**2) / 0.1)  # 頭
    dragon *= 1.0 + 0.15 * math.exp(-((ang - math.pi)**2) / 0.25)  # 尾
    dragon *= 1.0 + 0.06 * math.sin(ang * 5)  # うねり
    
    v.co.x *= dragon
    v.co.y *= dragon

bmesh.update_edit_mesh(tsuba.data)
bpy.ops.object.mode_set(mode='OBJECT')

tsuba.data.materials.append(mat_tsuba)
bpy.ops.object.shade_smooth()

# ============================================
# 3. 柄（HANDLE）
# ============================================
bpy.ops.mesh.primitive_cylinder_add(
    radius=HANDLE_RADIUS,
    depth=HANDLE_LENGTH,
    location=(0, 0, -HANDLE_LENGTH/2),
    vertices=16
)
handle = bpy.context.active_object
handle.name = "Handle"
handle.data.materials.append(mat_handle)
bpy.ops.object.shade_smooth()

# ============================================
# 結合
# ============================================
bpy.ops.object.select_all(action='DESELECT')
blade.select_set(True)
tsuba.select_set(True)
handle.select_set(True)
bpy.context.view_layer.objects.active = blade
bpy.ops.object.join()

katana = bpy.context.active_object
katana.name = "DragonKatana_v6"
bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')

print(f"✅ Katana v6 created: {katana.name}")
print(f"   Blade: {BLADE_LENGTH*100:.0f}cm x {BLADE_WIDTH*100:.0f}cm")

# ============================================
# カメラ（斜め角度から）
# ============================================
bpy.ops.object.camera_add(location=(0.5, -0.8, 0.5))
cam = bpy.context.active_object

# 斜め45度から見る
cam.rotation_euler = (math.radians(70), math.radians(5), math.radians(35))
bpy.context.scene.camera = cam

# ライト
bpy.ops.object.light_add(type='SUN', location=(2, -2, 5))
bpy.context.active_object.data.energy = 4.5

bpy.ops.object.light_add(type='AREA', location=(-0.3, -0.5, 0.4))
bpy.context.active_object.data.energy = 120

# 背景
world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
bpy.context.scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs['Color'].default_value = (0.02, 0.02, 0.05, 1.0)

# レンダリング
bpy.context.scene.render.resolution_x = 800
bpy.context.scene.render.resolution_y = 1000
bpy.context.scene.render.filepath = preview_path
bpy.ops.render.render(write_still=True)

print(f"📷 Preview: {preview_path}")

# クリーンアップ
for obj in bpy.data.objects:
    if obj.type in ['CAMERA', 'LIGHT']:
        bpy.data.objects.remove(obj)

# エクスポート
katana.select_set(True)
bpy.context.view_layer.objects.active = katana
fbx_path = os.path.join(output_dir, "DragonKatana_v6.fbx")
bpy.ops.export_scene.fbx(
    filepath=fbx_path,
    use_selection=True,
    object_types={'MESH'},
    axis_forward='-Z',
    axis_up='Y'
)

print(f"✅ Exported: {fbx_path}")
print("=== DONE ===")
