"""
AETHER Demo: Katana Rain (50 Falling Dragon Katanas)
Authentic Dragon Katanas falling from the sky with random positions and rotations.
Uses the v6 katana geometry with proper blade width and curvature.
"""
import bpy
import bmesh
import os
import math
import random

# === OUTPUT PATH ===
output_dir = r"C:\Users\mukic\OneDrive\デスクトップ\パイプラインオートメーション\パイプラインオートメーション\Assets\Generated"
preview_path = r"C:\Users\mukic\OneDrive\デスクトップ\パイプラインオートメーション\katana_rain_preview.png"
os.makedirs(output_dir, exist_ok=True)

# === CLEAR SCENE ===
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

print("=== Creating Katana Rain (50 Dragon Katanas) ===")

# ============================================
# 寸法（Dragon Katana v6と同じ）
# ============================================
BLADE_LENGTH = 0.72
BLADE_WIDTH = 0.06
BLADE_THICK = 0.012
CURVE_DEPTH = 0.025
TSUBA_RADIUS = 0.055
TSUBA_THICK = 0.008
HANDLE_LENGTH = 0.26
HANDLE_RADIUS = 0.018

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


def create_katana(name, location, rotation):
    """Create a single Dragon Katana at specified location and rotation."""
    
    # === 1. BLADE ===
    mesh = bpy.data.meshes.new(f"{name}_Blade_Mesh")
    bm = bmesh.new()
    
    segments = 16  # Reduced for performance (50 swords)
    for i in range(segments + 1):
        t = i / segments
        z = TSUBA_THICK + t * BLADE_LENGTH
        sori = CURVE_DEPTH * math.sin(t * math.pi)
        taper = 1.0 - t * 0.4
        w = BLADE_WIDTH * taper
        th = BLADE_THICK * taper
        
        verts = []
        verts.append(bm.verts.new((0, sori + th/2, z)))
        verts.append(bm.verts.new((w * 0.3, sori + th/4, z)))
        verts.append(bm.verts.new((w * 0.5, sori, z)))
        verts.append(bm.verts.new((w * 0.3, sori - th/2, z)))
        verts.append(bm.verts.new((-w * 0.15, sori, z)))
        verts.append(bm.verts.new((-w * 0.1, sori + th/4, z)))
        
        if i > 0:
            prev = bm.verts[-12:-6]
            curr = bm.verts[-6:]
            for j in range(6):
                j2 = (j + 1) % 6
                try:
                    bm.faces.new([prev[j], prev[j2], curr[j2], curr[j]])
                except:
                    pass
    
    # Tip
    tip_verts = bm.verts[-6:]
    tip_center = bm.verts.new((0, CURVE_DEPTH * 0.5, TSUBA_THICK + BLADE_LENGTH + 0.03))
    for j in range(6):
        j2 = (j + 1) % 6
        try:
            bm.faces.new([tip_verts[j], tip_verts[j2], tip_center])
        except:
            pass
    
    # Base
    base_verts = list(bm.verts)[:6]
    try:
        bm.faces.new(base_verts[::-1])
    except:
        pass
    
    bm.to_mesh(mesh)
    bm.free()
    blade = bpy.data.objects.new(f"{name}_Blade", mesh)
    bpy.context.collection.objects.link(blade)
    blade.data.materials.append(mat_blade)
    
    # === 2. TSUBA ===
    bpy.ops.mesh.primitive_cylinder_add(
        radius=TSUBA_RADIUS,
        depth=TSUBA_THICK,
        location=(0, 0, TSUBA_THICK/2),
        vertices=16
    )
    tsuba = bpy.context.active_object
    tsuba.name = f"{name}_Tsuba"
    tsuba.data.materials.append(mat_tsuba)
    
    # === 3. HANDLE ===
    bpy.ops.mesh.primitive_cylinder_add(
        radius=HANDLE_RADIUS,
        depth=HANDLE_LENGTH,
        location=(0, 0, -HANDLE_LENGTH/2),
        vertices=8
    )
    handle = bpy.context.active_object
    handle.name = f"{name}_Handle"
    handle.data.materials.append(mat_handle)
    
    # === JOIN ===
    bpy.ops.object.select_all(action='DESELECT')
    blade.select_set(True)
    tsuba.select_set(True)
    handle.select_set(True)
    bpy.context.view_layer.objects.active = blade
    bpy.ops.object.join()
    
    katana = bpy.context.active_object
    katana.name = name
    
    # Apply transforms
    katana.location = location
    katana.rotation_euler = rotation
    
    # Smooth shading
    for poly in katana.data.polygons:
        poly.use_smooth = True
    
    return katana


# === CREATE 50 KATANAS ===
katanas = []

for i in range(50):
    # Random position in the sky (compact spread for better preview)
    x = random.uniform(-4, 4)
    y = random.uniform(-4, 4)
    z = random.uniform(5, 12)
    
    # Random rotation (pointing mostly downward with some variation)
    rx = random.uniform(math.pi * 0.6, math.pi * 0.9)  # Mostly pointing down
    ry = random.uniform(0, math.pi * 2)
    rz = random.uniform(-0.3, 0.3)
    
    katana = create_katana(f"Katana_{i:03d}", (x, y, z), (rx, ry, rz))
    katanas.append(katana)
    
    if (i + 1) % 10 == 0:
        print(f"  Created {i + 1}/50 katanas...")

print(f"✅ Created {len(katanas)} katanas")

# === CAMERA & PREVIEW ===
bpy.ops.object.camera_add(location=(5, -6, 8))
cam = bpy.context.active_object
cam.rotation_euler = (math.radians(50), 0, math.radians(40))
cam.data.lens = 35
bpy.context.scene.camera = cam

bpy.ops.object.light_add(type='SUN', location=(10, -10, 30))
bpy.context.active_object.data.energy = 4.0

world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
bpy.context.scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs['Color'].default_value = (0.4, 0.5, 0.7, 1.0)  # Sky blue

bpy.context.scene.render.resolution_x = 1200
bpy.context.scene.render.resolution_y = 800
bpy.context.scene.render.filepath = preview_path
bpy.ops.render.render(write_still=True)

print(f"📷 Preview: {preview_path}")

# Delete camera/light for export
for obj in list(bpy.data.objects):
    if obj.type in ['CAMERA', 'LIGHT']:
        bpy.data.objects.remove(obj)

# === EXPORT ===
bpy.ops.object.select_all(action='SELECT')
output_path = os.path.join(output_dir, "KatanaRain_50.fbx")
bpy.ops.export_scene.fbx(
    filepath=output_path,
    use_selection=True,
    object_types={'MESH'},
    axis_forward='-Z',
    axis_up='Y',
    bake_space_transform=True
)

print(f"✅ Exported: {output_path}")
print("=== DONE ===")
print("Unity: Import FBX, add Rigidbody to each katana, and let them fall!")
