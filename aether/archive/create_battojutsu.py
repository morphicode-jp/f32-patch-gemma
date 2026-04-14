"""
Dragon Katana Draw/Sheath Animation (抜刀・納刀)
Creates an animated FBX with katana drawing from and returning to scabbard.

Animation Timeline:
- Frame 1-30: 納刀 (Sheathed/Rest position)
- Frame 30-60: 抜刀 (Drawing the sword)
- Frame 60-90: 構え (Guard/Ready position)
- Frame 90-120: 納刀 (Sheathing back)
"""

import bpy
import bmesh
import math
import os

# === SHARED BOUNDARY VARIABLES (SBV) ===
# Must match katana v6 and saya dimensions
BLADE_LENGTH = 0.72
HANDLE_LENGTH = 0.25
SAYA_LENGTH = 0.77

TOTAL_KATANA_LENGTH = BLADE_LENGTH + HANDLE_LENGTH  # 0.97m

OUTPUT_DIR = r"C:\Users\mukic\OneDrive\デスクトップ\パイプラインオートメーション\パイプラインオートメーション\Assets\Generated"
PREVIEW_PATH = r"C:\Users\mukic\OneDrive\デスクトップ\パイプラインオートメーション\battojutsu_preview.png"

# === CLEANUP ===
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

for mesh in bpy.data.meshes:
    bpy.data.meshes.remove(mesh)
for mat in bpy.data.materials:
    bpy.data.materials.remove(mat)
for action in bpy.data.actions:
    bpy.data.actions.remove(action)

print("=== Creating Battojutsu Animation (抜刀・納刀) ===")

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

# Materials
blade_mat = create_mat("Blade", (0.9, 0.9, 0.95, 1.0), 1.0, 0.1)
handle_mat = create_mat("Handle", (0.05, 0.05, 0.05, 1.0), 0.0, 0.8)
tsuba_mat = create_mat("Tsuba", (0.55, 0.35, 0.15, 1.0), 0.9, 0.4)
lacquer_mat = create_mat("Lacquer", (0.02, 0.02, 0.03, 1.0), 0.0, 0.2)

# === CREATE KATANA (Simplified v6) ===
def create_katana():
    # Blade
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, BLADE_LENGTH / 2 + HANDLE_LENGTH))
    blade = bpy.context.active_object
    blade.name = "Blade"
    blade.scale = (0.008, 0.04, BLADE_LENGTH)
    bpy.ops.object.transform_apply(scale=True)
    blade.data.materials.append(blade_mat)
    
    # Handle
    bpy.ops.mesh.primitive_cylinder_add(radius=0.015, depth=HANDLE_LENGTH, location=(0, 0, HANDLE_LENGTH / 2))
    handle = bpy.context.active_object
    handle.name = "Handle"
    handle.data.materials.append(handle_mat)
    
    # Tsuba
    bpy.ops.mesh.primitive_cylinder_add(radius=0.04, depth=0.008, location=(0, 0, HANDLE_LENGTH))
    tsuba = bpy.context.active_object
    tsuba.name = "Tsuba"
    tsuba.data.materials.append(tsuba_mat)
    
    # Join
    bpy.ops.object.select_all(action='DESELECT')
    blade.select_set(True)
    handle.select_set(True)
    tsuba.select_set(True)
    bpy.context.view_layer.objects.active = blade
    bpy.ops.object.join()
    
    katana = bpy.context.active_object
    katana.name = "Katana"
    
    # Origin at the bottom of handle (for proper rotation)
    bpy.context.scene.cursor.location = (0, 0, 0)
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
    
    return katana

# === CREATE SAYA (Simplified) ===
def create_saya():
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, SAYA_LENGTH / 2))
    saya = bpy.context.active_object
    saya.name = "Saya"
    saya.scale = (0.018, 0.04, SAYA_LENGTH)
    bpy.ops.object.transform_apply(scale=True)
    saya.data.materials.append(lacquer_mat)
    
    # Origin at the opening (top)
    bpy.context.scene.cursor.location = (0, 0, SAYA_LENGTH)
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
    
    return saya

# Create objects
katana = create_katana()
saya = create_saya()

# Position saya at the origin, opening at top
saya.location = (0, 0, 0)
saya.rotation_euler = (0, 0, 0)

# Initial position: Katana inside saya (納刀状態)
# Handle sticks out above the saya opening
katana.location = (0, 0, 0)  # Aligned with saya

print("✅ Created Katana and Saya")

# === CREATE ANIMATION ===
# Set up timeline
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = 120
bpy.context.scene.render.fps = 30

# Create action for katana
katana.animation_data_create()
action = bpy.data.actions.new(name="Battojutsu")
katana.animation_data.action = action

# Keyframe helper
def set_keyframe(obj, frame, loc_z):
    obj.location.z = loc_z
    obj.keyframe_insert(data_path="location", index=2, frame=frame)

# Animation keyframes
# Frame 1-30: 納刀状態 (Sheathed)
set_keyframe(katana, 1, 0)
set_keyframe(katana, 30, 0)

# Frame 30-60: 抜刀 (Drawing - blade moves up and out)
set_keyframe(katana, 45, BLADE_LENGTH * 0.5)  # Halfway out
set_keyframe(katana, 60, BLADE_LENGTH + 0.1)  # Fully drawn

# Frame 60-90: 構え (Guard position - hold)
set_keyframe(katana, 90, BLADE_LENGTH + 0.1)

# Frame 90-120: 納刀 (Sheathing back)
set_keyframe(katana, 105, BLADE_LENGTH * 0.5)  # Halfway in
set_keyframe(katana, 120, 0)  # Fully sheathed

print("✅ Animation created (120 frames @ 30fps = 4 seconds)")

# === CAMERA (適切な配置) ===
cam_distance = 1.5
bpy.ops.object.camera_add(location=(cam_distance * 0.8, -cam_distance, cam_distance * 0.6))
cam = bpy.context.active_object
cam.rotation_euler = (math.radians(55), 0, math.radians(35))
cam.data.lens = 50
bpy.context.scene.camera = cam

# === LIGHTING (明るく！) ===
bpy.ops.object.light_add(type='SUN', location=(5, -5, 10))
sun = bpy.context.active_object
sun.data.energy = 7.0

bpy.ops.object.light_add(type='AREA', location=(-3, 3, 5))
fill = bpy.context.active_object
fill.data.energy = 300

# 背景
world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
bpy.context.scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs['Color'].default_value = (0.3, 0.3, 0.35, 1.0)

# === RENDER PREVIEW (halfway through draw) ===
bpy.context.scene.frame_set(45)  # Show mid-draw
bpy.context.scene.render.resolution_x = 800
bpy.context.scene.render.resolution_y = 1000
bpy.context.scene.render.filepath = PREVIEW_PATH
bpy.ops.render.render(write_still=True)
print(f"📷 Preview (frame 45): {PREVIEW_PATH}")

# === EXPORT FBX WITH ANIMATION ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
fbx_path = os.path.join(OUTPUT_DIR, "DragonKatana_Battojutsu.fbx")

bpy.ops.object.select_all(action='DESELECT')
katana.select_set(True)
saya.select_set(True)
bpy.context.view_layer.objects.active = katana

bpy.ops.export_scene.fbx(
    filepath=fbx_path,
    use_selection=True,
    axis_forward='-Z',
    axis_up='Y',
    bake_space_transform=True,
    bake_anim=True,
    bake_anim_use_all_actions=True,
    bake_anim_use_nla_strips=False,
    bake_anim_use_all_bones=False,
    bake_anim_force_startend_keying=True,
    bake_anim_step=1.0,
    bake_anim_simplify_factor=1.0
)

print(f"✅ Exported with animation: {fbx_path}")
print("=== DONE ===")
print("Unity: Import FBX → Rig → Animation tab has 'Battojutsu' clip")
print("Timeline: 0-1s Sheathed → 1-2s Draw → 2-3s Guard → 3-4s Sheath")
