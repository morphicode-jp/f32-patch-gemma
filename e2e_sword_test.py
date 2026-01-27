"""
Genesis Pipeline End-to-End Test: Sword Creation
This script creates a simple sword in Blender and exports to Unity
"""
import bpy
import os

# === STEP 1: CLEAR SCENE ===
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# === STEP 2: CREATE SWORD GEOMETRY ===

# Blade (stretched cube)
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.6))
blade = bpy.context.active_object
blade.name = "Blade"
blade.scale = (0.05, 0.02, 0.5)
bpy.ops.object.transform_apply(scale=True)

# Guard (flattened cube)
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.05))
guard = bpy.context.active_object
guard.name = "Guard"
guard.scale = (0.15, 0.03, 0.03)
bpy.ops.object.transform_apply(scale=True)

# Handle (cylinder)
bpy.ops.mesh.primitive_cylinder_add(radius=0.025, depth=0.2, location=(0, 0, -0.1))
handle = bpy.context.active_object
handle.name = "Handle"

# Pommel (sphere)
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.04, location=(0, 0, -0.22))
pommel = bpy.context.active_object
pommel.name = "Pommel"

# === STEP 3: JOIN ALL PARTS ===
bpy.ops.object.select_all(action='SELECT')
bpy.context.view_layer.objects.active = blade
bpy.ops.object.join()
blade.name = "GenesisSword"

# === STEP 4: UV UNWRAP (CRITICAL!) ===
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.smart_project(angle_limit=66.0)
bpy.ops.object.mode_set(mode='OBJECT')

# === STEP 5: APPLY SIMPLE MATERIAL (Blender 5.0 compatible) ===
mat = bpy.data.materials.new(name="SwordMaterial")
mat.diffuse_color = (0.7, 0.7, 0.75, 1)  # Silver color
blade.data.materials.append(mat)

# === STEP 6: EXPORT FBX ===
output_path = r"C:\Users\mukic\OneDrive\デスクトップ\パイプラインオートメーション\パイプラインオートメーション\Assets\Generated\GenesisSword.fbx"

# Ensure directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

bpy.ops.export_scene.fbx(
    filepath=output_path,
    use_selection=True,
    object_types={'MESH'},
    axis_forward='-Z',
    axis_up='Y',
    bake_space_transform=True
)

print(f"SUCCESS: Exported to {output_path}")
