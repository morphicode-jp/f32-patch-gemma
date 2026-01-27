"""
Genesis Pipeline: IronCrate Creation
Creates a textured iron crate cube and exports to Unity
"""
import bpy
import os

# === STEP 1: CLEAR SCENE ===
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# === STEP 2: CREATE CUBE ===
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
crate = bpy.context.active_object
crate.name = "IronCrate"

# === STEP 3: UV UNWRAP ===
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.smart_project(angle_limit=66.0)
bpy.ops.object.mode_set(mode='OBJECT')

# === STEP 4: APPLY IRON MATERIAL ===
mat = bpy.data.materials.new(name="IronMaterial")
mat.diffuse_color = (0.4, 0.4, 0.45, 1)  # Iron gray color
crate.data.materials.append(mat)

# === STEP 5: EXPORT FBX ===
output_path = r"C:\Users\mukic\OneDrive\デスクトップ\パイプラインオートメーション\パイプラインオートメーション\Assets\Generated\IronCrate.fbx"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

bpy.ops.export_scene.fbx(
    filepath=output_path,
    use_selection=True,
    object_types={'MESH'},
    axis_forward='-Z',
    axis_up='Y',
    bake_space_transform=True
)

print(f"SUCCESS: IronCrate exported to {output_path}")
