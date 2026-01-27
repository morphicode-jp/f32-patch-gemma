"""
Genesis Pipeline: Domino Creation
Creates a thin domino piece (0.2 x 1.0 x 0.5) and exports to Unity
"""
import bpy
import os

# === STEP 1: CLEAR SCENE ===
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# === STEP 2: CREATE THIN DOMINO ===
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.5))  # Center at bottom
domino = bpy.context.active_object
domino.name = "Domino"

# Apply scale: 0.2 (width) x 1.0 (height) x 0.5 (depth)
domino.scale = (0.2, 0.5, 1.0)  # Blender: X=width, Y=depth, Z=height
bpy.ops.object.transform_apply(scale=True)

# === STEP 3: UV UNWRAP ===
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.smart_project(angle_limit=66.0)
bpy.ops.object.mode_set(mode='OBJECT')

# === STEP 4: APPLY DOMINO MATERIAL (white with black dots effect) ===
mat = bpy.data.materials.new(name="DominoMaterial")
mat.diffuse_color = (0.9, 0.9, 0.9, 1)  # White
domino.data.materials.append(mat)

# === STEP 5: EXPORT FBX ===
output_path = r"C:\Users\mukic\OneDrive\デスクトップ\パイプラインオートメーション\パイプラインオートメーション\Assets\Generated\Domino.fbx"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

bpy.ops.export_scene.fbx(
    filepath=output_path,
    use_selection=True,
    object_types={'MESH'},
    axis_forward='-Z',
    axis_up='Y',
    bake_space_transform=True
)

print(f"SUCCESS: Domino exported to {output_path}")
