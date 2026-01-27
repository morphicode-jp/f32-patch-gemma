"""
Genesis Pipeline: Pitagora Switch - Ramp + Iron Ball
Creates a ramp (slope) and a heavy iron ball for physics chain reaction
"""
import bpy
import os
import math

# === OUTPUT PATH ===
output_dir = r"C:\Users\mukic\OneDrive\デスクトップ\パイプラインオートメーション\パイプラインオートメーション\Assets\Generated"
os.makedirs(output_dir, exist_ok=True)

# === CLEAR SCENE ===
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# ============================================
# PART 1: CREATE RAMP (坂道)
# ============================================
# Create a wedge/ramp shape using a cube and shearing
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
ramp = bpy.context.active_object
ramp.name = "Ramp"

# Scale to make a long flat surface
ramp.scale = (2.0, 0.5, 0.1)  # Long, narrow, thin
bpy.ops.object.transform_apply(scale=True)

# Rotate to create the slope (30 degrees)
ramp.rotation_euler = (0, math.radians(-20), 0)
bpy.ops.object.transform_apply(rotation=True)

# Move up so it's above ground
ramp.location = (0, 0, 0.5)

# UV Unwrap
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.smart_project(angle_limit=66.0)
bpy.ops.object.mode_set(mode='OBJECT')

# Apply gray material
mat = bpy.data.materials.new(name="RampMaterial")
mat.diffuse_color = (0.4, 0.4, 0.4, 1)  # Gray stone
ramp.data.materials.append(mat)

# Export Ramp
bpy.ops.object.select_all(action='DESELECT')
ramp.select_set(True)
bpy.context.view_layer.objects.active = ramp
bpy.ops.export_scene.fbx(
    filepath=os.path.join(output_dir, "Ramp.fbx"),
    use_selection=True,
    object_types={'MESH'},
    axis_forward='-Z',
    axis_up='Y',
    bake_space_transform=True
)
print("SUCCESS: Ramp exported")

# ============================================
# PART 2: CREATE IRON BALL (鉄球)
# ============================================
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.15, location=(0, 0, 0))
ball = bpy.context.active_object
ball.name = "IronBall"

# Smooth shade
bpy.ops.object.shade_smooth()

# UV Unwrap
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.smart_project(angle_limit=66.0)
bpy.ops.object.mode_set(mode='OBJECT')

# Apply metallic material
mat_ball = bpy.data.materials.new(name="IronMaterial")
mat_ball.diffuse_color = (0.2, 0.2, 0.25, 1)  # Dark iron
ball.data.materials.append(mat_ball)

# Export Ball
bpy.ops.object.select_all(action='DESELECT')
ball.select_set(True)
bpy.context.view_layer.objects.active = ball
bpy.ops.export_scene.fbx(
    filepath=os.path.join(output_dir, "IronBall.fbx"),
    use_selection=True,
    object_types={'MESH'},
    axis_forward='-Z',
    axis_up='Y',
    bake_space_transform=True
)
print("SUCCESS: IronBall exported")
print("ALL DONE!")
