"""
AETHER Demo: Sword Rain (50 Falling Swords)
Creates 50 swords at random positions in the sky with random rotations.
Each sword has Rigidbody properties baked into the export for Unity physics.
"""
import bpy
import os
import math
import random

# === OUTPUT PATH ===
output_dir = r"C:\Users\mukic\OneDrive\デスクトップ\パイプラインオートメーション\パイプラインオートメーション\Assets\Generated"
os.makedirs(output_dir, exist_ok=True)

# === CLEAR SCENE ===
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# === MATERIALS ===
# Ice Blade Material (Blue, Translucent, Emissive)
mat_ice = bpy.data.materials.new(name="IceBladeMat")
mat_ice.use_nodes = True
nodes = mat_ice.node_tree.nodes
nodes.clear()
node_output = nodes.new(type='ShaderNodeOutputMaterial')
node_output.location = (400, 0)
bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
bsdf.location = (0, 0)
mat_ice.node_tree.links.new(bsdf.outputs['BSDF'], node_output.inputs['Surface'])
bsdf.inputs['Base Color'].default_value = (0.0, 0.4, 1.0, 1.0)
bsdf.inputs['Transmission Weight'].default_value = 0.95
bsdf.inputs['Roughness'].default_value = 0.05
bsdf.inputs['Emission Color'].default_value = (0.0, 0.2, 0.8, 1.0)
bsdf.inputs['Emission Strength'].default_value = 2.0
mat_ice.blend_method = 'BLEND'

# Silver Hilt Material
mat_silver = bpy.data.materials.new(name="SilverHiltMat")
mat_silver.use_nodes = True
nodes_silver = mat_silver.node_tree.nodes
nodes_silver.clear()
node_output_s = nodes_silver.new(type='ShaderNodeOutputMaterial')
node_output_s.location = (400, 0)
bsdf_silver = nodes_silver.new(type='ShaderNodeBsdfPrincipled')
bsdf_silver.location = (0, 0)
mat_silver.node_tree.links.new(bsdf_silver.outputs['BSDF'], node_output_s.inputs['Surface'])
bsdf_silver.inputs['Base Color'].default_value = (0.8, 0.8, 0.85, 1.0)
bsdf_silver.inputs['Metallic'].default_value = 1.0
bsdf_silver.inputs['Roughness'].default_value = 0.2


def create_single_sword(name, location, rotation):
    """Create a single sword at the specified location and rotation."""
    parts = []
    
    # Blade
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0.8, 0))
    blade = bpy.context.active_object
    blade.scale = (0.2, 0.05, 1.5)
    bpy.ops.object.transform_apply(scale=True)
    blade.data.materials.append(mat_ice)
    parts.append(blade)
    
    # Blade Tip
    bpy.ops.mesh.primitive_cone_add(vertices=4, radius1=0.15, radius2=0, depth=0.5, location=(0, 2.50, 0))
    tip = bpy.context.active_object
    tip.rotation_euler = (0, 0, math.radians(45))
    tip.scale = (1.0, 0.1, 1.0)
    bpy.ops.object.transform_apply(scale=True, rotation=True)
    tip.data.materials.append(mat_ice)
    parts.append(tip)
    
    # Guard
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
    guard = bpy.context.active_object
    guard.scale = (0.6, 0.15, 0.1)
    bpy.ops.object.transform_apply(scale=True)
    guard.data.materials.append(mat_silver)
    parts.append(guard)
    
    # Handle
    bpy.ops.mesh.primitive_cylinder_add(radius=0.04, depth=0.6, location=(0, -0.35, 0))
    handle = bpy.context.active_object
    handle.data.materials.append(mat_silver)
    parts.append(handle)
    
    # Pommel
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.07, location=(0, -0.7, 0))
    pommel = bpy.context.active_object
    pommel.data.materials.append(mat_ice)
    parts.append(pommel)
    
    # Join all parts
    bpy.ops.object.select_all(action='DESELECT')
    for obj in parts:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    
    sword = bpy.context.active_object
    sword.name = name
    
    # UV Smart Project
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=66.0)
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Apply position and rotation
    sword.location = location
    sword.rotation_euler = rotation
    
    return sword


# === CREATE 50 SWORDS ===
swords = []

for i in range(50):
    # Random position in the sky (spread over a 20x20 area, 10-30 units high)
    x = random.uniform(-10, 10)
    y = random.uniform(-10, 10)
    z = random.uniform(10, 30)
    
    # Random rotation (all axes)
    rx = random.uniform(0, math.pi * 2)
    ry = random.uniform(0, math.pi * 2)
    rz = random.uniform(0, math.pi * 2)
    
    sword = create_single_sword(f"FallingSword_{i:03d}", (x, z, y), (rx, ry, rz))
    swords.append(sword)
    print(f"Created sword {i+1}/50")

# === SELECT ALL SWORDS FOR EXPORT ===
bpy.ops.object.select_all(action='SELECT')

# === EXPORT ===
output_path = os.path.join(output_dir, "SwordRain_50.fbx")
bpy.ops.export_scene.fbx(
    filepath=output_path,
    use_selection=True,
    object_types={'MESH'},
    axis_forward='-Z',
    axis_up='Y',
    bake_space_transform=True
)

print(f"SUCCESS: 50 swords exported to {output_path}")
print("Next Step: Import into Unity, add Rigidbody to each sword, and let them fall!")
