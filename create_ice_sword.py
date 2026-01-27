"""
Genesis Pipeline: Fantasy Ice Sword
Creates a legendary Ice Sword with translucent blade and decorated silver hilt.
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

#Helper function for cleaning boolean operations or joining
def join_objects(objects):
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()

# ============================================
# PART 1: MATERIALS
# ============================================

# 1. Ice Blade Material (Blue, Translucent, Emissive)
mat_ice = bpy.data.materials.new(name="IceBladeMat")
mat_ice.use_nodes = True
nodes = mat_ice.node_tree.nodes
nodes.clear() # Safe start
# Create nodes
node_output = nodes.new(type='ShaderNodeOutputMaterial')
node_output.location = (400,0)
bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
bsdf.location = (0,0)
# Link
mat_ice.node_tree.links.new(bsdf.outputs['BSDF'], node_output.inputs['Surface'])

# Set Properties
bsdf.inputs['Base Color'].default_value = (0.0, 0.4, 1.0, 1.0) # Light Blue
bsdf.inputs['Transmission Weight'].default_value = 0.95 
bsdf.inputs['Roughness'].default_value = 0.05
bsdf.inputs['Emission Color'].default_value = (0.0, 0.2, 0.8, 1.0)
bsdf.inputs['Emission Strength'].default_value = 2.0
mat_ice.blend_method = 'BLEND'

# 2. Silver Hilt Material
mat_silver = bpy.data.materials.new(name="SilverHiltMat")
mat_silver.use_nodes = True
nodes_silver = mat_silver.node_tree.nodes
nodes_silver.clear()
node_output_s = nodes_silver.new(type='ShaderNodeOutputMaterial')
node_output_s.location = (400,0)
bsdf_silver = nodes_silver.new(type='ShaderNodeBsdfPrincipled')
bsdf_silver.location = (0,0)
mat_silver.node_tree.links.new(bsdf_silver.outputs['BSDF'], node_output_s.inputs['Surface'])

bsdf_silver.inputs['Base Color'].default_value = (0.8, 0.8, 0.85, 1.0)
bsdf_silver.inputs['Metallic'].default_value = 1.0
bsdf_silver.inputs['Roughness'].default_value = 0.2

# ============================================
# PART 2: MODELING
# ============================================

parts = []

# --- BLADE ---
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0.8, 0)) # Centered roughly
blade = bpy.context.active_object
blade.name = "Blade"
# Shape: Long (Z), Thin (Y), Medium width (X) -> Then Taper
blade.scale = (0.2, 0.05, 1.5)
bpy.ops.object.transform_apply(scale=True)

# Taper the tip
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='DESELECT')
# Select top vertices (Z > 1.0 roughly)
bpy.ops.object.mode_set(mode='OBJECT')
# Simple way: create a taper modifier or just manipulate verts. 
# Let's use simple geometry scaling on top face.
# Actually, let's just use another primitive for the tip or modify geometry directly.
# Let's switch to simple cone for tip + cube for body? No, let's edit mesh.
bpy.ops.object.mode_set(mode='EDIT')
# Select top face
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.bisect(plane_co=(0,0,2.0), plane_no=(0,0,1), use_fill=False, clear_inner=False, clear_outer=False) 
# Too complex to script robustly without exact indices.
# Let's stick to primitive composition for robustness.
bpy.ops.object.mode_set(mode='OBJECT')
blade.data.materials.append(mat_ice)
parts.append(blade)

# Blade Tip (Pyramid/Cone flattened)
bpy.ops.mesh.primitive_cone_add(vertices=4, radius1=0.15, radius2=0, depth=0.5, location=(0, 2.50, 0)) # Top of blade
tip = bpy.context.active_object
tip.rotation_euler = (0, 0, math.radians(45)) # Align diamond shape
tip.scale = (1.0, 0.1, 1.0) # Flatten
bpy.ops.object.transform_apply(scale=True, rotation=True)
tip.data.materials.append(mat_ice)
parts.append(tip)


# --- GUARD (Guard) ---
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
guard = bpy.context.active_object
guard.scale = (0.6, 0.15, 0.1)
bpy.ops.object.transform_apply(scale=True)
guard.data.materials.append(mat_silver)
parts.append(guard)

# Decorations on Guard (Spheres)
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.08, location=(0.35, 0, 0))
deco1 = bpy.context.active_object
deco1.data.materials.append(mat_ice) # Ice gems!
parts.append(deco1)

bpy.ops.mesh.primitive_uv_sphere_add(radius=0.08, location=(-0.35, 0, 0))
deco2 = bpy.context.active_object
deco2.data.materials.append(mat_ice)
parts.append(deco2)

# --- HANDLE (Hilt) ---
bpy.ops.mesh.primitive_cylinder_add(radius=0.04, depth=0.6, location=(0, -0.35, 0))
handle = bpy.context.active_object
handle.data.materials.append(mat_silver)
parts.append(handle)

# --- POMMEL (Bottom) ---
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.07, location=(0, -0.7, 0))
pommel = bpy.context.active_object
pommel.data.materials.append(mat_ice) # Ice jewel pommel
parts.append(pommel)


# Join all as one object for reliable prefab
join_objects(parts)
sword = bpy.context.active_object
sword.name = "IceSword"

# Move pivot to handle (approx)
# Currently origin is likely at 0,0,0 (Guard center). That's good for a hand grip.

# Ensure correct rotation/scale
sword.rotation_euler = (math.radians(-90), 0, 0) # Lay flat? No, let's keep it upright but handle export axis.
# Actually, the export command handles axis conversion. Blender Y-up vs Unity Y-up.
# Standard: Blender Z-up. create_domino used axis_forward='-Z', axis_up='Y'.
# If I model it standing up (Z-axis), usage of axis_up='Y' during export should orient it correctly in Unity (Y-up).

# UV Smart Project
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.smart_project(angle_limit=66.0)
bpy.ops.object.mode_set(mode='OBJECT')


# === EXPORT ===
output_path = os.path.join(output_dir, "IceSword.fbx")
bpy.ops.export_scene.fbx(
    filepath=output_path,
    use_selection=True,
    object_types={'MESH'},
    axis_forward='-Z',
    axis_up='Y',
    bake_space_transform=True
)
print(f"SUCCESS: IceSword exported to {output_path}")
