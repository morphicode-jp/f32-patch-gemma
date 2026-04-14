import bpy
import os

OUTPUT_DIR = r"C:\Users\mukic\OneDrive\デスクトップ\パイプラインオートメーション\パイプラインオートメーション\Assets\_Pipeline\Generated"
PREVIEW_DIR = r"C:\Users\mukic\OneDrive\デスクトップ\パイプラインオートメーション"

def join_objects(objects: list) -> bpy.types.Object:
    if not objects:
        return None
    
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        obj.select_set(True)
    
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
    
    return bpy.context.active_object

def apply_transforms(obj: bpy.types.Object):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

def smart_uv(obj: bpy.types.Object):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project()
    bpy.ops.object.mode_set(mode='OBJECT')

def set_origin_center(obj: bpy.types.Object):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')

def finalize_and_export(obj: bpy.types.Object, name: str, export_fbx: bool = True) -> str:
    obj.name = name
    
    apply_transforms(obj)
    
    smart_uv(obj)
    
    set_origin_center(obj)
    
    if export_fbx:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fbx_path = os.path.join(OUTPUT_DIR, f"{name}.fbx")
        
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        
        bpy.ops.export_scene.fbx(
            filepath=fbx_path,
            use_selection=True,
            axis_forward='-Z',
            axis_up='Y',
            bake_space_transform=True
        )
        
        print(f"[OK] Exported: {fbx_path}")
        return fbx_path
    
    return None

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    for mat in bpy.data.materials:
        bpy.data.materials.remove(mat)
    for action in bpy.data.actions:
        bpy.data.actions.remove(action)

def create_material(name: str, color: tuple, metallic: float = 0.0, roughness: float = 0.5) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    output = nodes.new(type='ShaderNodeOutputMaterial')
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    bsdf.inputs['Base Color'].default_value = color
    bsdf.inputs['Metallic'].default_value = metallic
    bsdf.inputs['Roughness'].default_value = roughness
    
    return mat

def finalize_hierarchy_and_export(root: bpy.types.Object, name: str, export_fbx: bool = True) -> str:
    root.name = name
    
    for obj in root.children_recursive:
        if obj.type == 'MESH':
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            
            bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='MEDIAN')
            
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    
    print(f"[OK] Transforms applied to {len(list(root.children_recursive))} children")
    
    if export_fbx:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fbx_path = os.path.join(OUTPUT_DIR, f"{name}.fbx")
        
        bpy.ops.object.select_all(action='DESELECT')
        root.select_set(True)
        for obj in root.children_recursive:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = root
        
        bpy.ops.export_scene.fbx(
            filepath=fbx_path,
            use_selection=True,
            object_types={'MESH', 'ARMATURE', 'EMPTY'},
            use_mesh_modifiers=True,
            mesh_smooth_type='FACE',
            bake_anim=False,
            path_mode='COPY',
            embed_textures=True,
            axis_forward='-Z',
            axis_up='Y',
            use_custom_props=True,
        )
        
        print(f"[OK] Hierarchy FBX Exported: {fbx_path}")
        return fbx_path
    
    return None

def create_root_with_children(name: str, children: list) -> bpy.types.Object:
    bpy.ops.object.empty_add(location=(0, 0, 0))
    root = bpy.context.active_object
    root.name = name
    
    for child in children:
        if hasattr(child, 'parent'):
            child.parent = root
    
    print(f"[OK] Created root '{name}' with {len(children)} children")
    return root
