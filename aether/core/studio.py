import bpy
import math
import os

AETHER_ROOT = r"C:\Users\mukic\OneDrive\デスクトップ\パイプラインオートメーション"
PREVIEW_DIR = os.path.join(AETHER_ROOT, "aether", "previews")

def setup_preview_scene(target_obj: bpy.types.Object = None):
    if target_obj:
        dims = target_obj.dimensions
        max_dim = max(dims)
        cam_distance = max_dim * 2.5
        look_at = target_obj.location
    else:
        cam_distance = 2.0
        look_at = (0, 0, 0.5)
    
    bpy.ops.object.camera_add(
        location=(cam_distance * 0.7, -cam_distance * 0.7, cam_distance * 0.5)
    )
    cam = bpy.context.active_object
    cam.name = "PreviewCamera"
    cam.rotation_euler = (math.radians(60), 0, math.radians(45))
    cam.data.lens = 50  # 標準レンズ
    bpy.context.scene.camera = cam
    
    bpy.ops.object.light_add(type='SUN', location=(5, -5, 10))
    sun = bpy.context.active_object
    sun.name = "MainSun"
    sun.data.energy = 6.0  # 明るめ
    sun.rotation_euler = (math.radians(45), 0, math.radians(30))
    
    bpy.ops.object.light_add(type='AREA', location=(-3, 3, 5))
    fill = bpy.context.active_object
    fill.name = "FillLight"
    fill.data.energy = 250
    
    world = bpy.data.worlds.new(name="StudioWorld")
    bpy.context.scene.world = world
    world.color = (0.28, 0.28, 0.32)
    
    print("[OK] Studio setup complete")

def setup_camera_for_object(obj: bpy.types.Object, angle: str = "diagonal"):
    dims = obj.dimensions
    max_dim = max(dims)
    cam_distance = max_dim * 2.5
    
    angles = {
        "diagonal": (math.radians(60), 0, math.radians(45)),
        "front": (math.radians(90), 0, 0),
        "side": (math.radians(90), 0, math.radians(90)),
        "top": (0, 0, 0),
    }
    
    if angle == "diagonal":
        loc = (cam_distance * 0.7, -cam_distance * 0.7, cam_distance * 0.5)
    elif angle == "front":
        loc = (0, -cam_distance, max_dim * 0.5)
    elif angle == "side":
        loc = (cam_distance, 0, max_dim * 0.5)
    else:  # top
        loc = (0, 0, cam_distance)
    
    if bpy.context.scene.camera:
        cam = bpy.context.scene.camera
        cam.location = loc
        cam.rotation_euler = angles.get(angle, angles["diagonal"])
    else:
        bpy.ops.object.camera_add(location=loc)
        cam = bpy.context.active_object
        cam.rotation_euler = angles.get(angle, angles["diagonal"])
        cam.data.lens = 50
        bpy.context.scene.camera = cam

def render(name: str = "preview", resolution: tuple = (800, 1000)) -> str:
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]
    
    filepath = os.path.join(PREVIEW_DIR, f"{name}.png")
    bpy.context.scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)
    
    print(f"📷 Rendered: {filepath}")
    return filepath

def render_animation(name: str = "animation", frame_start: int = 1, frame_end: int = 120) -> str:
    bpy.context.scene.frame_start = frame_start
    bpy.context.scene.frame_end = frame_end
    
    output_dir = os.path.join(PREVIEW_DIR, name)
    os.makedirs(output_dir, exist_ok=True)
    
    bpy.context.scene.render.filepath = os.path.join(output_dir, "frame_")
    bpy.ops.render.render(animation=True)
    
    print(f"🎬 Animation rendered: {output_dir}")
    return output_dir

def setup_studio(target_obj=None):
    return setup_preview_scene(target_obj)

def capture_preview(filepath: str, resolution: tuple = (800, 600)) -> str:
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]
    bpy.context.scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)
    
    print(f"📷 Preview saved: {filepath}")
    return filepath
