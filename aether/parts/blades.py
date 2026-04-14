"""
AETHER Parts - Blades (刀身パーツ)
"""

import bpy
import math

# utils は相対インポートではなく、Blender実行時に sys.path に追加して使う
# from aether.core import utils  # ← 実際の使用時


class Blade:
    """基本刀身クラス"""
    
    def __init__(self, length: float = 0.72, width: float = 0.04, thickness: float = 0.008):
        self.length = length
        self.width = width
        self.thickness = thickness
        self.obj = None
    
    def create(self) -> bpy.types.Object:
        """刀身メッシュを生成"""
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, self.length / 2))
        self.obj = bpy.context.active_object
        self.obj.name = "Blade"
        self.obj.scale = (self.thickness, self.width, self.length)
        bpy.ops.object.transform_apply(scale=True)
        return self.obj


class Handle:
    """柄クラス"""
    
    def __init__(self, length: float = 0.25, radius: float = 0.015):
        self.length = length
        self.radius = radius
        self.obj = None
    
    def create(self, z_offset: float = 0) -> bpy.types.Object:
        """柄メッシュを生成"""
        bpy.ops.mesh.primitive_cylinder_add(
            radius=self.radius,
            depth=self.length,
            location=(0, 0, z_offset - self.length / 2)
        )
        self.obj = bpy.context.active_object
        self.obj.name = "Handle"
        return self.obj


class Tsuba:
    """鍔クラス"""
    
    def __init__(self, radius: float = 0.04, thickness: float = 0.008):
        self.radius = radius
        self.thickness = thickness
        self.obj = None
    
    def create(self, z_offset: float = 0) -> bpy.types.Object:
        """鍔メッシュを生成"""
        bpy.ops.mesh.primitive_cylinder_add(
            radius=self.radius,
            depth=self.thickness,
            location=(0, 0, z_offset)
        )
        self.obj = bpy.context.active_object
        self.obj.name = "Tsuba"
        return self.obj


class Katana:
    """日本刀完成品クラス"""
    
    def __init__(self, blade_length: float = 0.72, handle_length: float = 0.25):
        self.blade = Blade(length=blade_length)
        self.handle = Handle(length=handle_length)
        self.tsuba = Tsuba()
        self.obj = None
    
    def create(self) -> bpy.types.Object:
        """日本刀を組み立て"""
        # 刀身（handleの上に配置）
        blade_obj = self.blade.create()
        blade_obj.location.z = self.handle.length
        
        # 鍔（刀身と柄の境目）
        tsuba_obj = self.tsuba.create(z_offset=self.handle.length)
        
        # 柄
        handle_obj = self.handle.create(z_offset=self.handle.length)
        
        # 結合
        bpy.ops.object.select_all(action='DESELECT')
        blade_obj.select_set(True)
        tsuba_obj.select_set(True)
        handle_obj.select_set(True)
        bpy.context.view_layer.objects.active = blade_obj
        bpy.ops.object.join()
        
        self.obj = bpy.context.active_object
        self.obj.name = "Katana"
        
        return self.obj
