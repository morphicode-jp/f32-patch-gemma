"""
AETHER Custom Parts - AI生成の新規パーツ
このファイルにはAIが新規作成したクラス・関数が自動追記されます
"""

import bpy
import bmesh
import math

# === AI Generated parts will be appended below ===


# === AI Generated ===
class Tanto:
    """短刀クラス（約25cm）"""
    def __init__(self, blade_length=0.25):
        self.blade_length = blade_length
    
    def create(self):
        import bpy
        bpy.ops.mesh.primitive_cube_add(size=1)
        obj = bpy.context.active_object
        obj.scale = (0.01, 0.02, self.blade_length)
        bpy.ops.object.transform_apply(scale=True)
        obj.name = "Tanto"
        return obj


# === AI Generated ===
class Yari:
    """槍クラス（約2m）"""
    def __init__(self, shaft_length=2.0):
        self.shaft_length = shaft_length
    
    def create(self):
        import bpy
        bpy.ops.mesh.primitive_cylinder_add(radius=0.02, depth=self.shaft_length)
        return bpy.context.active_object


# === AI Generated ===

class Kunai:
    """忍者のクナイ - ひし形の刃とリング付き持ち手"""
    
    def __init__(self, blade_length=0.15, ring_radius=0.025):
        self.blade_length = blade_length
        self.ring_radius = ring_radius
    
    def create(self):
        import bpy
        import bmesh
        import math
        
        bm = bmesh.new()
        mesh = bpy.data.meshes.new("KunaiMesh")
        
        # ひし形の刃 (4頂点のダイヤモンド形状)
        bl = self.blade_length
        bw = 0.02  # 刃幅
        bt = 0.005  # 刃厚
        
        # 刃の頂点
        tip = bm.verts.new((0, 0, bl))
        left = bm.verts.new((-bw, 0, bl * 0.3))
        right = bm.verts.new((bw, 0, bl * 0.3))
        front = bm.verts.new((0, bt, bl * 0.3))
        back = bm.verts.new((0, -bt, bl * 0.3))
        base = bm.verts.new((0, 0, 0))
        
        # 刃の面
        bm.faces.new([tip, left, front])
        bm.faces.new([tip, front, right])
        bm.faces.new([tip, right, back])
        bm.faces.new([tip, back, left])
        bm.faces.new([base, front, left])
        bm.faces.new([base, right, front])
        bm.faces.new([base, back, right])
        bm.faces.new([base, left, back])
        
        # 持ち手（細い四角柱）
        hw = 0.008
        hl = 0.08
        h_verts = []
        for z in [0, -hl]:
            for x, y in [(-hw, -hw), (hw, -hw), (hw, hw), (-hw, hw)]:
                h_verts.append(bm.verts.new((x, y, z)))
        
        # 持ち手の面
        bm.faces.new([h_verts[0], h_verts[1], h_verts[2], h_verts[3]])
        bm.faces.new([h_verts[4], h_verts[7], h_verts[6], h_verts[5]])
        for i in range(4):
            j = (i + 1) % 4
            bm.faces.new([h_verts[i], h_verts[j], h_verts[j+4], h_verts[i+4]])
        
        # リング（トーラス代わりの円）
        rr = self.ring_radius
        ring_center_z = -hl - rr
        segments = 12
        inner_r = rr * 0.6
        outer_r = rr
        
        inner_verts = []
        outer_verts = []
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            ix = inner_r * math.cos(angle)
            iy = inner_r * math.sin(angle)
            ox = outer_r * math.cos(angle)
            oy = outer_r * math.sin(angle)
            inner_verts.append(bm.verts.new((ix, iy, ring_center_z)))
            outer_verts.append(bm.verts.new((ox, oy, ring_center_z)))
        
        for i in range(segments):
            j = (i + 1) % segments
            bm.faces.new([inner_verts[i], outer_verts[i], outer_verts[j], inner_verts[j]])
        
        bm.to_mesh(mesh)
        bm.free()
        
        obj = bpy.data.objects.new("Kunai", mesh)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        
        return obj



# === AI Generated ===

class MagicStaff:
    """ねじれた木の杖 - 先端にクリスタル"""
    
    def __init__(self, length=1.8, twist_turns=3):
        self.length = length
        self.twist_turns = twist_turns
    
    def create(self):
        import bpy
        import bmesh
        import math
        
        bm = bmesh.new()
        mesh = bpy.data.meshes.new("MagicStaffMesh")
        
        # === ねじれた木の杖（螺旋構造）===
        segments = 40
        radius = 0.025
        
        # 螺旋に沿った頂点リング
        rings = []
        for i in range(segments + 1):
            t = i / segments
            z = t * self.length
            
            # 螺旋の回転
            twist_angle = t * self.twist_turns * 2 * math.pi
            
            # 8角形の断面
            ring_verts = []
            for j in range(8):
                angle = j * math.pi / 4 + twist_angle
                # 先端に向かって細くなる
                r = radius * (1.0 - t * 0.5)
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                ring_verts.append(bm.verts.new((x, y, z)))
            rings.append(ring_verts)
        
        # リング間を面で繋ぐ
        for i in range(len(rings) - 1):
            for j in range(8):
                j_next = (j + 1) % 8
                bm.faces.new([
                    rings[i][j], rings[i][j_next],
                    rings[i+1][j_next], rings[i+1][j]
                ])
        
        # 底面を閉じる
        bm.faces.new(rings[0])
        
        # === クリスタル（八面体 - 浮遊）===
        crystal_z = self.length + 0.15  # 杖の先端から浮かせる
        crystal_size = 0.08
        
        # 八面体の頂点
        crystal_top = bm.verts.new((0, 0, crystal_z + crystal_size))
        crystal_bot = bm.verts.new((0, 0, crystal_z - crystal_size))
        crystal_mid = []
        for i in range(4):
            angle = i * math.pi / 2 + math.pi / 4
            x = crystal_size * 0.7 * math.cos(angle)
            y = crystal_size * 0.7 * math.sin(angle)
            crystal_mid.append(bm.verts.new((x, y, crystal_z)))
        
        # 八面体の面
        for i in range(4):
            j = (i + 1) % 4
            bm.faces.new([crystal_top, crystal_mid[i], crystal_mid[j]])
            bm.faces.new([crystal_bot, crystal_mid[j], crystal_mid[i]])
        
        bm.to_mesh(mesh)
        bm.free()
        
        obj = bpy.data.objects.new("MagicStaff", mesh)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        
        return obj



# === AI Generated ===

class TurretBase:
    """SF防衛タレット - 土台"""
    
    def create(self):
        import bpy
        import bmesh
        import math
        
        bm = bmesh.new()
        mesh = bpy.data.meshes.new("TurretBaseMesh")
        
        # 八角形の重厚な土台
        segments = 8
        radius_outer = 0.6
        radius_inner = 0.4
        height = 0.25
        
        # 外周（上下）
        outer_bottom = []
        outer_top = []
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            x = radius_outer * math.cos(angle)
            y = radius_outer * math.sin(angle)
            outer_bottom.append(bm.verts.new((x, y, 0)))
            outer_top.append(bm.verts.new((x, y, height)))
        
        # 中央の穴（上面のみ）
        inner_top = []
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            x = radius_inner * math.cos(angle)
            y = radius_inner * math.sin(angle)
            inner_top.append(bm.verts.new((x, y, height)))
        
        # 底面
        bm.faces.new(outer_bottom)
        
        # 側面
        for i in range(segments):
            j = (i + 1) % segments
            bm.faces.new([outer_bottom[i], outer_bottom[j], outer_top[j], outer_top[i]])
        
        # 上面（リング状）
        for i in range(segments):
            j = (i + 1) % segments
            bm.faces.new([outer_top[i], inner_top[i], inner_top[j], outer_top[j]])
        
        # 内側の壁
        for i in range(segments):
            j = (i + 1) % segments
            bm.faces.new([inner_top[i], bm.verts.new((inner_top[i].co[0], inner_top[i].co[1], 0.1)),
                          bm.verts.new((inner_top[j].co[0], inner_top[j].co[1], 0.1)), inner_top[j]])
        
        bm.to_mesh(mesh)
        bm.free()
        
        obj = bpy.data.objects.new("Turret_Base", mesh)
        bpy.context.collection.objects.link(obj)
        return obj


class TurretBody:
    """SF防衛タレット - 回転本体"""
    
    def create(self, parent=None):
        import bpy
        import bmesh
        import math
        
        bm = bmesh.new()
        mesh = bpy.data.meshes.new("TurretBodyMesh")
        
        # 回転する本体（ドーム型）
        segments = 16
        rings = 5
        radius = 0.35
        height = 0.3
        
        # 半球の頂点
        verts = []
        for ring in range(rings + 1):
            phi = (math.pi / 2) * (ring / rings)
            r = radius * math.cos(phi)
            z = height * math.sin(phi) + 0.05
            
            ring_verts = []
            for seg in range(segments):
                theta = 2 * math.pi * seg / segments
                x = r * math.cos(theta)
                y = r * math.sin(theta)
                ring_verts.append(bm.verts.new((x, y, z)))
            verts.append(ring_verts)
        
        # 面を作成
        for ring in range(rings):
            for seg in range(segments):
                seg_next = (seg + 1) % segments
                if ring < rings - 1:
                    bm.faces.new([
                        verts[ring][seg], verts[ring][seg_next],
                        verts[ring+1][seg_next], verts[ring+1][seg]
                    ])
                else:
                    # 頂点に向かう三角形
                    top = bm.verts.new((0, 0, height + 0.1))
                    bm.faces.new([verts[ring][seg], verts[ring][seg_next], top])
        
        # 底面
        bm.faces.new(verts[0])
        
        # 砲身マウント（前方に突き出た部分）
        mount_z = 0.2
        mount_depth = 0.2
        mount_size = 0.15
        
        m_verts = [
            bm.verts.new((-mount_size, -radius, mount_z - mount_size/2)),
            bm.verts.new((mount_size, -radius, mount_z - mount_size/2)),
            bm.verts.new((mount_size, -radius, mount_z + mount_size/2)),
            bm.verts.new((-mount_size, -radius, mount_z + mount_size/2)),
            bm.verts.new((-mount_size, -radius - mount_depth, mount_z - mount_size/2)),
            bm.verts.new((mount_size, -radius - mount_depth, mount_z - mount_size/2)),
            bm.verts.new((mount_size, -radius - mount_depth, mount_z + mount_size/2)),
            bm.verts.new((-mount_size, -radius - mount_depth, mount_z + mount_size/2)),
        ]
        
        # マウントの面
        bm.faces.new([m_verts[0], m_verts[1], m_verts[2], m_verts[3]])
        bm.faces.new([m_verts[4], m_verts[7], m_verts[6], m_verts[5]])
        bm.faces.new([m_verts[0], m_verts[4], m_verts[5], m_verts[1]])
        bm.faces.new([m_verts[2], m_verts[6], m_verts[7], m_verts[3]])
        bm.faces.new([m_verts[0], m_verts[3], m_verts[7], m_verts[4]])
        bm.faces.new([m_verts[1], m_verts[5], m_verts[6], m_verts[2]])
        
        bm.to_mesh(mesh)
        bm.free()
        
        obj = bpy.data.objects.new("Turret_Body", mesh)
        bpy.context.collection.objects.link(obj)
        obj.location = (0, 0, 0.25)  # ベースの上
        
        if parent:
            obj.parent = parent
        
        return obj


class TurretCannons:
    """SF防衛タレット - 二連装砲身"""
    
    def create(self, parent=None):
        import bpy
        import math
        
        # 二連装砲身を作成
        cannon_length = 0.8
        cannon_radius = 0.05
        spacing = 0.12
        
        cannons = []
        for side in [-1, 1]:
            bpy.ops.mesh.primitive_cylinder_add(
                radius=cannon_radius,
                depth=cannon_length,
                location=(side * spacing, -0.6 - cannon_length/2, 0.2)
            )
            cannon = bpy.context.active_object
            cannon.rotation_euler = (math.radians(90), 0, 0)
            bpy.ops.object.transform_apply(rotation=True)
            cannons.append(cannon)
        
        # 砲身を結合
        bpy.ops.object.select_all(action='DESELECT')
        for c in cannons:
            c.select_set(True)
        bpy.context.view_layer.objects.active = cannons[0]
        bpy.ops.object.join()
        
        obj = bpy.context.active_object
        obj.name = "Turret_Cannons"
        
        # 回転の原点を砲身の根元に
        bpy.context.scene.cursor.location = (0, -0.55, 0.2)
        bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
        
        if parent:
            obj.parent = parent
            obj.location = (0, -0.55, 0.2)  # 親に対するローカル座標
        
        return obj


class SciFiTurret:
    """SF防衛タレット - 完成品（階層構造 + アニメーション）"""
    
    def create(self, animate=True, frame_end=60):
        import bpy
        import math
        
        # 各パーツを生成
        base = TurretBase().create()
        body = TurretBody().create(parent=base)
        cannons = TurretCannons().create(parent=body)
        
        if animate:
            # アニメーション設定
            bpy.context.scene.frame_start = 1
            bpy.context.scene.frame_end = frame_end
            
            # Body: Y軸回転（左右首振り）
            body.rotation_mode = 'XYZ'
            body.keyframe_insert(data_path='rotation_euler', frame=1)
            
            body.rotation_euler = (0, 0, math.radians(45))
            body.keyframe_insert(data_path='rotation_euler', frame=15)
            
            body.rotation_euler = (0, 0, math.radians(-45))
            body.keyframe_insert(data_path='rotation_euler', frame=45)
            
            body.rotation_euler = (0, 0, 0)
            body.keyframe_insert(data_path='rotation_euler', frame=60)
            
            # Cannons: X軸回転（上下仰角）
            cannons.rotation_mode = 'XYZ'
            cannons.keyframe_insert(data_path='rotation_euler', frame=1)
            
            cannons.rotation_euler = (math.radians(-20), 0, 0)
            cannons.keyframe_insert(data_path='rotation_euler', frame=20)
            
            cannons.rotation_euler = (math.radians(15), 0, 0)
            cannons.keyframe_insert(data_path='rotation_euler', frame=40)
            
            cannons.rotation_euler = (0, 0, 0)
            cannons.keyframe_insert(data_path='rotation_euler', frame=60)
            
            # イージングカーブを設定（Blender 5.0 API）
            for obj in [body, cannons]:
                if obj.animation_data and obj.animation_data.action:
                    action = obj.animation_data.action
                    # Blender 5.0: fcurvesはaction.layers[0].strips[0].channelbags[0].channels で取得
                    try:
                        for layer in action.layers:
                            for strip in layer.strips:
                                for bag in strip.channelbags:
                                    for channel in bag.channels:
                                        for kf in channel.keyframes:
                                            kf.interpolation = 'BEZIER'
                    except:
                        pass  # アニメーションは動くので、イージングは省略可
        
        return {'base': base, 'body': body, 'cannons': cannons}

