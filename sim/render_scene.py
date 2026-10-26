"""用 Blender 渲染擬真的拼裝過程影格（含人手），給 BrickPilot 當合成測試畫面。

blender -b -P sim/render_scene.py -- --model house.json --hand <human_base_meshes_bundle.blend> --out sim/frames [--fast 1] [--limit N] [--start N]
輸出：<out>/0000.png ...、<out>/../calib.json（底板四角在畫面上的像素座標）
--fast 1 用 EEVEE 快速預覽；預設 Cycles（GPU、降噪、CC0 室內 HDRI 環境光）。
"""
import json
import math
import random
import sys
from pathlib import Path

import bmesh
import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
args = dict(zip(argv[::2], argv[1::2]))
model = json.load(open(args["--model"]))
out = Path(args["--out"])
out.mkdir(parents=True, exist_ok=True)
FAST = args.get("--fast") == "1"

W, H = model["grid"]
PLATE_T = 0.4       # 1 單位 = 1 顆凸點間距（8 mm）
BRICK_H = 1.2
RES = (1280, 720)
SHOULDER = Vector((W / 2 + 4, -45, 30))   # 操作者的肩膀（畫面下方外），決定手臂方向
REST = Vector((W / 2 + 6, -16, 10))       # 手休息的位置（畫面外）
random.seed(3)

scene = bpy.context.scene
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)


def srgb_to_linear(c):
    c = c / 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def principled(name, rgb, rough, bump=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    p = nt.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (*rgb, 1)
    p.inputs["Roughness"].default_value = rough
    if bump:   # 細微表面起伏，避免完美 CG 感
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 60
        b = nt.nodes.new("ShaderNodeBump")
        b.inputs["Strength"].default_value = bump
        nt.links.new(noise.outputs["Fac"], b.inputs["Height"])
        nt.links.new(b.outputs["Normal"], p.inputs["Normal"])
    return m, p


def plastic(name, bgr):
    m, p = principled(name, [srgb_to_linear(v) for v in bgr[::-1]], 0.22, bump=0.03)
    p.inputs["IOR"].default_value = 1.5
    return m


def studded_box(name, w, d, h, mat):
    """原點在左下角底部的方塊加凸點，邊緣倒角。"""
    bm = bmesh.new()
    r = bmesh.ops.create_cube(bm, size=1)
    bmesh.ops.scale(bm, vec=(w - 0.04, d - 0.04, h), verts=r["verts"])
    bmesh.ops.translate(bm, vec=(w / 2, d / 2, h / 2), verts=r["verts"])
    for i in range(w):
        for j in range(d):
            c = bmesh.ops.create_cone(bm, cap_ends=True, segments=24, radius1=0.3, radius2=0.3, depth=0.2)
            bmesh.ops.translate(bm, vec=(i + 0.5, j + 0.5, h + 0.1), verts=c["verts"])
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    scene.collection.objects.link(obj)
    obj.data.materials.append(mat)
    bev = obj.modifiers.new("bevel", "BEVEL")
    bev.width, bev.segments = 0.05, 3
    return obj


# ---------- 場景 ----------
mats = {k: plastic(k, v) for k, v in model["palette"].items()}
studded_box("baseplate", W, H, PLATE_T, mats["base"])

table = bpy.data.meshes.new("table")
bm = bmesh.new()
bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=80)
bm.to_mesh(table)
bm.free()
t = bpy.data.objects.new("table", table)
t.location = (W / 2, H / 2, -0.01)
scene.collection.objects.link(t)
wood, wp = principled("wood", (0.3, 0.17, 0.08), 0.55)
nt = wood.node_tree
coord = nt.nodes.new("ShaderNodeTexCoord")
grain = nt.nodes.new("ShaderNodeMapping")   # 沿 x 拉長的雜訊 = 木紋
grain.inputs["Scale"].default_value = (0.03, 0.8, 1)
wave = nt.nodes.new("ShaderNodeTexNoise")
wave.inputs["Scale"].default_value = 3
wave.inputs["Detail"].default_value = 10
wave.inputs["Distortion"].default_value = 1.5
nt.links.new(coord.outputs["Object"], grain.inputs["Vector"])
nt.links.new(grain.outputs["Vector"], wave.inputs["Vector"])
ramp = nt.nodes.new("ShaderNodeValToRGB")
ramp.color_ramp.elements[0].color = (0.22, 0.12, 0.06, 1)
ramp.color_ramp.elements[1].color = (0.45, 0.28, 0.15, 1)
nt.links.new(wave.outputs["Fac"], ramp.inputs["Fac"])
nt.links.new(ramp.outputs["Color"], wp.inputs["Base Color"])
table.materials.append(wood)

cam_data = bpy.data.cameras.new("cam")
cam_data.lens = 99
cam = bpy.data.objects.new("cam", cam_data)
cam.location = (W / 2, H / 2, 110)     # 正上方、拉高：疊高積木的視差較小（真實架設也建議如此）
scene.collection.objects.link(cam)
scene.camera = cam

# 主光：桌燈般的大面光源，從左上斜照，陰影柔和；環境光：Blender 內附 CC0 室內 HDRI
key = bpy.data.objects.new("key", bpy.data.lights.new("key", "AREA"))
key.data.shape, key.data.size = "DISK", 40
key.data.energy = 1.6e5
key.data.color = (1.0, 0.95, 0.88)
key.location = (-30, 30, 70)
key.rotation_euler = (Vector((W / 2, H / 2, 0)) - key.location).to_track_quat("-Z", "Y").to_euler()
scene.collection.objects.link(key)

world = bpy.data.worlds.new("world")
scene.world = world
world.use_nodes = True
env = world.node_tree.nodes.new("ShaderNodeTexEnvironment")
env.image = bpy.data.images.load(str(Path(bpy.utils.system_resource("DATAFILES")) / "studiolights/world/interior.exr"))
world.node_tree.links.new(env.outputs["Color"], world.node_tree.nodes["Background"].inputs["Color"])
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.35

if FAST:
    for engine in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        try:
            scene.render.engine = engine
            break
        except TypeError:
            pass
else:
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.refresh_devices()
    for d in prefs.devices:
        d.use = d.type == "METAL"
    scene.cycles.device = "GPU"
    scene.cycles.samples = int(args.get("--samples", 64))
    scene.cycles.use_denoising = True
    scene.render.use_persistent_data = True   # 逐格渲染時不重建場景，省時間
scene.render.resolution_x, scene.render.resolution_y = RES
scene.view_settings.view_transform = "AgX"
scene.view_settings.look = "AgX - Medium High Contrast"


# ---------- 人手：Blender Studio「Human Base Meshes」寫實手部模型（CC0），自動綁骨架擺成捏取手勢 ----------
HAND_POSE = {  # 每節繞 X 彎向掌心（度）；拇指第一節另加 Y、Z（對向食指）
    "index": [35, 45, 25], "middle": [22, 35, 20], "ring": [28, 40, 25], "pinky": [32, 45, 30],
}
THUMB_POSE = [(40, -20, 0), (15, 0, 0), (0, 0, 0)]
WRIST_TILT = math.radians(35)   # 前臂往肩膀方向抬起的角度


def make_hand():
    """原點 = 拇指與食指捏住積木的位置；手指朝 +y，前臂往 -y 斜上離開畫面。"""
    with bpy.data.libraries.load(args["--hand"]) as (_, dst):
        dst.objects = ["Hand  - Realistic"]
    h = dst.objects[0]
    scene.collection.objects.link(h)
    h.location = (0, 0, 0)
    bpy.context.view_layer.objects.active = h
    bpy.ops.object.modifier_apply(modifier=h.modifiers[0].name)   # 套用 Multires 細節
    # 模型座標（公尺）：手指朝 -z、掌心朝 +y、拇指在 +x（右手）。手腕切口封面拿掉，往上擠出成前臂
    bm = bmesh.new()
    bm.from_mesh(h.data)
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.normal.z > 0.9 and f.calc_center_median().z > 0.07], context="FACES_ONLY")
    loop = [e for e in bm.edges if e.is_boundary]
    for _ in range(6):
        r = bmesh.ops.extrude_edge_only(bm, edges=loop)
        vs = [g for g in r["geom"] if isinstance(g, bmesh.types.BMVert)]
        c = sum((v.co for v in vs), Vector()) / len(vs)
        for v in vs:
            v.co = c + (v.co - c) * 1.05 + Vector((0, 0, 0.07))
        loop = [g for g in r["geom"] if isinstance(g, bmesh.types.BMEdge) and g.is_boundary]
    bm.to_mesh(h.data)
    bm.free()

    rig = bpy.data.objects.new("rig", bpy.data.armatures.new("rig"))
    scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    eb = rig.data.edit_bones

    def bone(name, a, b, parent=None):
        x = eb.new(name)
        x.head, x.tail = Vector(a), Vector(b)
        x.align_roll(Vector((0, 1, 0)))   # 骨頭 Z 軸朝掌心：繞 X 轉 = 手指彎曲
        if parent:
            x.parent = eb[parent]

    bone("forearm", (0.005, -0.004, 0.5), (0.005, -0.004, 0.0))
    bone("palm", (0.005, -0.004, 0.0), (0.019, 0.004, -0.083), "forearm")
    fingers = {  # 指根關節 → 指尖（從模型頂點量得）
        "index": ((0.046, 0.003, -0.082), (0.0539, 0.0126, -0.1893)), "middle": ((0.019, 0.004, -0.085), (0.0183, 0.0138, -0.2047)),
        "ring": ((-0.011, 0.004, -0.083), (-0.0115, 0.0206, -0.1938)), "pinky": ((-0.040, 0.006, -0.078), (-0.0533, 0.0206, -0.1657)),
    }
    for f, (a, t) in fingers.items():
        a, t = Vector(a), Vector(t)
        pip, dip = a.lerp(t, 0.47), a.lerp(t, 0.76)
        bone(f + "1", a, pip, "palm")
        bone(f + "2", pip, dip, f + "1")
        bone(f + "3", dip, t, f + "2")
    bone("thumb1", (0.030, 0.008, -0.020), (0.066, 0.017, -0.066), "palm")
    bone("thumb2", (0.066, 0.017, -0.066), (0.080, 0.021, -0.090), "thumb1")
    bone("thumb3", (0.080, 0.021, -0.090), (0.090, 0.022, -0.114), "thumb2")
    bpy.ops.object.mode_set(mode="OBJECT")
    for o in scene.objects:
        o.select_set(False)
    h.select_set(True)
    rig.select_set(True)
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")

    for f, angles in HAND_POSE.items():
        for i, a in enumerate(angles):
            rig.pose.bones[f"{f}{i + 1}"].rotation_mode = "XYZ"
            rig.pose.bones[f"{f}{i + 1}"].rotation_euler = (math.radians(a), 0, 0)
    for i, rot in enumerate(THUMB_POSE):
        rig.pose.bones[f"thumb{i + 1}"].rotation_mode = "XYZ"
        rig.pose.bones[f"thumb{i + 1}"].rotation_euler = [math.radians(a) for a in rot]
    bpy.context.view_layer.update()
    pinch = (rig.pose.bones["index3"].tail + rig.pose.bones["thumb3"].tail) / 2
    # 姿勢定型：套用骨架變形後刪掉骨架
    bpy.context.view_layer.objects.active = h
    h.parent = None
    bpy.ops.object.modifier_apply(modifier=h.modifiers[-1].name)
    bpy.data.objects.remove(rig)
    # 轉到場景座標（1 單位 = 8 mm）：掌心朝下、手指朝 +y、前臂往 -y 抬起；原點在捏取點上方一點（積木中段高度）
    to_scene = Matrix.Rotation(-WRIST_TILT, 4, "X") @ Matrix(((-1, 0, 0, 0), (0, 0, -1, 0), (0, -1, 0, 0), (0, 0, 0, 1))) @ Matrix.Scale(110, 4)   # 模型是偏大的男性手，縮到約 18 cm 長的一般成人手
    h.data.transform(Matrix.Translation((0, 0, BRICK_H * 0.6)) @ to_scene @ Matrix.Translation(-pinch))
    h.vertex_groups.clear()
    sub = h.modifiers.new("sub", "SUBSURF")
    sub.levels = sub.render_levels = 1
    for p in h.data.polygons:
        p.use_smooth = True
    skin, sp = principled("skin", (0.62, 0.36, 0.25), 0.45, bump=0.05)
    snt = skin.node_tree
    blotch = snt.nodes.new("ShaderNodeTexNoise")
    blotch.inputs["Scale"].default_value = 3
    tone = snt.nodes.new("ShaderNodeValToRGB")
    tone.color_ramp.elements[0].color = (0.52, 0.28, 0.19, 1)
    tone.color_ramp.elements[1].color = (0.66, 0.4, 0.29, 1)
    snt.links.new(blotch.outputs["Fac"], tone.inputs["Fac"])
    ao = snt.nodes.new("ShaderNodeAmbientOcclusion")   # 指縫、指節皺褶變暗，補回立體感
    ao.inputs["Distance"].default_value = 0.6
    shade = snt.nodes.new("ShaderNodeMapRange")
    shade.inputs["To Min"].default_value = 0.6
    snt.links.new(ao.outputs["AO"], shade.inputs["Value"])
    mul = snt.nodes.new("ShaderNodeMix")
    mul.data_type, mul.blend_type = "RGBA", "MULTIPLY"
    mul.inputs["Factor"].default_value = 1
    snt.links.new(tone.outputs["Color"], mul.inputs["A"])
    snt.links.new(shade.outputs["Result"], mul.inputs["B"])
    snt.links.new(mul.outputs["Result"], sp.inputs["Base Color"])
    pores = snt.nodes.new("ShaderNodeTexVoronoi")      # 毛孔
    pores.inputs["Scale"].default_value = 90
    pb = snt.nodes.new("ShaderNodeBump")
    pb.inputs["Strength"].default_value = 0.12
    pb.inputs["Distance"].default_value = 0.02
    snt.links.new(pores.outputs["Distance"], pb.inputs["Height"])
    snt.links.new(pb.outputs["Normal"], sp.inputs["Normal"])
    sp.inputs["Subsurface Weight"].default_value = 0.3
    sp.inputs["Subsurface Radius"].default_value = (1.0, 0.35, 0.2)
    sp.inputs["Subsurface Scale"].default_value = 0.4
    h.data.materials.clear()
    h.data.materials.append(skin)
    return h


hand = make_hand()


def set_hand(pos):
    """把手放到 pos（捏取點），並讓前臂指向肩膀。"""
    d = SHOULDER - pos
    hand.location = pos
    hand.rotation_euler = (0, 0, math.atan2(d.x, -d.y))


# ---------- 積木與零件堆 ----------
heights = [[0.0] * W for _ in range(H)]


def target_of(at):
    x, y, w, h = at
    z = max(heights[r][c] for r in range(y, y + h) for c in range(x, x + w))
    for r in range(y, y + h):
        for c in range(x, x + w):
            heights[r][c] = z + BRICK_H
    return Vector((x + w / 2, H - y - h / 2, PLATE_T + z))   # 積木底面中心


def pile_slot(i, w, d):
    """零件堆分在底板左右兩側；長邊橫放，加一點隨機偏移與角度。回傳底面中心與旋轉角。"""
    side, j = i % 2, i // 2
    col, row = j % 2, j // 2
    x0 = W + 3.5 if side == 0 else -8.5
    rot = random.uniform(-0.25, 0.25) + (math.pi / 2 if d > w else 0)
    return Vector((x0 + col * 4.8 + random.uniform(-0.3, 0.3), 2.5 + row * 3.4 + random.uniform(-0.2, 0.2), 0)), rot


def place(b, center, rot):
    """以底面中心 + 旋轉角擺放積木（積木原點在左下角）。"""
    half = Matrix.Rotation(rot, 3, "Z") @ Vector((b["w"] / 2, b["d"] / 2, 0))
    b["obj"].location = center - half
    b["obj"].rotation_euler = (0, 0, rot)


def new_brick(name, w, d, color, slot, target):
    o = studded_box(name, w, d, BRICK_H, mats[color])
    b = {"obj": o, "w": w, "d": d, "target": target}
    b["pile"], b["rot"] = pile_slot(slot, w, d)
    place(b, b["pile"], b["rot"])
    return b


bricks = [new_brick(f"b{i}", s["at"][2], s["at"][3], s["color"], i, target_of(s["at"]))
          for i, s in enumerate(model["steps"])]
wrong = new_brick("wrong", 2, 2, "blue", len(bricks), Vector((13, H - 13, PLATE_T)))

# ---------- 底板四角投影成像素座標，給追蹤器用 ----------
bpy.context.view_layer.update()   # 用 data API 建的物件要更新後 matrix_world 才正確
corners = []
for x, y in [(0, H), (W, H), (W, 0), (0, 0)]:
    p = world_to_camera_view(scene, cam, Vector((x, y, PLATE_T)))
    corners.append([round(p.x * RES[0], 1), round((1 - p.y) * RES[1], 1)])
json.dump(corners, open(out.parent / "calib.json", "w"))


# ---------- 動畫 ----------
def ease(t):
    return t * t * (3 - 2 * t)


def arc(a, b, n, lift, each):
    for f in range(n):
        t = ease(f / (n - 1))
        each(a.lerp(b, t) + Vector((0, 0, lift * math.sin(math.pi * t))), t)
        yield


def hold(n):
    for _ in range(n):
        yield


hand_pos = REST.copy()


def move(b, a, a_rot, z_to, to, to_rot):
    """手伸去拿積木 → 搬到目的地放下 → 收手到畫面外。"""
    global hand_pos
    yield from arc(hand_pos, a, 14, 5, lambda p, t: set_hand(p))
    yield from hold(3)
    yield from arc(a, to, 18, 5, lambda p, t: (set_hand(p), place(b, p, a_rot + (to_rot - a_rot) * t)))
    yield from hold(3)
    rest = Vector((to.x * 0.5 + REST.x * 0.5, REST.y, REST.z))
    yield from arc(to, rest, 10, 3, lambda p, t: set_hand(p))
    hand_pos = rest
    yield from hold(12)


def timeline():
    set_hand(REST)
    yield from hold(12)
    for i, b in enumerate(bricks):
        yield from move(b, b["pile"], b["rot"], 0, b["target"], 0)
        if i == 2:          # 放完 3 塊後，故意把藍色 2x2 放錯再拿回零件堆
            yield from move(wrong, wrong["pile"], wrong["rot"], 0, wrong["target"], 0)
            yield from hold(18)
            yield from move(wrong, wrong["target"], 0, 0, wrong["pile"], wrong["rot"])
    yield from hold(36)


limit = int(args.get("--limit", 10 ** 6))
start = int(args.get("--start", 0))     # 試渲染用：只輸出 [start, start+limit) 這幾格
n = 0
for _ in timeline():
    if n >= start + limit:
        break
    if n >= start:
        scene.render.filepath = str(out / f"{n:04d}.png")
        bpy.ops.render.render(write_still=True)
    n += 1
print(f"rendered frames {start}..{n - 1} to {out}")
