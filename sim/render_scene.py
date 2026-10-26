"""用 Blender 渲染擬真的拼裝過程影格，給 BrickPilot 當合成測試畫面。

blender -b -P sim/render_scene.py -- --model house.json --out sim/frames
輸出：<out>/0000.png ...、<out>/../calib.json（底板四角在畫面上的像素座標）
"""
import json
import math
import random
import sys
from pathlib import Path

import bmesh
import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

args = dict(zip(sys.argv[sys.argv.index("--") + 1::2], sys.argv[sys.argv.index("--") + 2::2]))
model = json.load(open(args["--model"]))
out = Path(args["--out"])
out.mkdir(parents=True, exist_ok=True)

W, H = model["grid"]
PLATE_T = 0.4       # 1 單位 = 1 顆凸點間距（8 mm）
BRICK_H = 1.2
FLY, HOLD = 16, 18  # 每塊積木飛行、停留的影格數
RES = (1280, 720)
random.seed(3)

scene = bpy.context.scene
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)


def srgb_to_linear(c):
    c = c / 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def plastic(name, bgr, rough=0.32):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (*[srgb_to_linear(v) for v in bgr[::-1]], 1)
    p.inputs["Roughness"].default_value = rough
    return m


def studded_box(name, w, d, h, mat):
    """原點在左下角底部的方塊加凸點，邊緣倒角。"""
    bm = bmesh.new()
    r = bmesh.ops.create_cube(bm, size=1)
    bmesh.ops.scale(bm, vec=(w - 0.04, d - 0.04, h), verts=r["verts"])
    bmesh.ops.translate(bm, vec=(w / 2, d / 2, h / 2), verts=r["verts"])
    for i in range(w):
        for j in range(d):
            c = bmesh.ops.create_cone(bm, cap_ends=True, segments=20, radius1=0.3, radius2=0.3, depth=0.2)
            bmesh.ops.translate(bm, vec=(i + 0.5, j + 0.5, h + 0.1), verts=c["verts"])
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    scene.collection.objects.link(obj)
    obj.data.materials.append(mat)
    bev = obj.modifiers.new("bevel", "BEVEL")
    bev.width, bev.segments = 0.04, 2
    return obj


# ---------- 場景 ----------
mats = {k: plastic(k, v) for k, v in model["palette"].items()}
studded_box("baseplate", W, H, PLATE_T, mats["base"])

table = bpy.data.meshes.new("table")
bm = bmesh.new()
bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=60)
bm.to_mesh(table)
bm.free()
t = bpy.data.objects.new("table", table)
t.location = (W, H / 2, -0.01)
scene.collection.objects.link(t)
wood = bpy.data.materials.new("wood")
wood.use_nodes = True
nt = wood.node_tree
wave = nt.nodes.new("ShaderNodeTexWave")
wave.inputs["Scale"].default_value = 4
wave.inputs["Distortion"].default_value = 3
wave.inputs["Detail"].default_value = 4
ramp = nt.nodes.new("ShaderNodeValToRGB")
ramp.color_ramp.elements[0].color = (0.32, 0.18, 0.09, 1)
ramp.color_ramp.elements[1].color = (0.55, 0.36, 0.2, 1)
nt.links.new(wave.outputs["Fac"], ramp.inputs["Fac"])
nt.links.new(ramp.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
nt.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.6
table.materials.append(wood)

cam_data = bpy.data.cameras.new("cam")
cam_data.lens = 99
cam = bpy.data.objects.new("cam", cam_data)
cam.location = (W / 2, H / 2, 110)     # 正上方、拉高：疊高積木的視差較小（真實架設也建議如此）
scene.collection.objects.link(cam)
scene.camera = cam

sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy, sun.data.angle = 1.7, math.radians(25)   # 模擬頂部環形燈：光幾乎從正上方來、陰影短而柔和
sun.rotation_euler = (math.radians(12), math.radians(-8), 0)
scene.collection.objects.link(sun)
scene.world = bpy.data.worlds.new("world")
scene.world.use_nodes = True
scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.45

for engine in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
    try:
        scene.render.engine = engine
        break
    except TypeError:
        pass
scene.render.resolution_x, scene.render.resolution_y = RES
scene.view_settings.view_transform = "Standard"

# ---------- 積木與零件堆 ----------
heights = [[0.0] * W for _ in range(H)]


def target_of(at):
    x, y, w, h = at
    z = max(heights[r][c] for r in range(y, y + h) for c in range(x, x + w))
    for r in range(y, y + h):
        for c in range(x, x + w):
            heights[r][c] = z + BRICK_H
    return Vector((x, H - y - h, PLATE_T + z))


def pile_slot(i, w, d):
    """零件堆分在底板左右兩側，各 2 欄 4 列；長邊一律橫放（直的就轉 90 度），加一點隨機偏移與角度。"""
    side, j = i % 2, i // 2
    col, row = j % 2, j // 2
    x0 = W + 1.5 if side == 0 else -10.5
    x, y = x0 + col * 4.8 + random.uniform(-0.3, 0.3), 1.5 + row * 3.4 + random.uniform(-0.2, 0.2)
    rot = random.uniform(-0.25, 0.25)
    if d > w:
        rot += math.pi / 2
        x += d          # 以左下角為軸轉 90 度後，積木會往 -x 延伸
    return Vector((x, y, 0)), rot


bricks = []
for i, s in enumerate(model["steps"]):
    o = studded_box(f"b{i}", s["at"][2], s["at"][3], BRICK_H, mats[s["color"]])
    o.location, o.rotation_euler.z = pile_slot(i, s["at"][2], s["at"][3])
    bricks.append({"obj": o, "pile": o.location.copy(), "rot": o.rotation_euler.z, "target": target_of(s["at"])})

wrong = studded_box("wrong", 2, 2, BRICK_H, mats["blue"])
wrong.location, wrong.rotation_euler.z = pile_slot(len(bricks), 2, 2)
wrong_pile, wrong_rot = wrong.location.copy(), wrong.rotation_euler.z
wrong_target = Vector((12, H - 12 - 2, PLATE_T))

# ---------- 底板四角投影成像素座標，給追蹤器用 ----------
scene.render.resolution_percentage = 100
bpy.context.view_layer.update()   # 用 data API 建的物件要更新後 matrix_world 才正確
corners = []
for x, y in [(0, H), (W, H), (W, 0), (0, 0)]:
    p = world_to_camera_view(scene, cam, Vector((x, y, PLATE_T)))
    corners.append([round(p.x * RES[0], 1), round((1 - p.y) * RES[1], 1)])
json.dump(corners, open(out.parent / "calib.json", "w"))


# ---------- 動畫 ----------
def fly(obj, a, b, rot_a, rot_b):
    for f in range(FLY):
        t = f / (FLY - 1)
        t = t * t * (3 - 2 * t)
        obj.location = a.lerp(b, t) + Vector((0, 0, 6 * math.sin(math.pi * t)))
        obj.rotation_euler.z = rot_a + (rot_b - rot_a) * t
        yield


def hold(n):
    for _ in range(n):
        yield


def timeline():
    yield from hold(HOLD)
    for i, b in enumerate(bricks):
        yield from fly(b["obj"], b["pile"], b["target"], b["rot"], 0)
        yield from hold(HOLD)
        if i == 2:          # 放完 3 塊後，故意把藍色 2x2 放錯再拿掉
            yield from fly(wrong, wrong_pile, wrong_target, wrong_rot, 0)
            yield from hold(HOLD * 2)
            yield from fly(wrong, wrong_target, wrong_pile, 0, wrong_rot)
            yield from hold(HOLD // 2)
    yield from hold(HOLD * 2)


limit = int(args.get("--limit", 10 ** 6))   # 試渲染用
n = 0
for _ in timeline():
    if n >= limit:
        break
    scene.render.filepath = str(out / f"{n:04d}.png")
    bpy.ops.render.render(write_still=True)
    n += 1
print(f"rendered {n} frames to {out}")
