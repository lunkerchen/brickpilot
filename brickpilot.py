"""BrickPilot MVP：用俯視鏡頭追蹤底板上的積木拼裝進度，指引下一步並即時抓錯。

python brickpilot.py demo --model house.json --out demo.mp4   # 合成測試影片
python brickpilot.py run  --model house.json --source 0       # 網路攝影機（或影片路徑）
python brickpilot.py measure --source done.png --step 14 --out palette.json  # 量實際看到的顏色
"""
import argparse
import json
import subprocess
import sys

import cv2
import numpy as np

CELL = 24          # 校正後每顆凸點的像素數
STABLE = 5         # 觀測連續一致幾幀才採用（濾掉手的移動）
UNKNOWN = "?"      # 顏色離調色盤太遠（手、陰影）的格子，不參與判斷
MAX_DIST = 25      # 預設顏色容許距離；實拍燈光不同時用 --max-dist 調
LAB_W = np.array([0.5, 1, 1])  # 亮度權重減半：陰影只讓顏色變暗，不該把綠底板判成黑色


def load_model(path, palette_path=None):
    m = json.load(open(path))
    # 鏡頭看到的顏色會受燈光影響，用 measure 量出的 palette 檔取代說明書的標準色
    seen = json.load(open(palette_path)) if palette_path else m["palette"]
    m["lab"] = {k: cv2.cvtColor(np.uint8([[v]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
                for k, v in seen.items()}
    m["states"] = expected_states(m)
    return m


def expected_states(m):
    """回傳每一步完成後的俯視顏色格 states[0]=空底板 ... states[n]=完成。"""
    w, h = m["grid"]
    g = np.full((h, w), "base", dtype=object)
    states = [g.copy()]
    for i, s in enumerate(m["steps"]):
        x, y, bw, bh = s["at"]
        g[y:y + bh, x:x + bw] = s["color"]
        # ponytail: 只看俯視頂面，所以每一步都必須改變頂面顏色；同色疊高要加側面鏡頭（Stage II）
        assert (g != states[-1]).any(), f"step {i + 1} is invisible from the top"
        states.append(g.copy())
    return states


def rectify_matrix(corners, grid):
    """底板四角（左上、右上、右下、左下）→ 俯視格點影像的透視矩陣。"""
    w, h = grid
    dst = np.float32([[0, 0], [w * CELL, 0], [w * CELL, h * CELL], [0, h * CELL]])
    return cv2.getPerspectiveTransform(np.float32(corners), dst)


def observe(frame, H, m, max_dist):
    """把畫面轉成每格的顏色標籤。"""
    w, h = m["grid"]
    top = cv2.warpPerspective(frame, H, (w * CELL, h * CELL))
    lab = cv2.cvtColor(top, cv2.COLOR_BGR2LAB).astype(float)
    names = list(m["lab"])
    refs = np.array([m["lab"][n] for n in names])
    obs = np.empty((h, w), dtype=object)
    p = int(CELL * 0.3)   # 只取格子中央，疊高積木的視差偏移不會滲進來
    for y in range(h):
        for x in range(w):
            c = np.median(lab[y * CELL + p:(y + 1) * CELL - p, x * CELL + p:(x + 1) * CELL - p].reshape(-1, 3), 0)
            d = np.linalg.norm((refs - c) * LAB_W, axis=1)
            obs[y, x] = names[d.argmin()] if d.min() <= max_dist else UNKNOWN
    return obs


def locate(obs, states, current):
    """找出最符合的步驟；回傳 (步驟 k, 放錯的格子)。看不清楚時回傳 None。"""
    known = obs != UNKNOWN
    if known.mean() < 0.5:
        return None
    diffs = [int(((obs != s) & known).sum()) for s in states]
    # 平手（通常是遮擋造成）時，選離目前步驟最近的，避免被手擋住時亂跳
    k = min(range(len(states)), key=lambda i: (diffs[i], abs(i - current)))
    wrong = known & (obs != states[k])
    if k + 1 < len(states):
        wrong &= obs != states[k + 1]   # 下一塊放到一半不算錯
    return k, wrong


def message(m, k, wrong):
    n = len(m["steps"])
    if wrong.any():
        ys, xs = np.nonzero(wrong)
        want = m["states"][k][ys[0], xs[0]]
        return "Wrong brick: remove it" if want == "base" else f"Wrong brick: should be {want}"
    if k == n:
        return "Build complete!"
    s = m["steps"][k]
    return f"Step {k + 1}/{n}: place a {s['color']} {s['at'][2]}x{s['at'][3]} brick in the green box"


class Tracker:
    def __init__(self, m, max_dist):
        self.m, self.max_dist = m, max_dist
        self.k, self.wrong = 0, np.zeros(m["states"][0].shape, bool)
        self.last, self.count = None, 0

    def update(self, frame, H):
        obs = observe(frame, H, self.m, self.max_dist)
        if self.last is not None and (obs == self.last).all():
            self.count += 1
        else:
            self.last, self.count = obs, 1
        if self.count == STABLE:
            r = locate(obs, self.m["states"], self.k)
            if r:
                self.k, self.wrong = r
        return message(self.m, self.k, self.wrong)


def draw(frame, H, m, k, wrong, msg, label=None):
    """把下一步位置（綠框）與放錯的格子（紅框）投影回原始畫面。"""
    Hi = np.linalg.inv(H)

    def poly(x, y, w, h, color, t):
        pts = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]).reshape(-1, 1, 2) * CELL
        cv2.polylines(frame, [cv2.perspectiveTransform(pts, Hi).astype(np.int32)], True, color, t)

    gw, gh = m["grid"]
    poly(0, 0, gw, gh, (200, 200, 200), 1)
    if not wrong.any() and k < len(m["steps"]):
        poly(*m["steps"][k]["at"], (0, 255, 0), 3)
    for y, x in zip(*np.nonzero(wrong)):
        poly(x, y, 1, 1, (0, 0, 255), 2)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(frame, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if label:
        cv2.putText(frame, label, (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    return frame


# ---------- 合成測試影片 ----------

DEMO_SIZE = (960, 720)
DEMO_CORNERS = [(250, 110), (730, 130), (820, 650), (150, 620)]


def render(state, m, rng, hand=None):
    """畫一張斜拍的合成畫面：底板、積木、凸點、雜訊，可選擇加一隻移動的手。"""
    h, w = state.shape
    top = np.zeros((h * CELL, w * CELL, 3), np.uint8)
    for y in range(h):
        for x in range(w):
            c = np.array(m["palette"][state[y, x]], int)
            top[y * CELL:(y + 1) * CELL, x * CELL:(x + 1) * CELL] = c
            cv2.circle(top, (x * CELL + CELL // 2, y * CELL + CELL // 2), CELL // 3,
                       tuple(int(v) for v in np.clip(c + 25, 0, 255)), -1)
    H = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [w * CELL, 0], [w * CELL, h * CELL], [0, h * CELL]]), np.float32(DEMO_CORNERS))
    frame = np.full((DEMO_SIZE[1], DEMO_SIZE[0], 3), 110, np.uint8)
    warped = cv2.warpPerspective(top, H, DEMO_SIZE)
    mask = cv2.warpPerspective(np.full(top.shape[:2], 255, np.uint8), H, DEMO_SIZE) > 0
    frame[mask] = warped[mask]
    if hand:
        cv2.ellipse(frame, hand, (90, 55), 30, 0, 360, (140, 170, 220), -1)
    gain = rng.uniform(0.92, 1.08)
    noisy = frame.astype(float) * gain + rng.normal(0, 6, frame.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def demo_script(m):
    """產生 (畫面狀態, 是否有手) 序列：逐步拼裝，第 4 步故意放錯再改正。"""
    st = m["states"]
    for k in range(len(st)):
        for f in range(6):                       # 手伸進來放積木
            yield st[k - 1] if k else st[0], (300 + 40 * f, 300 + 20 * f)
        for _ in range(14):
            yield st[k], None
        if k == 3:                               # 放錯：藍色 2x2 放到空位
            bad = st[k].copy()
            bad[12:14, 12:14] = "blue"
            for _ in range(25):
                yield bad, None
            for f in range(6):                   # 拿掉
                yield bad, (600 - 30 * f, 500)


def demo(m, out, max_dist):
    rng = np.random.default_rng(0)
    H = rectify_matrix(DEMO_CORNERS, m["grid"])
    t = Tracker(m, max_dist)
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 10, DEMO_SIZE)
    for state, hand in demo_script(m):
        frame = render(state, m, rng, hand)
        msg = t.update(frame, H)
        vw.write(draw(frame, H, m, t.k, t.wrong, msg, "SYNTHETIC TEST FOOTAGE"))
    vw.release()
    print(f"wrote {out}; final step {t.k}/{len(m['steps'])}")


# ---------- 即時模式 ----------

def calibrate(cap):
    """在畫面上依序點底板四角：左上、右上、右下、左下。"""
    pts = []
    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", lambda e, x, y, *_: pts.append((x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)
    while len(pts) < 4:
        ok, frame = cap.read()
        if not ok:
            sys.exit("camera read failed")
        for p in pts:
            cv2.circle(frame, p, 6, (0, 0, 255), -1)
        cv2.putText(frame, "Click baseplate corners: TL, TR, BR, BL", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("calibrate", frame)
        if cv2.waitKey(30) == ord("q"):
            sys.exit()
    cv2.destroyWindow("calibrate")
    return pts


def measure(m, source, step, calib_path, out):
    """拍一張已知進度（第 step 步完成）的畫面，量出每種顏色實際看起來的 BGR。"""
    frame = cv2.imread(source)
    w, h = m["grid"]
    top = cv2.warpPerspective(frame, rectify_matrix(json.load(open(calib_path)), m["grid"]), (w * CELL, h * CELL))
    p, state, samples = CELL // 5, m["states"][step], {}
    for y in range(h):
        for x in range(w):
            samples.setdefault(state[y, x], []).append(
                np.median(top[y * CELL + p:(y + 1) * CELL - p, x * CELL + p:(x + 1) * CELL - p].reshape(-1, 3), 0))
    palette = {k: [int(v) for v in np.median(samples.get(k, [m["palette"][k]]), 0)] for k in m["palette"]}
    json.dump(palette, open(out, "w"), indent=1)
    print(palette)


def run(m, source, max_dist, voice, calib_path="calib.json", out=None):
    """out 有值時不開視窗，把標註後的畫面寫成影片（批次處理錄好的影片或影格）。"""
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    vw = None
    try:
        corners = json.load(open(calib_path))
    except FileNotFoundError:
        corners = calibrate(cap)
        json.dump(corners, open(calib_path, "w"))
    H = rectify_matrix(corners, m["grid"])
    t, spoken = Tracker(m, max_dist), None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        msg = t.update(frame, H)
        if out:
            vw = vw or cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS) or 24,
                                       (frame.shape[1], frame.shape[0]))
            vw.write(draw(frame, H, m, t.k, t.wrong, msg, "SIMULATED FOOTAGE (Blender)"))
            continue
        if voice and msg != spoken and sys.platform == "darwin":
            subprocess.Popen(["say", msg])
            spoken = msg
        cv2.imshow("BrickPilot", draw(frame, H, m, t.k, t.wrong, msg))
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        if key == ord("r"):                      # 重新校正
            H = rectify_matrix(calibrate(cap), m["grid"])
    if vw:
        vw.release()
        print(f"wrote {out}; final step {t.k}/{len(m['steps'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["demo", "run", "measure"])
    ap.add_argument("--model", default="house.json")
    ap.add_argument("--out", help="demo/run: output video; measure: palette json")
    ap.add_argument("--calib", default="calib.json", help="baseplate corners TL, TR, BR, BL in pixels")
    ap.add_argument("--palette", help="measured colors from the measure mode")
    ap.add_argument("--step", type=int, help="measure: which step is complete in the image")
    ap.add_argument("--source", default="0", help="camera index or video path")
    # 實拍時燈光不同要調這個：太小會把積木當成手，太大會把手當成積木
    ap.add_argument("--max-dist", type=float, default=MAX_DIST)
    ap.add_argument("--voice", action="store_true", help="speak instructions (macOS say)")
    a = ap.parse_args()
    model = load_model(a.model, a.palette)
    if a.mode == "demo":
        demo(model, a.out or "demo.mp4", a.max_dist)
    elif a.mode == "measure":
        measure(model, a.source, a.step, a.calib, a.out or "palette.json")
    else:
        run(model, a.source, a.max_dist, a.voice, a.calib, a.out)
