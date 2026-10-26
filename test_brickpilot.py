"""合成畫面自我檢查：python test_brickpilot.py"""
import numpy as np

import brickpilot as bp

m = bp.load_model("house.json")
H = bp.rectify_matrix(bp.DEMO_CORNERS, m["grid"])
rng = np.random.default_rng(1)
n = len(m["steps"])

# 每一步的畫面都要辨識成正確步驟、沒有誤報
for k, state in enumerate(m["states"]):
    obs = bp.observe(bp.render(state, m, rng), H, m, bp.MAX_DIST)
    assert (obs == state).all(), f"step {k}: {np.argwhere(obs != state)[:3]}"
    got, wrong = bp.locate(obs, m["states"], k)
    assert got == k and not wrong.any(), (k, got)

# 放錯一塊：停在原步驟，且只標出那 4 格
bad = m["states"][3].copy()
bad[12:14, 12:14] = "blue"
k, wrong = bp.locate(bp.observe(bp.render(bad, m, rng), H, m, bp.MAX_DIST), m["states"], 3)
assert k == 3 and set(zip(*np.nonzero(wrong))) == {(12, 12), (12, 13), (13, 12), (13, 13)}
assert bp.message(m, k, wrong) == "Wrong brick: remove it"

# 手擋住下一塊的位置：不能提前跳步
obs = bp.observe(bp.render(m["states"][4], m, rng, hand=(450, 330)), H, m, bp.MAX_DIST)
assert (obs == bp.UNKNOWN).any()
assert bp.locate(obs, m["states"], 4)[0] == 4

# 整支 demo 跑完要停在最後一步
t = bp.Tracker(m, bp.MAX_DIST)
for state, hand in bp.demo_script(m):
    t.update(bp.render(state, m, rng, hand), H)
assert t.k == n and bp.message(m, t.k, t.wrong) == "Build complete!"
print("ok")
