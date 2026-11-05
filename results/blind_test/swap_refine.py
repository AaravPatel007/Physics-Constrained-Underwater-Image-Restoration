"""Replace the RefineNet entry of the blind test with refine_all_d1b. Same photos, same depth step as prep.py."""
import hashlib, json, sys
sys.path.insert(0, '/home/smeer/Downloads/MLOE---Group-4/src')
import numpy as np, torch
from pathlib import Path
from PIL import Image
import stress_test as st, depth as dav2
from eval_synth import load
from model import RefineNet
from device import DEV
from prep_uieb import resize_pair
OUT = Path('/home/smeer/Downloads/MLOE---Group-4/results/blind_test'); web = Path(sys.argv[1])
key = json.load(open(OUT/"key.json"))
base = load("baseline_v1_60ep_bigdata")[0].requires_grad_(False).eval()
r = RefineNet().to(DEV).eval(); r.load_state_dict(torch.load(st.ROOT/"results"/"refine_all_d1b"/"best.pt", map_location=DEV)["state"])
raws = []
for n, k in enumerate(key):
    p = web/k["source"] if n < 20 else st.EXT/k["source"]
    a = np.array(Image.open(p).convert("RGB")); raws.append(resize_pair(a, a)[0])
zs = [np.clip(z, .1, 30) for z in dav2.predict(raws, "metric-indoor", bs=1)]
def h(s):
    u = (np.clip(s, 0, 1)*255+.5).astype(np.uint8); return u, hashlib.sha1(u.tobytes()).hexdigest()[:16]
same = 0
for k, a, z in zip(key, raws, zs):
    I, zt = st.srgb_to_linear(st.t(a/255.0)), st.t(z[..., None].astype(np.float32))
    _, o = st.systems(base, {"refine_all_d1b": (r, True)}, I, zt)
    same += h(st.srgb_np(o["BetaNet + Day 1b"]))[1] == k["systems"]["Day 1b"]
    u, hh = h(st.srgb_np(o["RefineNet [refine_all_d1b]"]))
    Image.fromarray(u).save(OUT/"img"/f"{hh}.jpg", quality=93); k["systems"]["RefineNet"] = hh
print("Day 1b images reproduced exactly:", same, "of", len(key))
assert same >= 30, "inputs differ from prep.py"
json.dump(key, open(OUT/"key.json", "w"), indent=1)
