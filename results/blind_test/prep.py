"""Blind test images: 20 random web photos + 10 C60 + 5 HICRD test + 5 U45, drawn with a fixed seed, never picked by eye.

  python -B -I results/blind_test/prep.py <folder of web photos>
Writes results/blind_test/img/<opaque id>.jpg and key.json (the only place that maps ids to systems).
"""
import hashlib, json, random, sys
sys.path.insert(0, '/home/smeer/Downloads/MLOE---Group-4/src')
import numpy as np, torch
from pathlib import Path
from PIL import Image
import stress_test as st, depth as dav2
from eval_synth import load
from model import RefineNet
from device import DEV
from prep_uieb import resize_pair
from day3_touchup import load_touchup
from bridge_system import system
OUT = Path(__file__).parent
web = sys.argv[1]
base = load("baseline_v1_60ep_bigdata")[0].requires_grad_(False).eval()
r = RefineNet().to(DEV).eval(); r.load_state_dict(torch.load(st.ROOT/"results"/"refine_mix_d1b"/"best.pt", map_location=DEV)["state"])
st.TOUCH["day3a_touchup_global"] = load_touchup("day3a_touchup_global")[0]; st.EXTRA.append(system)
rng = random.Random(20261105)
E = st.EXT
paths = [Path(web)/f'{i:02d}.jpg' for i in [0,2,3,4,5,6,8,9,10,11,12,14,17,18,19,20,21,23,24,25]]
paths += rng.sample(sorted((E/"uieb_c60"/"extracted"/"challenging-60").glob("*.png")), 10)
paths += rng.sample(sorted((E/"hicrd"/"HICRD_paired"/"testA").glob("*.png")), 5)
paths += rng.sample(sorted((E/"u45"/"upload"/"U45"/"U45").glob("*.png")), 5)
raws = []
for p in paths:
    a = np.array(Image.open(p).convert("RGB")); raws.append(resize_pair(a, a)[0])
zs = [np.clip(z, .1, 30) for z in dav2.predict(raws, "metric-indoor", bs=1)]
names = {"Day 1b": "BetaNet + Day 1b", "day3a": "TouchUp [day3a_touchup_global]", "RefineNet": "RefineNet [refine_mix_d1b]", "BridgeNet": "BridgeNet", "Hybrid": "Hybrid"}
def save(s):
    u = (np.clip(s, 0, 1)*255+.5).astype(np.uint8); h = hashlib.sha1(u.tobytes()).hexdigest()[:16]
    Image.fromarray(u).save(OUT/"img"/f"{h}.jpg", quality=93); return h
key = []
for n, (p, a, z) in enumerate(zip(paths, raws, zs)):
    I, zt = st.srgb_to_linear(st.t(a/255.0)), st.t(z[..., None].astype(np.float32))
    _, o = st.systems(base, {"refine_mix_d1b": (r, True)}, I, zt)
    key.append({"photo": n, "source": str(p.relative_to(E)) if n >= 20 else p.name, "raw": save(a/255.0), "systems": {k: save(st.srgb_np(o[v])) for k, v in names.items()}})
    print(n, flush=True)
json.dump(key, open(OUT/"key.json", "w"), indent=1)
