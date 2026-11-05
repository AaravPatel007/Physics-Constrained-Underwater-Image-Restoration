#!/usr/bin/env python3
"""Live status page for this project. Local only, standard library only.

  python3 results/logs/dashboard.py        then open http://127.0.0.1:8765

Sizes, processes, epochs and scores are read from disk on every refresh. The teammate table and the
dataset list below are hand-kept: edit them when a dataset or a result changes.
"""
import json
import re
import shutil
import subprocess
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES, LOGS, DATA = ROOT / "results", ROOT / "results" / "logs", ROOT / "data"
EXT = DATA / "external"
PORT = 8765
GB = 1e9

TEAM = [  # net, by, trained on, UIEB PSNR, UIEB colour error, note (Aarav's rows: docs/review2_plan.md on origin/main)
    ("day3a_touchup_global (chosen)", "Aarav (main)", "680 UIEB pairs", 21.68, 9.66, "Global colour matrix + tone curve. Weights here, score reproduced"),
    ("day3b global + local", "Aarav (main)", "680 UIEB pairs", 21.86, 9.53, "Four runs: 21.68 to 22.09 PSNR, 9.22 to 9.70 colour error. Inside seed noise, not chosen"),
    ("day3c control, raw input", "Aarav (main)", "680 UIEB pairs, no physics", 22.63, 10.24, "No gain on the grey charts"),
    ("day3d + LSUI mixed in", "Aarav (main)", "UIEB + 4,277 LSUI", 20.56, 11.11, "Worse on UIEB and charts"),
    ("day3e global + local + LSUI", "Aarav (main)", "UIEB + LSUI", 20.87, 10.90, "Worse on UIEB and charts"),
    ("day3f LSUI, then UIEB", "Aarav (main)", "LSUI, then UIEB", 21.59, 9.96, "Level with day3a, slightly better charts"),
    ("UNet black box (RGB-D)", "Team baseline", "NYU rooms + synthetic water", 17.6, 15.7,
     "Older UIEB run, not the same 90 photos. 25.1 dB on synthetic"),
    ("BridgeNet (NAFNet)", "Friend's De-water", "300 NYU rooms, 3 fixed water types", 15.77, 18.42,
     "Our stress test, UIEB val. Synthetic colour error 4.49"),
]

DONE, USING, LATER, NEEDS, CUT = "Done using", "Being used", "Going to be used", "Needs you", "Cut for now"
SQUID = {f["key"]: f["size"] for f in json.load(open(EXT / "squid" / "record.json"))["files"]}
FILES = [  # label, path, full size in bytes (None = complete), usage, note
    ("Sea-thru full", EXT / "seathru_full", None, CUT, "Shrink cut before the review. Originals kept"),
    *[(f"SQUID {k[:-4]}", EXT / "squid" / k, v, CUT, "Paused, not needed for the review") for k, v in SQUID.items()],
    ("R-CAUSTIC", EXT / "rcaustic" / "R-CAUSTIC_v2.0.zip", 10566641604, CUT, "Paused, not needed for the review"),
    ("FLSea", EXT / "flsea", 64e9, CUT, "Not needed for the review"),
    ("HICRD", EXT / "hicrd", None, USING, "Training and stress test. Size includes the zips, deleting them saves 11 GB"),
    ("TURBID", EXT / "turbid", None, CUT, "Not needed for the review. No public link, needs an email to the authors"),
    ("UIEB C60", EXT / "uieb_c60", None, USING, "Stress test (hard cases, no references)"),
    ("FathomNet sample", EXT / "fathomnet", 500e6, LATER, "Rough %: ~140 of 500 images"),
    ("OceanDark", EXT / "oceandark", None, LATER, "Outlier test set"),
    ("LSUI pairs", DATA / "lsui_pairs.npz", None, USING, "Training"),
    ("EUVP pairs", DATA / "euvp_pairs.npz", None, USING, "Training"),
    ("NYU train (big)", DATA / "nyu_train_big.npz", None, USING, "Training"),
    ("UIEB pairs", DATA / "uieb_train.npz", None, USING, "Training"),
    ("UIEB originals", DATA / "uieb", None, USING, "Evaluation"),
    ("NYU test", DATA / "nyu_test.npz", None, USING, "Stress test"),
    ("TankImage-I", EXT / "tankimage_i", None, USING, "Stress test"),
    ("TUB", EXT / "tub", None, USING, "Stress test"),
    ("RUIE", EXT / "ruie", None, USING, "Stress test"),
    ("Sea-thru small", DATA / "seathru_small.npz", None, USING, "Stress test charts"),
    ("U45", EXT / "u45", None, USING, "Stress test"),
    ("Stress cache", DATA / "stress_cache", None, USING, "Stress test"),
    ("LSUI originals", EXT / "lsui", None, DONE, "Pairs built"),
    ("EUVP originals", EXT / "euvp_dark", None, DONE, "Pairs built"),
    ("Sea-thru originals", DATA / "seathru", None, DONE, "Small file built"),
    ("NYU train (small)", DATA / "nyu_train.npz", None, DONE, "Replaced by the big set"),
    ("Hugging Face cache", Path.home() / ".cache" / "huggingface", None, DONE, "Download cache, not checked file by file"),
]

MIX = "UIEB + LSUI + EUVP"
RUNS = [  # refiner, input, trained on, started from
    ("refine_v1", "BetaNet", "NYU synthetic", "scratch"),
    ("refine_ft", "BetaNet", "UIEB + synthetic", "refine_v1"),
    ("refine_mix", "BetaNet", f"{MIX} (session 5 subset)", "refine_v1"),
    ("refine_mix_full", "BetaNet", f"{MIX} (full)", "refine_v1"),
    ("refine_lsui", "BetaNet", "LSUI + synthetic", "refine_v1"),
    ("refine_lsui_uieb", "BetaNet", "LSUI, then UIEB", "refine_lsui"),
    ("refine_d1b", "Day 1b", "UIEB + synthetic", "refine_ft"),
    ("refine_mix_d1b", "Day 1b", f"{MIX} (full)", "refine_mix_full"),
    ("refine_all_d1b", "Day 1b", f"{MIX} + HICRD", "refine_mix_d1b"),
    ("refine_only_uieb", "Day 1b", "UIEB only (control)", "refine_v1"),
    ("refine_no_hicrd", "Day 1b", "All but HICRD", "refine_v1"),
    ("refine_no_lsui", "Day 1b", "All but LSUI", "refine_v1"),
    ("refine_no_euvp", "Day 1b", "All but EUVP", "refine_v1"),
    ("refine_no_uieb", "Day 1b", "All but UIEB", "refine_v1"),
    ("refine_d1b_free", "Day 1b", "UIEB, no anchor or synthetic", "refine_ft"),
]
SCORES = ["stress_test_final", "stress_test_bridge", "stress_test_day3", "stress_test_s6b", "stress_test_s6", "stress_test"]  # newest first
# minutes per stress set, measured in session 6; only used for the time-left estimate
STRESS = {"synth": 4, "uieb160": 2, "uieb90": 1, "lsui": 5, "euvp": 4, "tub": 5, "uccs_blue": .5, "uccs_green": .5,
          "uccs_blue-green": .5, "uiqs": 1, "u45": .5, "tank": .5, "charts": .5}
NIGHT = [  # key in s6_night.log, step, what it answers, minutes (refine_* use the measured average once one has finished)
    ("stress_test_bridge", "Score BridgeNet and the hybrid", "How they do on the scored sets", 57),
    ("refine_all_d1b", "Train RefineNet on all 4 datasets", "Does adding HICRD fix the pink overshoot?", 43),
    ("refine_only_uieb", "Train UIEB-only control", "Baseline for the leave-one-out runs", 43),
    ("refine_no_hicrd", "Train without HICRD", "Does it work on water it never saw?", 43),
    ("refine_no_lsui", "Train without LSUI", "Does it work on water it never saw?", 43),
    ("refine_no_euvp", "Train without EUVP", "Does it work on water it never saw?", 43),
    ("refine_no_uieb", "Train without UIEB", "Does it work on water it never saw?", 43),
    ("refine_d1b_free", "Rematch against day3a", "Can RefineNet catch up with day3a's freedom on UIEB?", 43),
    ("stress_test_final", "Full stress test", "Final numbers for every new model", 40),
    ("seed 1", "day3b seed 1", "Is day3b really level with day3a?", 10),
    ("seed 2", "day3b seed 2", "Is day3b really level with day3a?", 10),
]
AFTER = [  # who, when, what
    ("You", "Morning", "Get 4-5 people through the blind test at http://127.0.0.1:8766"),
    ("You", "After all votes", "Open /results on the blind test page for the tally"),
    ("You", "Morning", "Make the slides from agent_logs.md and rehearse"),
    ("You", "Before or after the review", "Decide whether to commit session 6"),
    ("Team", "About 13:00", "Review"),
    ("Reference", "Any time", "Results and slide framing: agent_logs.md, section Overnight results"),
]
SCRIPTS = {p.name for p in (ROOT / "src").glob("*.py")}

_du, _sizes = {}, {}


def size(p):
    """Bytes on disk; folders are cached for 60 s."""
    if p.is_file():
        return p.stat().st_size
    if not p.exists():
        return 0
    t, v = _du.get(p, (0, 0))
    if time.time() - t > 60:
        v = int(subprocess.run(["du", "-sb", str(p)], capture_output=True, text=True).stdout.split()[0])
        _du[p] = (time.time(), v)
    return v


def speed(name, now_bytes):
    """Bytes per second over the last 5 minutes of samples."""
    q = _sizes.setdefault(name, deque())
    q.append((time.time(), now_bytes))
    while len(q) > 2 and q[-1][0] - q[0][0] > 300:
        q.popleft()
    dt = q[-1][0] - q[0][0]
    return (q[-1][1] - q[0][1]) / dt if dt > 5 else None


def training_row(name, args, secs):
    log = LOGS / f"s6_{name}.log"
    eps = [json.loads(l) for l in log.read_text().splitlines() if l.startswith('{"epoch"')] if log.exists() else []
    m = re.search(r"--epochs (\d+)", args)
    total = int(m.group(1)) if m else 12
    per = sum(e["time"] for e in eps) / len(eps) if eps else None
    return {"name": f"Training {name}", "detail": f"Epoch {len(eps)} of {total}", "pct": 100 * len(eps) / total,
            "left": (total - len(eps)) * per if per else None}


def stress_row(args, secs):
    out = re.search(r"--out (\S+)", args).group(1)
    log = LOGS / {"stress_test_s6": "s6_stress.log", "stress_test_s6b": "s6_stress_b.log", "stress_test_day3": "s6_stress_c60.log"}.get(out, f"s6_{out}.log")
    only = re.search(r"--only (\S+)", args)
    sets = only.group(1).split(",") if only else list(STRESS)
    seen = [l for l in log.read_text(errors="ignore").replace("\r", "\n").splitlines() if l in sets]
    i = sets.index(seen[-1]) if seen else 0
    w = [STRESS.get(x, .5) for x in sets]
    done, total = sum(w[:i]), sum(w)
    return {"name": "Stress test", "detail": f"Set {i + 1} of {len(sets)}: {seen[-1] if seen else 'starting'}",
            "pct": 100 * done / total, "left": secs / done * (total - done) if done else None}


def day3_row(args):
    m = re.findall(r"^(day3\w+) ep (\d+) .*\((\d+)s\)$", (LOGS / "s6_day3.log").read_text(errors="ignore"), re.M)
    if "train" not in args or not m:
        return {"name": "Touch-up (Aarav's script)", "detail": args.split("day3_touchup.py")[-1].strip(), "pct": None, "left": None}
    name, ep, t = m[-1][0], int(m[-1][1]), int(m[-1][2])
    return {"name": f"Training {name}", "detail": f"Epoch {ep} of 120", "pct": 100 * ep / 120, "left": t / ep * (120 - ep)}


def running():
    rows = []
    ps = subprocess.run(["ps", "-eo", "stat=,etimes=,args="], capture_output=True, text=True).stdout
    for line in ps.splitlines():
        stat, secs, args = line.strip().split(None, 2)
        secs, argv = int(secs), args.split()
        found = [Path(x).name for x in argv[1:3] if Path(x).name in SCRIPTS]
        if Path(argv[0]).name.startswith("python") and found:
            script = found[0]
            n = re.search(r"--name (\S+)", args)
            if script == "stress_test.py":
                rows.append(stress_row(args, secs))
            elif script == "day3_touchup.py":
                rows.append(day3_row(args))
            elif script in ("finetune_refine.py", "refine.py") and n and "--test" not in args:
                rows.append(training_row(n.group(1), args, secs))
            else:
                rows.append({"name": f"{'Scoring ' + n.group(1) if n else script}", "detail": f"Running {secs // 60} min",
                             "pct": None, "left": None})
        elif Path(argv[0]).name == "curl" and "-o" in argv:
            f = argv[argv.index("-o") + 1]
            hit = next((x for x in FILES if x[1].name == f), None)
            if hit:
                now = size(hit[1])
                s = speed(f, now)
                paused = "T" in stat
                rows.append({"name": f"{'Paused' if paused else 'Downloading'} {hit[0]}", "pct": 100 * now / hit[2],
                             "detail": f"{now / GB:.1f} of {hit[2] / GB:.1f} GB" + (f", {s / 1e6:.2f} MB/s" if s and not paused else ""),
                             "left": (hit[2] - now) / s if s and not paused else None})
    return sorted(rows, key=lambda r: r["name"].startswith(("Downloading", "Paused")))


def queue():
    """Overnight steps from s6_night.log, with estimated finish times."""
    start, end, failed, finished = {}, {}, set(), False
    for line in (LOGS / "s6_night.log").read_text(errors="ignore").splitlines():
        m = re.match(r"\[(\d+):(\d+):(\d+)\] (\S+) (.+)", line)
        if m:
            t = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])
            word, rest = m[4], m[5].split(":")[0]
            if word == "done":
                end[rest] = t
            else:
                start[rest] = t
        elif line.startswith("FAILED "):
            failed.add(line[7:])
        finished |= line == "S6_NIGHT_DONE"
    alive = subprocess.run(["pgrep", "-f", "s6_night.sh"], capture_output=True).returncode == 0
    took = [end[k] - start[k] for k in end if k.startswith("refine_") and k in start]
    train = sum(took) / len(took) if took else None
    # shortcut: clock times only, assumes the queue stays within one calendar day
    lt = time.localtime()
    clock = now = lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec
    rows = []
    for i, (key, step, why, mins) in enumerate(NIGHT):
        est = train if train and key.startswith("refine_") else mins * 60
        later = any(k[0] in start for k in NIGHT[i + 1:])
        if key in failed:
            state, at = "Failed", None
        elif key in end or later or (finished and key in start):
            state, at = "Done", end.get(key)
        elif key in start:
            state = "Running" if alive else "Stopped"
            clock = max(start[key] + est, now + 60)
            at = clock
        else:
            state = "Waiting" if alive else "Stopped"
            clock += est
            at = clock
        rows.append({"step": step, "why": why, "state": state, "at": f"{int(at) // 3600 % 24:02d}:{int(at) // 60 % 60:02d}" if at else ""})
    return rows


def storage():
    rows = []
    for label, p, full, usage, note in FILES:
        have = size(p)
        pct = 100 * min(have / full, 1) if full else (100 if have else 0)
        rows.append({"name": label, "gb": have / GB, "pct": pct, "usage": usage, "note": note})
    order = {NEEDS: 0, USING: 1, LATER: 2, CUT: 3, DONE: 4}
    return sorted(rows, key=lambda r: (order[r["usage"]], -r["gb"]))


def score(key):
    """Duplicate-free UIEB val mean and CI vs Day 1b from the newest run that has this system."""
    for out in SCORES:
        f = RES / out / "clean_rescore.json"
        if f.exists():
            u = json.load(open(f))["uieb90"]
            if key in u["mean"]:
                return u["mean"][key], u["ci"].get(f"{key} - BetaNet + Day 1b"), out
    return None, None, None


def verdict(m, ci, ref):
    if not m:
        return ("wait", "Waiting for stress test")
    d = m["psnr"] - ref["psnr"]
    if not ci:  # older run, no CI against Day 1b
        return ("bad", "Below Day 1b") if d < 0 else ("level", "Close to Day 1b")
    p, e = ci["psnr"], ci["de2000"]
    if p[1] > 0 and e[2] < 0:
        return ("good", "Beats Day 1b")
    if p[2] < 0 or e[1] > 0:
        return ("bad", "Worse than Day 1b")
    return ("level", "Better PSNR, colour level" if p[1] > 0 else "Level with Day 1b")


def history():
    ref = score("BetaNet + Day 1b")[0]
    rows = [{"name": "BetaNet alone", "data": "NYU synthetic, 60 epochs", "progress": "Done", "pct": 100,
             "m": score("BetaNet")[0], "v": ("bad", "Below Day 1b")},
            {"name": "BetaNet + Day 1b", "data": "No training (fixed recipe)", "progress": "Done", "pct": 100, "m": ref,
             "v": ("ref", "Reference")}]
    for name, inp, data, init in RUNS:
        if not (RES / name / "best.pt").exists():
            continue
        m, ci, _ = score(f"RefineNet [{name}]")
        log = RES / name / "log.json"
        eps = json.load(open(log)) if log.exists() else []
        total = max((e["epoch"] for e in eps), default=0)
        finished = (RES / name / "ft_eval.json").exists() or name == "refine_v1"
        rows.append({"name": f"{'BetaNet + Day 1b' if inp == 'Day 1b' else 'BetaNet'} + {name}",
                     "data": f"{data}, from {init}", "progress": f"{total} epochs", "pct": 100 if finished else None,
                     "m": m, "v": verdict(m, ci, ref)})
    for label, key, data in (("BetaNet + Day 1b + day3a", "TouchUp [day3a_touchup_global]", "680 UIEB pairs (Aarav's weights)"),
                             ("BridgeNet", "BridgeNet", "NYU synthetic, older pipeline"), ("Hybrid (BridgeNet + day3a)", "Hybrid", "No training")):
        m, ci, _ = score(key)
        if m:
            rows.append({"name": label, "data": data, "progress": "Done", "pct": 100, "m": m, "v": verdict(m, ci, ref)})
    scored = [r for r in rows if r["m"]]
    best = max(scored, key=lambda r: r["m"]["psnr"])
    best["best"] = True
    for r in rows:
        if r["m"]:
            r["psnr"], r["de"], r["dp"] = r["m"]["psnr"], r["m"]["de2000"], r["m"]["psnr"] - ref["psnr"]
        del r["m"]
    return rows, {"best": best["name"]}


def snapshot():
    du = shutil.disk_usage(ROOT)
    hist, top = history()
    return {"time": time.strftime("%H:%M:%S"), "disk": {"used": du.used / GB, "total": du.total / GB, "free": du.free / GB},
            "team": [dict(zip(("name", "by", "data", "psnr", "de", "note"), t)) for t in TEAM],
            "storage": storage(), "running": running(), "queue": queue(),
            "after": [dict(zip(("who", "when", "what"), a)) for a in AFTER], "history": hist, "top": top}


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BetaNet Live Status</title><style>
:root{color-scheme:light;--bg:#fcfcfb;--text:#0b0b0b;--muted:#52514e;--line:#e4e3df;--track:#ecebe7;--fill:#2a78d6;
--good:#0ca30c;--warn:#fab219;--bad:#d03b3b;--rowhi:#eef4fc}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--bg:#1a1a19;--text:#f1f0ec;--muted:#b0aea7;--line:#383733;--track:#32312e;
--fill:#5b9ae6;--rowhi:#22303f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,sans-serif}
main{max-width:1180px;margin:0 auto;padding:20px 16px 48px}
header{display:flex;flex-wrap:wrap;gap:6px 24px;align-items:baseline;justify-content:space-between}
h1{font-size:20px;margin:0}h2{font-size:15px;margin:30px 0 8px}
.sub{color:var(--muted)}.scroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;min-width:760px}
th{font-weight:600;color:var(--muted);text-align:left;font-size:12px;padding:6px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:middle}
.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}tr.hi td{background:var(--rowhi)}
.bar{display:flex;align-items:center;gap:8px;min-width:150px}
.track{flex:1;height:8px;border-radius:4px;background:var(--track);overflow:hidden}
.fill{height:100%;border-radius:4px;background:var(--fill);transition:width .6s}
.pct{width:38px;text-align:right;font-variant-numeric:tabular-nums;color:var(--muted)}
.tag{white-space:nowrap}.tag i{font-style:normal;display:inline-block;width:1.2em;text-align:center}
.good i{color:var(--good)}.bad i{color:var(--bad)}.warn i{color:var(--warn)}.mute{color:var(--muted)}
#stale{color:var(--bad);font-weight:600}
</style></head><body><main>
<header><h1>BetaNet live status</h1><div class="sub"><span id="stale"></span> <span id="when">Loading…</span></div></header>
<div id="root"></div></main><script>
const e=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const n=(v,d=2)=>v==null?'<span class="mute">–</span>':Number(v).toFixed(d);
const bar=p=>p==null?'<span class="mute">–</span>':`<div class="bar"><div class="track" role="progressbar" aria-valuenow="${Math.round(p)}" aria-valuemin="0" aria-valuemax="100"><div class="fill" style="width:${Math.min(p,100)}%"></div></div><span class="pct">${Math.round(p)}%</span></div>`;
const left=s=>{if(s==null)return'<span class="mute">Estimating</span>';s=Math.round(s/60);return s<1?"Under 1 min":s<90?`${s} min`:`${(s/60).toFixed(1)} h`};
const table=(cols,rows)=>`<div class="scroll"><table><thead><tr>${cols.map(c=>`<th${c[0]=="#"?' class="num"':""}>${e(c.replace("#",""))}</th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody></table></div>`;
const usage={"Done using":["good","✓"],"Being used":["","●"],"Going to be used":["mute","○"],"Needs you":["warn","!"],"Cut for now":["mute","–"]};
const tag=(cls,icon,t)=>`<span class="tag ${cls}"><i>${icon}</i>${e(t)}</span>`;
const qicon={Done:["good","✓"],Running:["","●"],Waiting:["mute","○"],Failed:["bad","✕"],Stopped:["bad","!"]};
const vicon={good:["good","✓"],bad:["bad","✕"],level:["mute","≈"],ref:["mute","◆"],wait:["mute","…"]};
function draw(d){
 const run=d.running.length?table(["Process","Status","Progress","Time left"],d.running.map(r=>`<tr><td>${e(r.name)}</td><td>${e(r.detail)}</td><td>${bar(r.pct)}</td><td>${left(r.left)}</td></tr>`)):'<p class="sub">Nothing running.</p>';
 const open=d.queue.filter(q=>q.state=="Running"||q.state=="Waiting"),nd=d.queue.filter(q=>q.state=="Done").length;
 const qnote=open.length?`${nd} of ${d.queue.length} steps done. Estimated finish ${open[open.length-1].at}.`:`${nd} of ${d.queue.length} steps done.`;
 const q=table(["Step","What it answers","Status","Finish"],d.queue.map(r=>{const v=qicon[r.state];return`<tr${r.state=="Running"?' class="hi"':""}><td>${e(r.step)}</td><td>${e(r.why)}</td><td>${tag(v[0],v[1],r.state)}</td><td>${r.state=="Done"?e(r.at):r.at?"About "+e(r.at):""}</td></tr>`}));
 const after=table(["Who","When","What"],d.after.map(a=>`<tr><td>${e(a.who)}</td><td>${e(a.when)}</td><td>${e(a.what)}</td></tr>`));
 const done=table(["Combination","Trained on","Progress","Finished","#UIEB PSNR","#Colour error","#PSNR vs Day 1b","Verdict"],d.history.map(r=>{const v=vicon[r.v[0]];return`<tr${r.best?' class="hi"':""}><td>${e(r.name)}${r.best?" <strong>(best)</strong>":""}</td><td>${e(r.data)}</td><td class="tag">${e(r.progress)}</td><td>${bar(r.pct)}</td><td class="num">${n(r.psnr)}</td><td class="num">${n(r.de)}</td><td class="num">${r.dp==null?n(null):(r.dp>=0?"+":"")+r.dp.toFixed(2)}</td><td>${tag(v[0],v[1],r.v[1])}</td></tr>`}));
 const team=table(["Black-box net","By","Trained on","#UIEB PSNR","#Colour error","Note"],d.team.map(t=>`<tr><td>${e(t.name)}</td><td>${e(t.by)}</td><td>${e(t.data)}</td><td class="num">${n(t.psnr)}</td><td class="num">${n(t.de)}</td><td>${e(t.note)}</td></tr>`));
 const store=table(["Dataset","#Size (GB)","Downloaded","Usage","Note"],d.storage.map(s=>{const u=usage[s.usage];return`<tr><td>${e(s.name)}</td><td class="num">${n(s.gb,1)}</td><td>${bar(s.pct)}</td><td>${tag(u[0],u[1],s.usage)}</td><td>${e(s.note)}</td></tr>`}));
 document.getElementById("root").innerHTML=`
 <h2>Running now</h2>${run}
 <h2>Overnight queue</h2><p class="sub">${qnote} Finish times ahead are estimates.</p>${q}
 <h2>After the queue</h2>${after}
 <h2>Finished runs</h2><p class="sub">UIEB val, duplicates removed. Best: ${e(d.top.best)}.</p>${done}
 <h2>Teammates' black-box nets</h2>${team}
 <h2>Storage</h2><p class="sub">Disk: ${d.disk.free.toFixed(0)} GB free of ${d.disk.total.toFixed(0)} GB.</p>${bar(100*d.disk.used/d.disk.total)}<div style="height:10px"></div>${store}`;
 document.getElementById("when").textContent="Updated "+d.time+", refreshes every 5 s";
}
async function tick(){try{const r=await fetch("/data");if(!r.ok)throw 0;draw(await r.json());document.getElementById("stale").textContent=""}
catch{document.getElementById("stale").textContent="Lost connection, showing last data."}}
tick();setInterval(tick,5000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/data":
            body, kind = json.dumps(snapshot()).encode(), "application/json"
        elif self.path == "/":
            body, kind = PAGE.encode(), "text/html; charset=utf-8"
        else:
            return self.send_error(404)
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"http://127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
