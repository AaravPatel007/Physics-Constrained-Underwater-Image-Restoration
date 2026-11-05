#!/usr/bin/env python3
"""Blind A/B test. python3 results/blind_test/server.py, then http://127.0.0.1:8766 (one person at a time, on this laptop).
Votes append to votes.jsonl. http://127.0.0.1:8766/results shows the tally: look only after everyone has voted."""
import itertools, json, random, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
D = Path(__file__).parent
KEY = json.load(open(D/"key.json"))
SYS = sorted(KEY[0]["systems"])
PAIRS = list(itertools.combinations(SYS, 2))
LOCK = threading.Lock()

def trials(rater):
    """Every pair equally often, shuffled per rater; sides random. Names never reach the page."""
    g = random.Random("blind:" + rater)
    ps = [PAIRS[i % len(PAIRS)] for i in range(len(KEY))]
    g.shuffle(ps)
    out = []
    for k, (a, b) in zip(KEY, ps):
        if g.random() < .5:
            a, b = b, a
        out.append({"photo": k["photo"], "raw": k["raw"], "left": k["systems"][a], "right": k["systems"][b]})
    g.shuffle(out)
    return out

def tally():
    who = {h: s for k in KEY for s, h in k["systems"].items()}
    votes = {}
    f = D/"votes.jsonl"
    for l in (f.read_text().splitlines() if f.exists() else []):
        v = json.loads(l); votes[(v["rater"], v["photo"])] = v  # a repeat replaces the earlier vote
    w = {s: {"wins": 0, "losses": 0, "ties": 0} for s in SYS}
    head = {f"{a} vs {b}": [0, 0, 0] for a, b in PAIRS}
    for v in votes.values():
        l, r = who[v["left"]], who[v["right"]]
        a, b = sorted((l, r))
        c = v["choice"]
        if c == "tie":
            w[l]["ties"] += 1; w[r]["ties"] += 1; head[f"{a} vs {b}"][2] += 1
        else:
            win, lose = (l, r) if c == "left" else (r, l)
            w[win]["wins"] += 1; w[lose]["losses"] += 1; head[f"{a} vs {b}"][0 if win == a else 1] += 1
    for s in w.values():
        n = s["wins"] + s["losses"] + s["ties"]
        s["score"] = round((s["wins"] + .5*s["ties"]) / n, 3) if n else None
    return {"votes": len(votes), "raters": len({r for r, _ in votes}), "systems": w, "head_to_head [first, second, tie]": head}

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Blind Photo Test</title>
<style>:root{color-scheme:light dark;--bg:#fcfcfb;--text:#0b0b0b;--muted:#52514e;--line:#c9c8c3}@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--text:#f1f0ec;--muted:#b0aea7;--line:#4a4944}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,sans-serif}main{max-width:1200px;margin:0 auto;padding:16px}
h1{font-size:19px;margin:0 0 4px}.sub{color:var(--muted)}.raw{display:block;height:130px;margin:10px auto 4px;border-radius:4px}.rawcap{text-align:center;font-size:12px;color:var(--muted)}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px}.pair img{width:100%;display:block;border-radius:4px}
button{font:inherit;padding:10px 14px;border:1px solid var(--line);border-radius:6px;background:transparent;color:inherit;cursor:pointer;width:100%;margin-top:6px}.mid{max-width:240px;margin:10px auto 0}input{font:inherit;padding:9px;border:1px solid var(--line);border-radius:6px;background:transparent;color:inherit}</style></head><body><main>
<h1>Which looks more like the real scene?</h1><div class="sub" id="status">Enter your name to start. Keys: A = left, L = right, T = can't tell.</div><div id="app"><p><input id="name" placeholder="Your name" aria-label="Your name"> <button style="width:auto" onclick="start()">Start</button></p></div></main>
<script>let T=[],i=0,R="";
async function start(){R=document.getElementById("name").value.trim();if(!R)return;T=await(await fetch("/trials?rater="+encodeURIComponent(R))).json();show()}
function show(){if(i>=T.length){document.getElementById("app").innerHTML="<p>Done, thank you.</p>";document.getElementById("status").textContent=T.length+" of "+T.length;return}
const t=T[i];document.getElementById("status").textContent="Photo "+(i+1)+" of "+T.length+". Keys: A = left, L = right, T = can't tell.";
document.getElementById("app").innerHTML=`<img class="raw" src="/img/${t.raw}.jpg" alt="Original underwater photo"><div class="rawcap">Original</div><div class="pair"><div><img src="/img/${t.left}.jpg" alt="Option left"><button onclick="vote('left')">Left is more real</button></div><div><img src="/img/${t.right}.jpg" alt="Option right"><button onclick="vote('right')">Right is more real</button></div></div><div class="mid"><button onclick="vote('tie')">Can't tell</button></div>`}
async function vote(c){const t=T[i];await fetch("/vote",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({rater:R,photo:t.photo,left:t.left,right:t.right,choice:c})});i++;show()}
addEventListener("keydown",e=>{if(!T.length||e.target.tagName=="INPUT")return;const k=e.key.toLowerCase();if(k=="a")vote("left");if(k=="l")vote("right");if(k=="t")vote("tie")})</script></body></html>"""

class H(BaseHTTPRequestHandler):
    def send(self, body, kind):
        self.send_response(200); self.send_header("Content-Type", kind); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            return self.send(PAGE.encode(), "text/html; charset=utf-8")
        if u.path == "/trials":
            return self.send(json.dumps(trials(parse_qs(u.query).get("rater", [""])[0])).encode(), "application/json")
        if u.path == "/results":
            return self.send(json.dumps(tally(), indent=1).encode(), "application/json")
        name = u.path[5:-4] if u.path.startswith("/img/") and u.path.endswith(".jpg") else ""
        if name.isalnum() and (D/"img"/f"{name}.jpg").exists():
            return self.send((D/"img"/f"{name}.jpg").read_bytes(), "image/jpeg")
        self.send_error(404)
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        if self.path != "/vote" or n > 2000:
            return self.send_error(400)
        v = json.loads(self.rfile.read(n))
        if v.get("choice") not in ("left", "right", "tie"):
            return self.send_error(400)
        with LOCK, open(D/"votes.jsonl", "a") as f:
            f.write(json.dumps(v) + "\n")
        self.send(b"ok", "text/plain")
    def log_message(self, *a):
        pass

if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8766), H).serve_forever()
