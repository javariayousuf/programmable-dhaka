import argparse, json, os, shutil, threading, time, webbrowser
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import cv2

# Review vehicles instead of frames. Links each vehicle box to the same vehicle in neighboring stills, shows
# one card per vehicle with a few thumbnails, and saves one answer per vehicle. Runs on your own machine only:
# the server binds to 127.0.0.1, nothing is uploaded anywhere.
# usage: python review_cards.py <dataset_dir>     (needs annotations.coco.json and images/ inside it)
ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("--iou", type=float, default=0.5, help="strict linking. 0.3 groups more but sometimes merges two vehicles")
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--no-open", action="store_true")
args = ap.parse_args()

D = Path(args.dataset)
REV = D / "review"
NAMES = {1: "person", 2: "bicycle", 3: "motorcycle", 4: "rickshaw", 5: "cart", 6: "car"}
VEH = (2, 3, 4, 5, 6)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    u = a[2] * a[3] + b[2] * b[3] - i
    return i / u if u else 0


def build():
    coco = json.load(open(D / "annotations.coco.json"))
    boxes = [a for a in coco["annotations"] if a["category_id"] in VEH]
    # two boxes on the same vehicle in one still (the model sometimes draws a rickshaw box and a car box on one van):
    # keep the more confident one and remember the other so apply_review.py can delete it
    kept, dropped = [], []
    for n in sorted({b["image_id"] for b in boxes}):
        here = sorted([b for b in boxes if b["image_id"] == n], key=lambda b: -b.get("score", 1.0))
        chosen = []
        for b in here:
            if any(iou(b["bbox"], c["bbox"]) > 0.7 for c in chosen): dropped.append(b["id"])
            else: chosen.append(b)
        kept += chosen
    boxes = kept
    par = {b["id"]: b["id"] for b in boxes}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    by = {}
    for b in boxes:
        by.setdefault(b["image_id"], []).append(b)
    for n in sorted(by):
        for a in by[n]:
            for step in (1, 2):                      # allow one missed still in between
                best = None
                for b in by.get(n + step, []):
                    s = iou(a["bbox"], b["bbox"])
                    if s > args.iou and (best is None or s > best[0]):
                        best = (s, b)
                if best:
                    par[find(a["id"])] = find(best[1]["id"]); break
    # a vehicle can be lost for a few stills (glare, a passing bus) and then picked up again as a "new" one.
    # Join a track to the one that starts right after it ends, if the boxes line up and are about the same size.
    groups = {}
    for b in boxes:
        groups.setdefault(find(b["id"]), []).append(b)
    gl = [sorted(g, key=lambda b: b["image_id"]) for g in groups.values()]
    cand = []
    for i, A in enumerate(gl):
        for j, B in enumerate(gl):
            gap = B[0]["image_id"] - A[-1]["image_id"]
            if i != j and 1 <= gap <= 3:
                r = (B[0]["bbox"][2] * B[0]["bbox"][3]) / max(1, A[-1]["bbox"][2] * A[-1]["bbox"][3])
                v = iou(A[-1]["bbox"], B[0]["bbox"])
                if v > 0.3 and 0.5 <= r <= 2.0: cand.append((v, i, j))
    used_end, used_start = set(), set()
    for v, i, j in sorted(cand, reverse=True):
        if i in used_end or j in used_start: continue
        used_end.add(i); used_start.add(j); par[find(gl[i][0]["id"])] = find(gl[j][0]["id"])
    groups = {}
    for b in boxes:
        groups.setdefault(find(b["id"]), []).append(b)
    files = {i["id"]: i["file_name"] for i in coco["images"]}
    shutil.rmtree(REV / "thumbs", ignore_errors=True)
    (REV / "thumbs").mkdir(parents=True, exist_ok=True)
    tracks = []
    for k, g in enumerate(sorted(groups.values(), key=lambda g: min(b["image_id"] for b in g))):
        g.sort(key=lambda b: b["image_id"])
        votes = Counter()
        for b in g:
            votes[b["category_id"]] += b.get("score", 1.0)
        pick = [g[round(i * (len(g) - 1) / 3)] for i in range(4)] if len(g) >= 4 else g
        thumbs = []
        for j, b in enumerate(pick):
            img = cv2.imread(str(D / "images" / files[b["image_id"]]))
            x, y, w, h = b["bbox"]; pad = 0.12
            x1, y1 = max(0, int(x - pad * w)), max(0, int(y - pad * h))
            x2, y2 = min(img.shape[1], int(x + w * (1 + pad))), min(img.shape[0], int(y + h * (1 + pad)))
            crop = img[y1:y2, x1:x2]
            s = 170 / max(1, crop.shape[0])
            crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), 170))
            name = f"t{k}_{j}.jpg"
            cv2.imwrite(str(REV / "thumbs" / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 82])
            thumbs.append(name)
        tracks.append({"id": k, "box_ids": [b["id"] for b in g], "frames": [g[0]["image_id"], g[-1]["image_id"]], "n": len(g),
                       "model": NAMES[votes.most_common(1)[0][0]], "conf": round(sum(b.get("score", 1.0) for b in g) / len(g), 2),
                       "thumbs": thumbs, "mixed_in_model": len(votes) > 1})
    old_tracks, old_ans = REV / "tracks.json", REV / "answers.json"
    if old_tracks.exists() and old_ans.exists():
        owner = {bid: t["id"] for t in json.load(open(old_tracks)) for bid in t["box_ids"]}
        answers = {int(k): v for k, v in json.load(open(old_ans)).items()}
        shutil.copy(old_ans, REV / "answers.before_rebuild.json")
        new_ans = {}
        for t in tracks:
            got = {answers[owner[b]] for b in t["box_ids"] if b in owner and owner[b] in answers}
            if len(got) == 1: new_ans[str(t["id"])] = got.pop()      # every earlier answer on this vehicle agrees
        json.dump(new_ans, open(old_ans, "w"))
        print(f"carried over {len(new_ans)} of your {len(answers)} earlier answers")
    json.dump(tracks, open(old_tracks, "w"))
    json.dump(dropped, open(REV / "dropped.json", "w"))
    return tracks


PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vehicle review</title><style>
:root{--bg:#fff;--fg:#111;--mut:#666;--card:#f4f5f7;--line:#d8dbe0;--rick:#db3069;--cart:#e0ff4f;--moto:#3B0086;--bike:#EC9F05;--car:#8a8f98}
@media(prefers-color-scheme:dark){:root{--bg:#15161a;--fg:#f1f1f1;--mut:#9aa0a8;--card:#202228;--line:#34373f}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.4 -apple-system,Segoe UI,Helvetica,Arial,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 16px;z-index:5}
#bar{height:8px;background:var(--line);border-radius:4px;margin-top:8px;overflow:hidden}#fill{height:8px;background:var(--rick);width:0}
.tabs button{margin-right:8px}
main{padding:16px;max-width:980px;margin:0 auto}
.card{background:var(--card);border:2px solid transparent;border-radius:12px;padding:12px;margin-bottom:14px}
.card.cur{border-color:var(--rick)}.card.done{opacity:.55}
.th{display:flex;gap:8px;overflow-x:auto}.th img{height:170px;border-radius:8px}
.meta{margin:8px 0;color:var(--mut)}
button{font:inherit;padding:8px 12px;border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--fg);cursor:pointer}
button.pick{outline:3px solid var(--rick)}button.keep{font-weight:600}
kbd{font:12px monospace;border:1px solid var(--line);border-radius:4px;padding:0 4px;color:var(--mut)}
h2{margin:24px 0 8px}
</style></head><body>
<header><div><b id="count"></b> <span style="color:var(--mut)" id="hint">Keys: <kbd>Enter</kbd> keep model's answer <kbd>r</kbd> rickshaw <kbd>c</kbd> cart <kbd>m</kbd> motorcycle <kbd>b</kbd> bicycle <kbd>k</kbd> car <kbd>x</kbd> not a vehicle <kbd>f</kbd> mixed, check frames <kbd>&larr;</kbd><kbd>&rarr;</kbd> move</span></div>
<div id="bar"><div id="fill"></div></div>
<div class="tabs" style="margin-top:8px"><button id="t_all">All</button><button id="t_todo">Not answered</button></div></header>
<main id="main"></main>
<script>
const OPT=[["rickshaw","r"],["cart","c"],["motorcycle","m"],["bicycle","b"],["car","k"],["not a vehicle","x"],["mixed: check frames","f"]];
let tracks=[],ans={},cur=0,onlyTodo=false;
let build='';
async function save(){
  const r=await fetch('/save',{method:'POST',headers:{'X-Build':build},body:JSON.stringify(ans)});
  if(r.status==409){alert('These cards were rebuilt since this page opened, so nothing was saved. Reload the page (Cmd+R).');location.reload()}
}
function render(){
  const main=document.getElementById('main');main.innerHTML='';
  const multi=tracks.filter(t=>t.n>1),single=tracks.filter(t=>t.n==1);
  const done=Object.keys(ans).length;
  document.getElementById('count').textContent=done+' of '+tracks.length+' vehicles answered';
  document.getElementById('fill').style.width=(100*done/tracks.length)+'%';
  function sect(title,list){
    if(!list.length)return;const h=document.createElement('h2');h.textContent=title;main.appendChild(h);
    list.forEach(t=>{ if(onlyTodo&&ans[t.id])return;
      const c=document.createElement('div');c.className='card'+(t.id==cur?' cur':'')+(ans[t.id]?' done':'');c.id='c'+t.id;
      c.innerHTML='<div class="th">'+t.thumbs.map(f=>'<img src="/thumbs/'+f+'">').join('')+'</div>'+
        '<div class="meta">Model says <b>'+t.model+'</b> (confidence '+t.conf+') &middot; seen in '+t.n+' still'+(t.n>1?'s':'')+' (stills '+t.frames[0]+' to '+t.frames[1]+')'+(t.mixed_in_model?' &middot; model changed its mind along the way':'')+'</div>';
      const row=document.createElement('div');
      const keep=document.createElement('button');keep.className='keep'+(ans[t.id]==t.model?' pick':'');keep.textContent='Keep: '+t.model+' (Enter)';keep.onclick=()=>set(t.id,t.model);row.appendChild(keep);
      OPT.forEach(([n,k])=>{const b=document.createElement('button');b.textContent=n+' ('+k+')';b.style.marginLeft='6px';if(ans[t.id]==n)b.className='pick';b.onclick=()=>set(t.id,n);row.appendChild(b)});
      c.appendChild(row);c.onclick=()=>{cur=t.id;mark()};main.appendChild(c)})}
  sect('Seen in several stills ('+multi.length+')',multi);
  sect('Seen in only one still ('+single.length+'), often false alarms',single);
}
function mark(){document.querySelectorAll('.card').forEach(c=>c.classList.remove('cur'));const e=document.getElementById('c'+cur);if(e){e.classList.add('cur');e.scrollIntoView({block:'center'})}}
function order(){const m=tracks.filter(t=>t.n>1),s=tracks.filter(t=>t.n==1);return m.concat(s).map(t=>t.id)}
function step(d){const o=order();let i=o.indexOf(cur)+d;i=Math.max(0,Math.min(o.length-1,i));cur=o[i];render();mark()}
function nextTodo(){const o=order();const i=o.indexOf(cur);for(let k=1;k<=o.length;k++){const id=o[(i+k)%o.length];if(!ans[id]){cur=id;return}}}
function set(id,v){ans[id]=v;save();cur=id;nextTodo();render();mark()}
document.addEventListener('keydown',e=>{
  if(e.key=='ArrowRight'||e.key=='ArrowDown'){e.preventDefault();step(1);return}
  if(e.key=='ArrowLeft'||e.key=='ArrowUp'){e.preventDefault();step(-1);return}
  const t=tracks.find(t=>t.id==cur);if(!t)return;
  if(e.key=='Enter'){set(cur,t.model);return}
  const o=OPT.find(([n,k])=>k==e.key);if(o)set(cur,o[0]);
});
document.getElementById('t_all').onclick=()=>{onlyTodo=false;render()};
document.getElementById('t_todo').onclick=()=>{onlyTodo=true;render()};
(async()=>{build=(await (await fetch('/version')).text());tracks=await (await fetch('/tracks')).json();ans=await (await fetch('/answers')).json();
  const o=order();cur=o.find(i=>!ans[i])??o[0];render();mark()})();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def send(self, body, ctype="application/json"):
        self.send_response(200); self.send_header("Content-Type", ctype); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/": self.send(PAGE.encode(), "text/html; charset=utf-8")
        elif p == "/version": self.send(BUILD.encode(), "text/plain")
        elif p == "/tracks": self.send((REV / "tracks.json").read_bytes())
        elif p == "/answers": self.send((REV / "answers.json").read_bytes() if (REV / "answers.json").exists() else b"{}")
        elif p.startswith("/thumbs/"):
            f = REV / "thumbs" / os.path.basename(p)
            if f.exists(): self.send(f.read_bytes(), "image/jpeg")
            else: self.send_error(404)
        else: self.send_error(404)

    def do_POST(self):
        if self.path == "/save":
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.headers.get("X-Build") != BUILD:         # a tab from before a rebuild must never overwrite answers
                self.send_error(409); return
            try:
                json.loads(body)                             # refuse anything that is not valid JSON
            except ValueError:
                self.send_error(400); return
            (REV / "answers.json").write_bytes(body)
            self.send(b"{}")
        else: self.send_error(404)


BUILD = str(time.time())
tracks = build()
multi = sum(1 for t in tracks if t["n"] > 1)
print(f"{len(tracks)} vehicles to review ({multi} seen in several stills, {len(tracks) - multi} in one). Answers save to {REV / 'answers.json'}")
srv = ThreadingHTTPServer(("127.0.0.1", args.port), H)
url = f"http://127.0.0.1:{args.port}/"
print("Open", url, "(Ctrl+C to stop; your answers are already saved)")
if not args.no_open: threading.Timer(0.8, lambda: webbrowser.open(url)).start()
try: srv.serve_forever()
except KeyboardInterrupt: pass
