import argparse, json, os, threading, time, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Add the boxes the model never drew, and remove wrong ones. review_cards.py can only judge boxes that already exist,
# so a rickshaw the model missed never reaches it. This page shows one still at a time: drag a box around a vehicle,
# press a key for its class. Runs on your own machine only (127.0.0.1), nothing is uploaded anywhere.
# usage: python add_boxes.py <dataset_dir>      then: python apply_added.py <dataset_dir>
ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--no-open", action="store_true")
ap.add_argument("--show-people", action="store_true", help="also draw the person boxes (hidden by default, they crowd the picture)")
args = ap.parse_args()

D = Path(args.dataset)
REV = D / "review_added"
REV.mkdir(exist_ok=True)
SAVE = REV / "added.json"
coco = json.load(open(D / "annotations.coco.json"))
CATS = {c["id"]: c["name"] for c in coco["categories"]}
KEYS = {"r": "rickshaw", "m": "motorcycle", "b": "bicycle", "k": "car", "t": "truck", "c": "cart"}
BUILD = str(time.time())

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Add boxes</title><style>
:root{--bg:#fff;--fg:#111;--mut:#666;--line:#d8dbe0}
@media(prefers-color-scheme:dark){:root{--bg:#15161a;--fg:#f1f1f1;--mut:#9aa0a8;--line:#34373f}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.4 -apple-system,Segoe UI,Helvetica,Arial,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;z-index:5}
kbd{font:12px monospace;border:1px solid var(--line);border-radius:4px;padding:0 4px;color:var(--mut)}
#wrap{overflow:auto;height:calc(100vh - 92px)}
canvas{display:block;cursor:crosshair}
button{font:inherit;padding:4px 10px;border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--fg);cursor:pointer}
</style></head><body>
<header>
<div><b id="pos"></b> &middot; <span id="msg" style="color:var(--mut)"></span></div>
<div style="color:var(--mut)">Drag a box around a vehicle, then press a class key:
<kbd>r</kbd> rickshaw <kbd>m</kbd> motorcycle <kbd>b</kbd> bicycle <kbd>k</kbd> car <kbd>t</kbd> truck or bus <kbd>c</kbd> cart.
Click a box to select it, <kbd>x</kbd> removes it. <kbd>u</kbd> undo. <kbd>Esc</kbd> cancel.
<kbd>&larr;</kbd><kbd>&rarr;</kbd> change still. <kbd>z</kbd> zoom 1x/2x/3x.</div>
</header>
<div id="wrap"><canvas id="cv"></canvas></div>
<script>
const KEYS={r:"rickshaw",m:"motorcycle",b:"bicycle",k:"car",t:"truck",c:"cart"};
const COL={rickshaw:"#db3069",motorcycle:"#7a2bd6",bicycle:"#EC9F05",car:"#8a8f98",truck:"#ef233c",cart:"#e0ff4f",person:"#3777ff"};
let data,added=[],removed=[],idx=0,zoom=1,img=new Image(),drag=null,pending=null,sel=null,build="";
const cv=document.getElementById("cv"),cx=cv.getContext("2d");
async function save(){
  const r=await fetch("/save",{method:"POST",headers:{"X-Build":build},body:JSON.stringify({added:added,removed:removed})});
  if(r.status==409){alert("This page is out of date (the server restarted). Reload with Cmd+R.");location.reload()}
}
function cur(){return data.images[idx]}
function load(){
  const im=cur();img=new Image();img.onload=draw;img.src="/img/"+im.file_name;
  document.getElementById("pos").textContent="Still "+(idx+1)+" of "+data.images.length;
  pending=null;sel=null;msg();
}
function msg(){
  const n=added.filter(a=>a.image_id==cur().id).length;
  document.getElementById("msg").textContent=n+" added here, "+added.length+" added in total, "+removed.length+" removed in total"+(pending?" | now press a class key":"");
}
function size(){
  const w=Math.round(document.getElementById("wrap").clientWidth*zoom)-(zoom>1?0:0);
  cv.width=img.naturalWidth;cv.height=img.naturalHeight;
  cv.style.width=w+"px";cv.style.height=Math.round(w*img.naturalHeight/img.naturalWidth)+"px";
}
function boxes(){
  const id=cur().id;
  const ex=data.boxes.filter(b=>b.image_id==id&&!removed.includes(b.id)).map(b=>({kind:"old",id:b.id,bbox:b.bbox,cat:b.cat}));
  const ad=added.map((a,i)=>({a:a,i:i})).filter(o=>o.a.image_id==id).map(o=>({kind:"new",i:o.i,bbox:o.a.bbox,cat:o.a.category}));
  return ex.concat(ad);
}
function draw(){
  size();cx.drawImage(img,0,0);
  const lw=Math.max(3,img.naturalWidth/450);
  boxes().forEach(b=>{
    const [x,y,w,h]=b.bbox;const same=sel&&sel.kind==b.kind&&(sel.id==b.id&&sel.i==b.i);
    cx.lineWidth=same?lw*2:lw;cx.strokeStyle=COL[b.cat]||"#fff";cx.setLineDash(b.kind=="new"?[lw*3,lw*2]:[]);
    cx.strokeRect(x,y,w,h);cx.setLineDash([]);
    cx.font=(lw*7)+"px sans-serif";cx.fillStyle=COL[b.cat]||"#fff";cx.fillText(b.cat+(b.kind=="new"?" (added)":""),x,Math.max(lw*7,y-lw*2));
  });
  if(drag||pending){const r=pending||drag;cx.lineWidth=lw;cx.strokeStyle="#fff";cx.setLineDash([lw*2,lw*2]);cx.strokeRect(r.x,r.y,r.w,r.h);cx.setLineDash([])}
}
function pt(e){const r=cv.getBoundingClientRect();return{x:(e.clientX-r.left)*cv.width/r.width,y:(e.clientY-r.top)*cv.height/r.height}}
cv.addEventListener("mousedown",e=>{pending=null;const p=pt(e);drag={x0:p.x,y0:p.y,x:p.x,y:p.y,w:0,h:0}});
cv.addEventListener("mousemove",e=>{if(!drag)return;const p=pt(e);drag.x=Math.min(drag.x0,p.x);drag.y=Math.min(drag.y0,p.y);drag.w=Math.abs(p.x-drag.x0);drag.h=Math.abs(p.y-drag.y0);draw()});
window.addEventListener("mouseup",e=>{
  if(!drag)return;const d=drag;drag=null;
  if(d.w>12&&d.h>12){pending={x:d.x,y:d.y,w:d.w,h:d.h};sel=null;msg();draw();return}
  const p={x:d.x0,y:d.y0};let best=null;
  boxes().forEach(b=>{const[x,y,w,h]=b.bbox;if(p.x>=x&&p.x<=x+w&&p.y>=y&&p.y<=y+h&&(!best||w*h<best.bbox[2]*best.bbox[3]))best=b});
  sel=best;draw();
});
document.addEventListener("keydown",e=>{
  if(e.key=="ArrowRight"||e.key==" "){e.preventDefault();idx=Math.min(data.images.length-1,idx+1);load();return}
  if(e.key=="ArrowLeft"){e.preventDefault();idx=Math.max(0,idx-1);load();return}
  if(e.key=="Escape"){pending=null;sel=null;msg();draw();return}
  if(e.key=="z"){zoom=zoom==1?2:zoom==2?3:1;draw();return}
  if(e.key=="u"){for(let i=added.length-1;i>=0;i--){if(added[i].image_id==cur().id){added.splice(i,1);break}}save();msg();draw();return}
  if(e.key=="x"&&sel){if(sel.kind=="old")removed.push(sel.id);else added.splice(sel.i,1);sel=null;save();msg();draw();return}
  if(pending&&KEYS[e.key]){added.push({image_id:cur().id,bbox:[pending.x,pending.y,pending.w,pending.h],category:KEYS[e.key]});pending=null;save();msg();draw()}
});
window.addEventListener("resize",draw);
(async()=>{build=await (await fetch("/version")).text();data=await (await fetch("/data")).json();
  const s=data.saved;added=s.added||[];removed=s.removed||[];idx=0;load()})();
</script></body></html>"""


def payload():
    ims = [{"id": i["id"], "file_name": i["file_name"]} for i in coco["images"]]
    keep = [a for a in coco["annotations"] if args.show_people or CATS[a["category_id"]] != "person"]
    boxes = [{"id": a["id"], "image_id": a["image_id"], "bbox": a["bbox"], "cat": CATS[a["category_id"]]} for a in keep]
    saved = json.load(open(SAVE)) if SAVE.exists() else {"added": [], "removed": []}
    return {"images": ims, "boxes": boxes, "saved": saved}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def send(self, body, ctype="application/json"):
        self.send_response(200); self.send_header("Content-Type", ctype); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/": self.send(PAGE.encode(), "text/html; charset=utf-8")
        elif p == "/version": self.send(BUILD.encode(), "text/plain")
        elif p == "/data": self.send(json.dumps(payload()).encode())
        elif p.startswith("/img/"):
            f = D / "images" / os.path.basename(p)
            if f.exists(): self.send(f.read_bytes(), "image/jpeg")
            else: self.send_error(404)
        else: self.send_error(404)

    def do_POST(self):
        if self.path != "/save": self.send_error(404); return
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.headers.get("X-Build") != BUILD: self.send_error(409); return   # a tab from before a restart must never overwrite answers
        try:
            d = json.loads(body)
            assert isinstance(d.get("added"), list) and isinstance(d.get("removed"), list)
        except (ValueError, AssertionError):
            self.send_error(400); return
        SAVE.write_bytes(body)
        self.send(b"{}")


n = len(coco["images"])
print(f"{n} stills. Your additions save to {SAVE}")
srv = ThreadingHTTPServer(("127.0.0.1", args.port), H)
url = f"http://127.0.0.1:{args.port}/"
print("Open", url, "(Ctrl+C to stop; everything is already saved)")
if not args.no_open: threading.Timer(0.8, lambda: webbrowser.open(url)).start()
try: srv.serve_forever()
except KeyboardInterrupt: pass
