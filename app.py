"""
TRAFFIC-AWARE PATHFINDING SYSTEM USING ARTIFICIAL INTELLIGENCE
Run:  pip install flask   ->   python app.py   ->   open http://127.0.0.1:5000
"""
import heapq, math, zlib
from flask import Flask, jsonify, request, render_template_string

app = Flask(__name__)

# ------------------------------------------------------------------ CITY MODEL
NODES = {
    "A": ("North Gate", 0.5, 0.5),   "B": ("Tech Park", 3.0, 0.4),
    "C": ("Lake View", 6.0, 0.6),    "D": ("Airport", 8.8, 0.5),
    "E": ("Old Town", 0.4, 3.0),     "F": ("City Centre", 3.2, 3.0),
    "G": ("Market Square", 6.1, 2.9),"H": ("Stadium", 8.9, 3.1),
    "I": ("University", 0.6, 5.6),   "J": ("Central Station", 3.0, 5.7),
    "K": ("Hospital", 6.0, 5.5),     "L": ("Harbour", 8.7, 5.6),
}
# (u, v, base speed km/h, rush-hour sensitivity 0..1)
EDGES = [
    ("A","B",50,.5),("B","C",50,.5),("C","D",60,.4),
    ("E","F",40,.9),("F","G",40,1.0),("G","H",40,.8),
    ("I","J",50,.5),("J","K",50,.5),("K","L",60,.4),
    ("A","E",40,.6),("E","I",40,.6),("B","F",40,.9),("F","J",40,.9),
    ("C","G",40,.8),("G","K",40,.8),("D","H",50,.5),("H","L",50,.5),
    ("A","F",70,.3),("F","K",70,.35),("C","H",70,.3),   # ring/express roads
]
MAX_SPEED = 70.0

def dist(a, b):
    return math.hypot(NODES[a][1] - NODES[b][1], NODES[a][2] - NODES[b][2])

ADJ = {k: [] for k in NODES}
EDGE = {}
for u, v, spd, sens in EDGES:
    d = round(dist(u, v), 2)
    for a, b in ((u, v), (v, u)):
        ADJ[a].append(b)
        EDGE[(a, b)] = dict(dist=d, speed=spd, sens=sens)

# ------------------------------------------------- AI: CONGESTION PREDICTION
def _g(x, mu, sd):
    return math.exp(-((x - mu) ** 2) / (2 * sd ** 2))

def predict_congestion(u, v, minute):
    """
    Lightweight predictive model (0..1 congestion).
    = learned daily profile (morning + evening peaks, scaled by road sensitivity)
      + small per-road, per-15-min-slot variation (stands in for live sensor data).
    Replace with a trained ML model (e.g. LSTM / gradient boosting) using real data.
    """
    h = (minute % 1440) / 60.0
    e = EDGE[(u, v)]
    profile = 0.12 + e["sens"] * (0.75 * _g(h, 8.5, 1.3) + 0.85 * _g(h, 17.5, 1.6) + 0.35 * _g(h, 13, 2))
    key = f"{min(u,v)}{max(u,v)}{int(minute // 15)}"
    noise = (zlib.crc32(key.encode()) % 1000) / 1000 * 0.12
    return max(0.0, min(1.0, profile + noise))

def level(c):
    return "Low" if c < .30 else "Medium" if c < .50 else "Heavy" if c < .70 else "High Congestion"

def travel_min(u, v, depart_min):
    e = EDGE[(u, v)]
    c = predict_congestion(u, v, depart_min)
    speed = max(6.0, e["speed"] * (1 - 0.78 * c))
    return e["dist"] / speed * 60, c

# ------------------------------------------------------ TIME-DEPENDENT A*
def td_astar(start, goal, t0):
    """Edge cost depends on the clock time at which the edge is entered."""
    h = lambda n: dist(n, goal) / MAX_SPEED * 60      # admissible heuristic
    pq = [(h(start), 0.0, start)]
    best, prev, expanded = {start: 0.0}, {}, 0
    while pq:
        _, g, u = heapq.heappop(pq)
        if g > best.get(u, 1e18):
            continue
        expanded += 1
        if u == goal:
            break
        for v in ADJ[u]:
            dt, _ = travel_min(u, v, t0 + g)
            ng = g + dt
            if ng < best.get(v, 1e18):
                best[v], prev[v] = ng, u
                heapq.heappush(pq, (ng + h(v), ng, v))
    return rebuild(prev, start, goal), expanded

def shortest_distance(start, goal):
    """Classic Dijkstra on distance only (ignores traffic) – the baseline."""
    pq, best, prev = [(0.0, start)], {start: 0.0}, {}
    while pq:
        d, u = heapq.heappop(pq)
        if d > best.get(u, 1e18):
            continue
        if u == goal:
            break
        for v in ADJ[u]:
            nd = d + EDGE[(u, v)]["dist"]
            if nd < best.get(v, 1e18):
                best[v], prev[v] = nd, u
                heapq.heappush(pq, (nd, v))
    return rebuild(prev, start, goal)

def rebuild(prev, s, g):
    p = [g]
    while p[-1] != s:
        p.append(prev[p[-1]])
    return p[::-1]

def evaluate(path, t0):
    t, km, wsum, steps = 0.0, 0.0, 0.0, []
    for u, v in zip(path, path[1:]):
        dt, c = travel_min(u, v, t0 + t)
        d = EDGE[(u, v)]["dist"]
        steps.append(dict(frm=NODES[u][0], to=NODES[v][0], km=d, min=round(dt, 1),
                          congestion=round(c * 100), level=level(c)))
        t += dt; km += d; wsum += c * d
    avg = wsum / km if km else 0
    return dict(path=path, names=[NODES[n][0] for n in path], km=round(km, 2),
                minutes=round(t, 1), traffic=level(avg), avg_congestion=round(avg * 100),
                steps=steps)

# ------------------------------------------------------------------- API
def parse_time(s):
    try:
        hh, mm = s.split(":"); return int(hh) * 60 + int(mm)
    except Exception:
        return 8 * 60 + 30

@app.route("/api/graph")
def api_graph():
    t = parse_time(request.args.get("time", "08:30"))
    edges = []
    for u, v, *_ in EDGES:
        c = predict_congestion(u, v, t)
        edges.append(dict(u=u, v=v, c=round(c, 2), level=level(c)))
    nodes = {k: dict(name=n[0], x=n[1], y=n[2]) for k, n in NODES.items()}
    return jsonify(nodes=nodes, edges=edges)

@app.route("/api/route")
def api_route():
    s, g = request.args.get("start"), request.args.get("end")
    t = parse_time(request.args.get("time", "08:30"))
    if s not in NODES or g not in NODES or s == g:
        return jsonify(error="Choose two different locations."), 400
    best_path, expanded = td_astar(s, g, t)
    best = evaluate(best_path, t)
    short = evaluate(shortest_distance(s, g), t)
    saved = round(short["minutes"] - best["minutes"], 1)
    return jsonify(best=best, shortest=short, time_saved=max(0, saved),
                   expanded=expanded, nodes_total=len(NODES))

@app.route("/")
def home():
    return render_template_string(PAGE, nodes=NODES)

# ------------------------------------------------------------------- UI
PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Traffic-Aware Pathfinding System</title>
<style>
:root{--bg:#0f172a;--card:#1e293b;--tx:#e2e8f0;--mut:#94a3b8;--ac:#38bdf8;
--g:#22c55e;--y:#eab308;--o:#f97316;--r:#ef4444}
*{box-sizing:border-box}body{margin:0;font-family:system-ui,Arial,sans-serif;background:var(--bg);color:var(--tx)}
header{padding:70px 20px;text-align:center;background:linear-gradient(135deg,#0ea5e9,#6366f1)}
header h1{margin:0 0 8px;font-size:clamp(26px,5vw,44px)}header p{margin:0 0 24px;opacity:.9}
button{background:var(--ac);color:#0f172a;border:0;padding:12px 24px;border-radius:8px;font-weight:700;font-size:16px;cursor:pointer}
button:hover{filter:brightness(1.1)}
main{max-width:1000px;margin:auto;padding:20px;display:none}
section{background:var(--card);border-radius:12px;padding:20px;margin-bottom:20px}
h2{margin-top:0;color:var(--ac);font-size:20px}
.legend{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:10px;font-size:14px}
.legend i{display:inline-block;width:14px;height:14px;border-radius:3px;margin-right:6px;vertical-align:-2px}
svg{width:100%;background:#0b1220;border-radius:8px}
.row{display:flex;gap:14px;flex-wrap:wrap;align-items:end}
.row label{display:flex;flex-direction:column;font-size:14px;color:var(--mut);gap:4px;flex:1;min-width:150px}
select,input{padding:10px;border-radius:8px;border:1px solid #334155;background:#0f172a;color:var(--tx);font-size:15px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:14px 0}
.card{background:#0f172a;border-radius:10px;padding:14px}.card small{color:var(--mut)}
.card b{display:block;font-size:22px;margin-top:4px}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:8px;text-align:left;border-bottom:1px solid #334155}
th{color:var(--mut)}.pill{padding:2px 10px;border-radius:99px;color:#0f172a;font-weight:700;font-size:12px}
ol{line-height:1.7;color:var(--mut)}#out{display:none}.good{color:var(--g)}
</style></head><body>

<header id="home"><h1>🚦 Traffic-Aware Pathfinding System</h1>
<p>Using Artificial Intelligence · Time-Dependent A* · Congestion Prediction</p>
<button onclick="startNav()">Start Navigation</button></header>

<main id="app">
<section><h2>Live Traffic Map</h2>
 <div class="legend"><span><i style="background:var(--g)"></i>Green – Low Traffic</span>
 <span><i style="background:var(--y)"></i>Yellow – Medium Traffic</span>
 <span><i style="background:var(--o)"></i>Orange – Heavy Traffic</span>
 <span><i style="background:var(--r)"></i>Red – High Congestion</span></div>
 <svg id="map" viewBox="-40 -40 980 700"></svg></section>

<section><h2>Route Selection</h2><div class="row">
 <label>Starting Point<select id="s">{% for k,n in nodes.items() %}<option value="{{k}}">{{n[0]}}</option>{% endfor %}</select></label>
 <label>Destination<select id="e">{% for k,n in nodes.items() %}<option value="{{k}}" {% if k=='L' %}selected{% endif %}>{{n[0]}}</option>{% endfor %}</select></label>
 <label>Departure Time<input id="t" type="time" value="08:30" onchange="loadMap()"></label>
 <button onclick="findRoute()">Find Best Route</button></div></section>

<div id="out">
<section><h2>AI-Based Route Prediction</h2>
 <ol>
  <li>Traffic congestion predicted for every road at the departure time (<span id="pt"></span>).</li>
  <li>Time-dependent A*: each road's travel time is evaluated at the moment it will be entered.</li>
  <li>Search explored <b id="ex"></b> of <b id="tot"></b> intersections using an admissible distance/speed heuristic.</li>
  <li>Best route identified and compared with the shortest-distance route.</li></ol>
 <table id="steps"></table></section>

<section><h2>Final Output</h2>
 <p><b>Recommended Route:</b> <span id="rt"></span></p>
 <div class="cards">
  <div class="card"><small>Total Distance</small><b id="dk"></b></div>
  <div class="card"><small>Estimated Travel Time</small><b id="tm"></b></div>
  <div class="card"><small>Traffic Level</small><b id="tl"></b></div>
  <div class="card"><small>Time Saved</small><b id="sv" class="good"></b></div></div>
 <h3>Comparison with Shortest-Distance Route</h3>
 <table><tr><th></th><th>AI Best Route</th><th>Shortest Distance</th></tr>
  <tr><td>Route</td><td id="c1r"></td><td id="c2r"></td></tr>
  <tr><td>Distance</td><td id="c1d"></td><td id="c2d"></td></tr>
  <tr><td>Travel Time</td><td id="c1t"></td><td id="c2t"></td></tr>
  <tr><td>Traffic Level</td><td id="c1l"></td><td id="c2l"></td></tr></table></section></div>
</main>

<script>
const COL={"Low":"#22c55e","Medium":"#eab308","Heavy":"#f97316","High Congestion":"#ef4444"};
const SC=100,$=id=>document.getElementById(id);let G=null;
function startNav(){$("app").style.display="block";$("app").scrollIntoView({behavior:"smooth"});loadMap()}
async function loadMap(route){
  G=await (await fetch("/api/graph?time="+$("t").value)).json();draw(route)}
function P(k){return [G.nodes[k].x*SC,G.nodes[k].y*SC+20]}
function draw(r){
  let h="";
  G.edges.forEach(e=>{const[a,b]=[P(e.u),P(e.v)];
   h+=`<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="${COL[e.level]}" stroke-width="7" stroke-linecap="round"><title>${e.level} (${Math.round(e.c*100)}%)</title></line>`});
  const poly=(p,c,d)=>{if(!p)return"";return`<polyline fill="none" stroke="${c}" stroke-width="3" stroke-dasharray="${d}" points="${p.map(k=>P(k).join(",")).join(" ")}"/>`};
  if(r){h+=poly(r.shortest.path,"#c084fc","8 6");h+=poly(r.best.path,"#38bdf8","")}
  for(const k in G.nodes){const[x,y]=P(k);
   h+=`<circle cx="${x}" cy="${y}" r="11" fill="#0f172a" stroke="#fff" stroke-width="2"/><text x="${x}" y="${y-18}" fill="#e2e8f0" font-size="14" text-anchor="middle">${G.nodes[k].name}</text>`}
  h+=`<text x="0" y="650" fill="#38bdf8" font-size="14">━ AI best route</text><text x="150" y="650" fill="#c084fc" font-size="14">╌ shortest-distance route</text>`;
  $("map").innerHTML=h}
const pill=l=>`<span class="pill" style="background:${COL[l]}">${l}</span>`;
async function findRoute(){
  const q=`start=${$("s").value}&end=${$("e").value}&time=${$("t").value}`;
  const r=await (await fetch("/api/route?"+q)).json();
  if(r.error){alert(r.error);return}
  await loadMap(r);$("out").style.display="block";
  $("pt").textContent=$("t").value;$("ex").textContent=r.expanded;$("tot").textContent=r.nodes_total;
  $("steps").innerHTML="<tr><th>Road segment</th><th>Km</th><th>Min</th><th>Predicted congestion</th></tr>"+
   r.best.steps.map(s=>`<tr><td>${s.frm} → ${s.to}</td><td>${s.km}</td><td>${s.min}</td><td>${s.congestion}% ${pill(s.level)}</td></tr>`).join("");
  $("rt").textContent=r.best.names.join(" → ");
  $("dk").textContent=r.best.km+" km";$("tm").textContent=r.best.minutes+" min";
  $("tl").innerHTML=pill(r.best.traffic);$("sv").textContent=r.time_saved+" min";
  $("c1r").textContent=r.best.names.join(" → ");$("c2r").textContent=r.shortest.names.join(" → ");
  $("c1d").textContent=r.best.km+" km";$("c2d").textContent=r.shortest.km+" km";
  $("c1t").textContent=r.best.minutes+" min";$("c2t").textContent=r.shortest.minutes+" min";
  $("c1l").innerHTML=pill(r.best.traffic);$("c2l").innerHTML=pill(r.shortest.traffic);
  $("out").scrollIntoView({behavior:"smooth"})}
</script></body></html>"""

if __name__ == "__main__":
    app.run(debug=True)
