#!/usr/bin/env python3
"""Tiny LAN log viewer. Serves a mobile-friendly page that polls a file and
live-updates, auto-scrolling to the bottom. Usage: serve_log.py [file] [port]"""
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ds4_max_reasoning.txt"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8090

PAGE = """<!doctype html><html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<title>ds4 · live reasoning</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; background:#0d1117; color:#c9d1d9;
         font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  header { position:sticky; top:0; z-index:5; background:#161b22; border-bottom:1px solid #30363d;
           padding:10px 14px; display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
  header b { font-size:15px; }
  .dot { width:10px; height:10px; border-radius:50%; background:#3fb950;
         box-shadow:0 0 8px #3fb950; animation:pulse 1.4s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
  .meta { color:#8b949e; font-size:12px; margin-left:auto; }
  pre { white-space:pre-wrap; word-wrap:break-word; overflow-wrap:anywhere;
        margin:0; padding:14px 14px 96px; font:14px/1.6 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
  .think { color:#8b949e; }
  .answer { color:#e6edf3; }
  .banner { display:block; color:#58a6ff; font-weight:700; margin:18px 0 8px; }
  .ans-banner { color:#3fb950; }
  #jump { position:fixed; right:14px; bottom:18px; background:#238636; color:#fff; border:none;
          border-radius:24px; padding:11px 16px; font-size:14px; box-shadow:0 3px 10px rgba(0,0,0,.5);
          display:none; }
</style></head><body>
<header><span class="dot" id="dot"></span><b>ds4 · max reasoning</b>
  <span class="meta" id="meta">connecting…</span></header>
<pre id="log"></pre>
<button id="jump" onclick="toBottom(true)">↓ live</button>
<script>
const logEl=document.getElementById('log'), metaEl=document.getElementById('meta'),
      dot=document.getElementById('dot'), jump=document.getElementById('jump');
let stuck=true;
function nearBottom(){return window.innerHeight+window.scrollY >= document.body.offsetHeight-80;}
function toBottom(force){ if(force) stuck=true; window.scrollTo(0,document.body.scrollHeight); jump.style.display='none'; }
window.addEventListener('scroll',()=>{ stuck=nearBottom(); jump.style.display=stuck?'none':'block'; });
function render(t){
  let h=html_escape(t);
  h=h.replace(/#+\\s*THINKING\\s*#+/g,'<span class="banner">— THINKING —</span>');
  h=h.replace(/#+\\s*ANSWER\\s*#+/g,'<span class="banner ans-banner">— ANSWER —</span>');
  return h;
}
function html_escape(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
async function tick(){
  try{
    const r=await fetch('/log?t='+Date.now());
    const t=await r.text();
    const done=r.headers.get('X-Done')==='1';
    logEl.innerHTML=render(t);
    const kb=(t.length/1024).toFixed(1);
    metaEl.textContent=kb+' KB · '+(done?'finished':'streaming')+' · '+new Date().toLocaleTimeString();
    if(done){ dot.style.background='#8b949e'; dot.style.animation='none'; }
    if(stuck) toBottom(false);
  }catch(e){ metaEl.textContent='offline · retrying'; }
}
tick(); setInterval(tick, 1000);
</script></body></html>"""

class H(BaseHTTPRequestHandler):
    def log_message(self, format, *args): pass  # silence request logging
    def _send(self, code, body, ctype, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        if self.path.startswith("/log"):
            try:
                with open(LOG, "rb") as f:
                    data = f.read()
            except FileNotFoundError:
                data = b"(waiting for log to appear...)"
            # "done" heuristic: footer line written by the streamer
            done = b"done in " in data[-400:]
            self._send(200, data, "text/plain; charset=utf-8", {"X-Done": "1" if done else "0"})
        else:
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")

if __name__ == "__main__":
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), H)
    print(f"serving {LOG} on http://0.0.0.0:{PORT}", flush=True)
    srv.serve_forever()
