"""Serve the review page locally and keep its decisions and seeds as JSON files (no claude.ai needed).

usage: python scripts/review_server.py <page_dir> [--port 8000] [--host 127.0.0.1]

<page_dir> is the output of review_gui.py (index.html, manifest.json, brains/, img/). Open http://localhost:8000.
The page saves to <page_dir>/db/<collection>/<id>.json (collections picks, seeds, requests, status), one file per
brain, written atomically. Python standard library only.

Remote machine (e.g. the page built on haas): run the server there and forward the port,
  ssh -L 8000:localhost:8000 haas   then open http://localhost:8000 on your own computer.

API used by the page: GET api/ping, GET api/<collection> -> {id: doc}, PUT api/<collection>/<id> (JSON body).
"""
import argparse
import json
import os
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

COLLECTIONS = ("picks", "seeds", "requests", "status")
NAME = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")


class Handler(SimpleHTTPRequestHandler):
    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def _parts(self):
        p = self.path.split("?")[0].strip("/").split("/")
        return p[1:] if p and p[0] == "api" else None

    def do_GET(self):
        p = self._parts()
        if p is None:
            return super().do_GET()
        if p == ["ping"]:
            return self._json(200, {"ok": True})
        if len(p) == 1 and p[0] in COLLECTIONS:
            d = os.path.join(self.directory, "db", p[0])
            out = {}
            if os.path.isdir(d):
                for f in sorted(os.listdir(d)):
                    if f.endswith(".json"):
                        with open(os.path.join(d, f)) as fh:
                            out[f[:-5]] = json.load(fh)
            return self._json(200, out)
        return self._json(404, {"error": "not found"})

    def do_PUT(self):
        p = self._parts()
        if not p or len(p) != 2 or p[0] not in COLLECTIONS or not NAME.match(p[1]):
            return self._json(400, {"error": "bad path"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            assert isinstance(body, dict)
        except Exception:
            return self._json(400, {"error": "bad JSON"})
        d = os.path.join(self.directory, "db", p[0])
        os.makedirs(d, exist_ok=True)
        f = os.path.join(d, p[1] + ".json")
        with open(f + ".tmp", "w") as fh:
            json.dump(body, fh, indent=1)
        os.replace(f + ".tmp", f)
        return self._json(200, {"ok": True})

    def end_headers(self):
        if self.path.endswith((".json", ".html", "/")):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("page_dir")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), partial(Handler, directory=os.path.abspath(a.page_dir)))
    print(f"review page: http://localhost:{a.port}   decisions -> {os.path.abspath(a.page_dir)}/db/   (Ctrl+C stops)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
