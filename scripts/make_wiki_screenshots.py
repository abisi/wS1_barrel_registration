"""Screenshots of the review page for the wiki (headless Chrome via Playwright).

usage: python scripts/make_wiki_screenshots.py PAGE_DIR --out <wiki clone>/images [--db DB_DIR]

The wiki is the GitHub wiki of this repository (git clone https://github.com/abisi/wS1_barrel_registration.wiki.git);
commit and push the new images there.

PAGE_DIR: a built review page (scripts/review_gui.py --out). --db: decisions / seeds to show (picks/, seeds/ with
one JSON per brain, e.g. PAGE_DIR/db); they are served read-only, nothing is written. Needs `pip install playwright`
and Google Chrome (channel "chrome"; or run `playwright install chromium` and drop the channel).
The brains shown (AB140, MH038cmdii, AB126, AB077) are from the AB / MH dataset: change BRAINS for another one.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

from playwright.sync_api import sync_playwright

BRAINS = dict(compare="AB140", seed="MH038cmdii", seed_plain="AB126", dark="AB077")
PORT = 8765


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("page_dir")
    ap.add_argument("--db")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = {c: {} for c in ("picks", "seeds", "status", "requests")}
    if a.db:
        for c in D:
            for f in glob.glob(os.path.join(a.db, c, "*.json")):
                x = json.load(open(f))
                D[c][x.get("brain") or os.path.basename(f)[:-5]] = x
    stub = ("window.claude = {use: async (n) => { if (n !== 'db') return null; const D = %s; return {"
            "collection: (c) => ({onSnapshot(cb) { setTimeout(() => cb({docs: Object.entries(D[c] || {}).map(([id, x]) => "
            "({id, data: () => x}))}), 50); return () => {}; }}), doc: () => ({set: async () => {}})}; }};" % json.dumps(D))
    os.makedirs(a.out, exist_ok=True)
    srv = subprocess.Popen([sys.executable, "-m", "http.server", str(PORT), "-d", a.page_dir],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5)
    try:
        with sync_playwright() as p:
            br = p.chromium.launch(channel="chrome")

            def page(scheme):
                pg = br.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5, color_scheme=scheme)
                pg.add_init_script(stub)
                pg.goto(f"http://localhost:{PORT}/index.html")
                pg.wait_for_timeout(2500)
                return pg

            def brain(pg, b, tab=None):
                pg.click("#filters .chip[data-f='all']")
                pg.wait_for_timeout(300)
                pg.click(f"#list button.brain:has-text('{b}')")
                pg.wait_for_timeout(2500)
                if tab:
                    pg.click(f"button[data-t='{tab}']")
                    pg.wait_for_timeout(2500)

            out = lambda n: os.path.join(a.out, n)
            pg = page("light")
            brain(pg, BRAINS["compare"])
            pg.screenshot(path=out("01_compare_AB140.png"))
            pg.locator("#side").screenshot(path=out("02_fits_table.png"))
            pg.locator("nav").screenshot(path=out("03_brain_list.png"))
            brain(pg, BRAINS["seed"], "seed")
            pg.screenshot(path=out("04_seed_MH038cmdii.png"))
            for _ in range(2):
                pg.keyboard.press("h")
                pg.wait_for_timeout(1500)
            pg.screenshot(path=out("05_seed_sharpened.png"))
            pg.locator("#side").screenshot(path=out("06_seed_panel.png"))
            brain(pg, BRAINS["seed_plain"], "seed")
            pg.screenshot(path=out("07_seed_AB126.png"))
            brain(pg, BRAINS["seed"], "compare")  # Compare tab of the seeded brain: seeded and sharpened refits next to the first pass
            pg.screenshot(path=out("09_compare_seeded_MH038cmdii.png"))
            pg2 = page("dark")
            brain(pg2, BRAINS["dark"])
            pg2.screenshot(path=out("08_compare_dark_AB077.png"))
    finally:
        srv.terminate()
    print("screenshots in", a.out)


if __name__ == "__main__":
    main()
