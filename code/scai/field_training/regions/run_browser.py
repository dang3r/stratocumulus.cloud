"""Score a CSV manifest with columns file,label,panel using the PR's photo model."""

import argparse
import csv
import functools
import http.server
import json
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

here = Path(__file__).resolve().parent
repo = here.parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("manifest", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--chromium")
parser.add_argument(
    "--runtime", type=Path, help="Local onnxruntime-web 1.22.0 dist directory"
)
args = parser.parse_args()
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(repo))
with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=args.chromium)
        page = browser.new_page()
        if args.runtime:
            from urllib.parse import urlparse

            def runtime(route):
                path = args.runtime / Path(urlparse(route.request.url).path).name
                route.fulfill(
                    path=path,
                    headers={
                        "Access-Control-Allow-Origin": "*",
                        "Content-Type": "application/wasm"
                        if path.suffix == ".wasm"
                        else "text/javascript",
                    },
                )

            page.route("https://cdn.jsdelivr.net/**", runtime)
        page.goto(f"http://127.0.0.1:{server.server_port}/")
        page.add_script_tag(path=here / "multiview.js")
        with args.manifest.open() as source, args.output.open("w") as output:
            for row in csv.DictReader(source):
                image = args.manifest.parent / row["file"]
                page.route("**/sample.jpg*", lambda route: route.fulfill(path=image))
                result = page.evaluate("""async () => {
                    const im = new Image();
                    im.src = '/sample.jpg?' + Math.random();
                    await im.decode();
                    const t = performance.now();
                    const result = await measureRegions(im);
                    return {...result, ms: performance.now() - t};
                }""")
                page.unroute("**/sample.jpg*")
                output.write(json.dumps({**row, **result}) + "\n")
                output.flush()
        browser.close()
    server.shutdown()
