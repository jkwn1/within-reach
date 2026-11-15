"""Step 9: build the Within Reach College Finder, a single self-contained web page (docs/index.html).

It embeds results/finder_data.json (written by step 5b) into src/finder_template.html, so the page
works offline, can be hosted on GitHub Pages from the docs/ folder, and always matches the results.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # import common.py from any working dir
from common import RESULTS, ROOT  # noqa: E402


def main():
    data = json.loads((RESULTS / "finder_data.json").read_text())
    template = (Path(__file__).resolve().parent / "finder_template.html").read_text()
    payload = (json.dumps(data, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e")
               .replace("&", "\\u0026"))  # safe inside <script>
    page = template.replace("/*__DATA__*/", payload).replace("__N__", f"{len(data['colleges']):,}")
    out = ROOT / "docs"
    out.mkdir(exist_ok=True)
    (out / "index.html").write_text(page)
    print(f"wrote docs/index.html: {len(data['colleges'])} colleges, {len(page) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
