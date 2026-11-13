"""Render aggregate-only reports into the shared reports directory."""
import argparse
from pathlib import Path
import os
import re
from urllib.parse import urlsplit, unquote
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPORTS = {
    "h4-ja": ("analysis/h4/notebooks/h4_hypothesis_ja.qmd", "h4_ja.html"),
    "h4-en": ("analysis/h4/notebooks/h4_hypothesis_en.qmd", "h4_en.html"),
    "supplementary": ("analysis/ratings/supplementary/report.qmd", "supplementary.html"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", choices=REPORTS)
    args = parser.parse_args()
    for name in ([args.report] if args.report else REPORTS):
        source, output = REPORTS[name]
        source = ROOT / source
        print(f"Rendering {name}", flush=True)
        subprocess.run(["quarto", "render", source.name, "--to", "html", "--output", output],
                       cwd=source.parent, check=True)
        rendered = source.parent / output
        destination = ROOT / "reports" / output
        def relocate_link(match):
            target = match[1]
            url = urlsplit(target)
            if url.scheme or url.netloc or not url.path:
                return match[0]
            resolved = (source.parent / unquote(url.path)).resolve()
            if not resolved.is_relative_to(ROOT):
                return match[0]
            href = os.path.relpath(resolved, destination.parent)
            if url.fragment:
                href += "#" + url.fragment
            return 'href="' + href + '"'
        content = re.sub(r'href="([^"\n]+)"', relocate_link, rendered.read_text())
        destination.write_text("\n".join(line.rstrip() for line in content.splitlines()) + "\n")
        rendered.unlink()


if __name__ == "__main__":
    main()
