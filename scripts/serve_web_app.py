#!/usr/bin/env python3
"""Serve the browser app (and static model files) over HTTP."""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    web_root = repo_root / "web-app"

    if not web_root.exists():
        raise SystemExit(f"Missing web app directory: {web_root}")

    handler = partial(SimpleHTTPRequestHandler, directory=str(web_root))
    server = ThreadingHTTPServer(("127.0.0.1", 8000), handler)

    print("Serving web app at http://127.0.0.1:8000")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
