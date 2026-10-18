"""Start the live training dashboard in the browser.

Example:
    python scripts/dashboard.py            # http://127.0.0.1:8765/
    python scripts/dashboard.py --port 9000 --no-browser
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from uavta.visualization.dashboard import serve


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", default="results/runs")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    serve(Path(args.runs), port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
