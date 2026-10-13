"""Download the M5 data into data/raw.

Default source is the public Nixtla mirror (no Kaggle account needed). It contains the
original competition files plus ``sales_test_evaluation.csv`` (d_1942-d_1969), which this
project uses as an extra held-out window. Alternatively use the Kaggle CLI:

    kaggle competitions download -c m5-forecasting-accuracy -p data/raw && unzip ...
"""

from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
NEEDED = ["calendar.csv", "sales_train_evaluation.csv", "sell_prices.csv", "sales_test_evaluation.csv"]


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    if all((RAW / f).exists() for f in NEEDED):
        print("M5 files already present in", RAW)
        return 0
    zpath = RAW / "m5.zip"
    print("downloading", URL)
    urllib.request.urlretrieve(URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if Path(name).name in NEEDED:
                (RAW / Path(name).name).write_bytes(z.read(name))
    zpath.unlink()
    missing = [f for f in NEEDED if not (RAW / f).exists()]
    if missing:
        print("missing after extraction:", missing)
        return 1
    print("done ->", RAW)
    return 0


if __name__ == "__main__":
    sys.exit(main())
