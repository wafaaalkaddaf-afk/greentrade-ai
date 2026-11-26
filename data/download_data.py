"""Download the UCI Steel Industry Energy Consumption dataset (id 851).

Usage: python data/download_data.py
Saves data/raw/Steel_industry_data.csv
If the download fails, download the zip manually from
https://archive.ics.uci.edu/dataset/851 and put the CSV in data/raw/.
"""
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/851/steel+industry+energy+consumption.zip"
OUT = Path(__file__).resolve().parent / "raw"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Downloading ...")
    with urllib.request.urlopen(URL, timeout=60) as r:
        buf = io.BytesIO(r.read())
    with zipfile.ZipFile(buf) as z:
        z.extractall(OUT)
        print("Extracted:", z.namelist())
    csvs = list(OUT.glob("*.csv"))
    if csvs and csvs[0].name != "Steel_industry_data.csv":
        csvs[0].rename(OUT / "Steel_industry_data.csv")
    print("Done ->", OUT / "Steel_industry_data.csv")


if __name__ == "__main__":
    main()
