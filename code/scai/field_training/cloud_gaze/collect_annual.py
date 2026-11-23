import argparse
import csv, io, zipfile, math, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import requests
from PIL import Image, ImageOps


def read(f):
    return list(csv.DictReader(open(f)))


parser = argparse.ArgumentParser(description="Sample directional GLOBE photographs from an annual archive.")
parser.add_argument("year", type=int, choices=(2019, 2021, 2023))
year = parser.parse_args().year
folder = "annual-field" if year == 2019 else f"annual{year % 100}-field"
sample_limit = {2019: 500, 2021: 350, 2023: 220}[year]
blocked = {
    r["group"]
    for r in read("gaze-clean.csv")
    + pd.read_csv(Path(__file__).with_name("urban-sites.csv.gz")).to_dict("records")
}
if year >= 2021:
    blocked.update(r["group"] for r in read("annual-field.csv"))
if year == 2023:
    blocked.update(r["group"] for r in read("annual21-field.csv") + read("gaze-independent.csv"))
kinds = "Cirrus Cirrocumulus Cirrostratus Altostratus Altocumulus Cumulus Nimbostratus Stratus Stratocumulus Cumulonimbus".split()
z = zipfile.ZipFile(f"annual{year % 100}.zip")
items = []
for r in csv.DictReader(io.TextIOWrapper(z.open(z.namelist()[0]))):
    if r["is Citizen Science"] != "1" or r["Total Cloud Cover"].lower() not in (
        "broken",
        "overcast",
        "scattered",
    ):
        continue
    tags = [k for k in kinds if r[k] == "1"]
    if len(tags) != 1:
        continue
    site = f"{math.floor(float(r['Observation Latitude']) * 4)}_{math.floor(float(r['Observation Longitude']) * 4)}"
    if site in blocked:
        continue
    urls = [
        r["Ground Image " + d]
        for d in ["North", "South", "East", "West"]
        if "https://data.globe.gov/" in r["Ground Image " + d]
    ]
    if not urls:
        continue
    u = random.Random(r["Observation Id"]).choice(urls)
    i = u.split("/")[-2]
    items.append(
        dict(
            file=folder + "/" + i + ".jpg",
            url=u,
            group=site,
            observer=r["Observation Id"].split("-")[-2],
            genus=tags[0],
            label=int(tags == ["Stratocumulus"]),
            cover=r["Total Cloud Cover"],
        )
    )
random.Random({2019: 77291, 2021: 77292, 2023: 77293}[year]).shuffle(items)
groups = set()
people = set()
ct = [0, 0]
rows = []
for r in items:
    if r["group"] in groups or r["observer"] in people or ct[r["label"]] >= sample_limit:
        continue
    rows.append(r)
    groups.add(r["group"])
    people.add(r["observer"])
    ct[r["label"]] += 1
print("selected", ct, flush=True)
Path(folder).mkdir(exist_ok=True)


def fetch(r):
    try:
        res = requests.get(r["url"], timeout=26)
        res.raise_for_status()
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(res.content))).convert(
            "RGB"
        )
        img.thumbnail((720, 720))
        img.save(r["file"], quality=93)
        return r
    except (OSError, requests.RequestException):
        pass


with open(folder + ".csv", "w") as f:
    writer = csv.DictWriter(f, rows[0].keys())
    writer.writeheader()
    with ThreadPoolExecutor(8) as ex:
        for row in ex.map(fetch, rows):
            if row:
                writer.writerow(row)
                f.flush()
