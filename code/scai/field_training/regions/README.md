# Can sky crops recover stratocumulus behind buildings?

This experiment runs the PR's frozen photo classifier on the full image and up to three sky windows. The sky model proposes one rectangle per scale on its 28 × 28 output grid. Selection favors high mean sky probability and penalizes overlap. Each crop comes from the original-resolution photo and receives its own sky mask and cloud score. The frame score is the larger of the full-frame score and the average of the full-frame and best crop scores. A foreground rejection contributes zero. The browser demo continues to use its existing decision rule.

## Results

Both operating points produced 7 false positives among 80 development negatives. The full-frame threshold was 0.25. The crop threshold was 0.451965 (rounded), the lowest *observed development-negative score* meeting that budget. Thresholds were frozen for the transfer comparison. Different thresholds between adjacent negative scores can give the same development false-positive count.

| Panel | Positives / negatives | Full-frame detections / false positives | Multi-crop detections / false positives |
|---|---:|---:|---:|
| Development phone photos | 19 / 80 | 13 / 7 | 11 / 7 |
| New phone photos, observation tags | 32 / 30 | 11 / 6 | 9 / 4 |
| Visually labeled subset of new photos | 5 / 20 | 2 / 2 | 2 / 1 |
| Earlier real-building phone slice | 9 / 31 | 6 / 10 | 6 / 9 |
| Building composites, reviewed donors | 20 / 20 | 0 / 0 | 1 / 0 |
| Building composites, whole-donor tags | 20 / 20 | 0 / 0 | 1 / 1 |

The two smallest-sky reviewed pairs had at most 20% sky. Both methods missed both positives. On the 62 new photos, median combined inference time was 1.79 seconds and the 95th percentile was 3.16 seconds. Runs used desktop Chromium, with some other benchmarking running concurrently. No new video or physical-phone timing was collected.

As a descriptive ranking check, choosing a fresh threshold for each test panel after scoring would recover 4 of 5 reviewed positives with at most two false positives, and 8 of 9 earlier building positives with at most ten. On the larger observation-tagged panel the best count at six false positives was 10 of 32 (full frame 11). These retrospective cutoffs cannot be used to claim a transferable improvement.

## Photos and labels

[phone-regions.csv](phone-regions.csv) lists 62 directional photos collected with GLOBE Observer. The sampling targeted observations reporting cloud genera on eight dates in 2024–25, balanced by whether observers reported stratocumulus. The records were restricted to one per contributor and site within the draw. A difference-hash check excluded near duplicates of previous panels. This deliberately enriched sample does not estimate how often stratocumulus appears in everyday camera use. It includes buildings, trees, obstructions, sky-only views and at least one letterboxed image. The acquisition protocol directs people to photograph the sky. Incidental street video, indoor scenes and nighttime use require separate evaluation.

The visual review used indexed, resized contact sheets. Labels were recorded before inspecting prediction scores. A single AI reviewer selected one illustrative rectangle in each of 25 photos. Each rectangle uses fractional left, top, right and bottom coordinates relative to the EXIF-oriented image. Five show likely stratocumulus, 16 show another cloud appearance and four show clear sky. The other 37 photos are indeterminate. They remain available for independent annotation. The boxes describe visible examples and do not trace cloud boundaries or establish total cloud coverage. The observer's genus list covers the surrounding sky and is retained separately from the visual label. These provisional labels have no independent meteorological confirmation.

[region-pairs.csv](region-pairs.csv) identifies 20 building scenes from ADE20K and 40 source photographs from the WMO Cloud Atlas. The first synthetic panel pastes the upper 60% of each donor into ADE's labeled sky pixels. Inspection found that several nominal positives contain mostly clear sky after cropping. Their donor labels serve only as a stress test of weak supervision.

[reviewed-panels.csv](reviewed-panels.csv) identifies a second set of 40 composites on the same backgrounds. Every building appears once with a reviewed positive patch and once with a reviewed clear or other-cloud patch. Only the ADE sky pixels change. Five positive donors repeat four times each. Compositing can distort cloud scale, and the ADE sky outline specifies where donor pixels are visible. It does not establish the genus of every pixel. The synthetic results share donors with the real-photo review and do not constitute independent confirmation.

GLOBE photo URLs and WMO source URLs are in the manifests. The ADE images and annotations are available through the parent recipe. Image redistribution is unnecessary for these experiments. [GLOBE Observer Clouds](https://observer.globe.gov/do-globe-observer/clouds) · [WMO Cloud Atlas](https://cloudatlas.wmo.int/) · [ADE20K](https://huggingface.co/datasets/zhoubolei/scene_parse_150)

## Reproduce

Run the following from a data directory with the ADE validation parquet described in the parent training README. Set REGIONS to this checkout's code/scai/field_training/regions directory. Python dependencies are pandas, numpy, requests, Pillow, pyarrow and playwright. Install Playwright Chromium or supply a Chromium executable using the runner's --chromium option. The runner uses the repository page and ONNX files and accepts a local onnxruntime-web 1.22.0 dist directory via --runtime.

```sh
python "$REGIONS/prepare.py" ade.parquet
python "$REGIONS/run_browser.py" phone-panels.csv phone-results.jsonl
python "$REGIONS/run_browser.py" reviewed-panels.csv paired-results.jsonl
python "$REGIONS/run_browser.py" oracle-panels.csv oracle-results.jsonl
python "$REGIONS/summarize.py"
```

The last command summarizes the archived per-image browser scores. Each record includes the selected windows in grid coordinates, individual crop scores, full and combined scores, and end-to-end inference time. The reported latency excludes image decoding and is specific to this Linux Chromium host.

## Where the pipeline loses the target

On the 20 reviewed-positive building composites, the proposal stage returned a window on 14. In 13 of the 20, at least one proposed window contained 80% or more true ADE sky. Only five had any crop score reach the original 0.25 cutoff. The 50/50 blend further suppresses a strong patch when the full frame scores near zero. Both proposal failures and classification failures contribute.

Feeding just the manually marked rectangles to the original classifier gives a separate diagnostic. It detected 3 of the 5 likely-positive patches and called 2 of the 20 clear or other-cloud patches positive. On the corresponding full photos it detected 2 of 5, again with 2 false positives. Five positives are too few to select an architecture. The strongest automatic crop was centered inside the marked example in two of the five photos. The other three centers can still fall on valid cloud outside that example.

The highest-value next training data are real, building-dominant views with a human-verified genus in a specific sky region. Record an explicit uncertain label for distant cloud, and include hard negatives such as altocumulus, stratus, glare and blue sky. Keep an uncropped reference alongside the rectangle so the reviewer can use apparent cloud size and surrounding structure. Separate capture sessions and contributors before making crops or synthetic variants. With more independent positive scenes, train a patch head on those regions and fit its aggregation on a separate development set. Evaluate against the frozen photo model on new real scenes, stratified by sky coverage.
