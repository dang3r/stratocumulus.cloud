# Mixed-scene stratocumulus model

Open the app and select **Try the mixed-scene model for buildings and sky**. The building setting is selected initially. Check **Higher sensitivity for sky-dominant views** when pointing the camera at open sky. Both settings handle uploads and camera frames. Processing happens locally in the browser. The camera requires HTTPS or localhost. Run this branch from the repository root.

```sh
python -m http.server 8000
```

Open <http://localhost:8000/?model=cloud-gaze>. The first visit fetches the model files and ONNX Runtime Web. The default mode remains available through the model switch.

## Browser results

Each cell reports true positives / positives, then false alarms / negatives. The two mixed-scene settings share one encoder and ONNX head. The building setting uses a margin of 0.83, the largest hundredth that retained at least 85% of the nine positives in the 2019 calibration set. The optional higher-sensitivity setting uses the original learned margin of zero. The 2021 outcomes at zero had already been inspected when stricter calibration was considered.

| Evaluation set | Previous photo model | Building setting | Higher sensitivity |
| --- | --- | --- | --- |
| CLOUD GAZE building views, geographic cross-validation | 11/30, 34/122 | 23/30, 12/122 | 27/30, 32/122 |
| New CLOUD GAZE skyward photos | 5/34, 9/96 | 14/34, 3/96 | 30/34, 12/96 |
| GLOBE 2019 buildings, calibration | 7/9, 5/31 | 8/9, 3/31 | 8/9, 10/31 |
| GLOBE 2021 buildings | 7/9, 8/35 | 9/9, 7/35 | 9/9, 14/35 |
| GLOBE 2023 buildings, fully prospective | 12/13, 4/24 | 11/13, 2/24 | 12/13, 4/24 |
| Wikimedia Commons buildings | 0/13, 0/20 | 6/13, 7/20 | 10/13, 12/20 |
| Cloud Appreciation Society gallery | 11/39, 24/227 | 31/39, 82/227 | 37/39, 118/227 |
| Paired synthetic buildings | 0/20, 0/20 | 1/20, 2/20 | 11/20, 5/20 |
| ADE no-sky controls | —, 0/182 | —, 0/182 | —, 0/182 |

On the geographically held-out building photos, higher sensitivity achieved 90.0% recall (95% Wilson interval 74.4–96.5%) and 26.2% false positives (19.2–34.7%). Two of the five folds exceeded 30% false positives. Only one of the 130 independent skyward photos showed a prominent building. In the annual checks, 43 of 83 candidate building photos from 2019 and 42 of 86 from 2021 were visually indeterminate. In 2023, 28 of 65 reviewed scenes were indeterminate. Across the two years outside threshold fitting (2021 and 2023), the building setting caught 20/22 (90.9%) and flagged 9/59 negatives (15.3%). The 2023 sample alone was fully prospective and yielded 84.6% recall and 8.3% false positives. The higher-sensitivity setting met both targets on that sample. Each threshold has a different failure mode across collections. All year-specific rates exclude visually indeterminate scenes. Across the three years we screened 1,610 downloaded photographs. The browser scores for both settings use the same recorded inference margins.

The three model files total 28.55 MB, including the 3.52 MB mixed-scene head, plus ONNX Runtime Web. Chromium and WebKit exercised the app upload path, and a simulated courtyard camera stream exercised the live path. Geographic benchmark inference took a median of 0.85 seconds and a 95th percentile of 3.68 seconds in desktop Chromium with concurrent jobs and cold starts. Physical handset latency and genus accuracy on independently labeled video clips still need measurement.

## What changed

The existing sky segmenter weights the image patches in a quantized DINOv2-Small encoder. Each photo provides two views: the full frame and, when the mask supports one, a native-resolution sky window. A predominantly sky frame reuses its embedding. The general classifier compares these embeddings with learned examples. A second fitted linear classifier adjusts its score using the full and cropped features, their scores and the fraction of the image predicted to be sky. The final ONNX head contains a single quantized reference matrix shared by both views. The original photo encoder now exposes its pooled embedding alongside its existing scores.

The general classifier uses 4,544 training embeddings from the original training partitions, CLOUD GAZE, Wikimedia Commons and synthetic building views. It excludes the geographic cells in the urban adaptation benchmark from its CLOUD GAZE training data. Of the retained examples, 952 are real directional CLOUD GAZE photos. The urban data contain 152 building views from 98 quarter-degree geographic cells. Thirty images are positive according to CLOUD GAZE, covering 23 cells. Across five outer folds, an entire cell stays on one side of each split. Four inner folds select regularization and the operating threshold. The objective maximizes the smaller of the margins above 85% recall and 70% specificity. The final head uses all 152 urban photos after the cross-validation runs.

## Interpreting the tests

NASA CLOUD GAZE volunteers assessed each camera direction separately. The urban benchmark uses their image-specific stratocumulus flags and retains disagreement levels in the manifest. Geographic splitting matters because the same observation can contribute several directional photos. The release protocol also lets volunteers mark photos with more than 25% of the sky obstructed as “other.” Those rows have no usable cloud-genus label. A learned ADE scene segmenter and visual review located urban examples; additional contact-sheet review recovered examples the screen missed. Building visibility was reviewed separately from the cloud scores. Everyday street video can contain considerably less sky than these photos.

The 2019, 2021 and 2023 phone checks use a different label source. Each observer reported a single genus across the surrounding sky with scattered, broken or overcast coverage. Each sampled photo comes from a different contributor and geographic cell within its year. An AI reviewer identified prominent foreground buildings and rated visible stratocumulus, other cloud/clear sky, or indeterminate before opening predictions. These provisional visual labels have no independent expert adjudication. The tables report classifiable photos separately from the total screened. Sampling enriched positives and does not estimate their prevalence in everyday use.

The Commons comparison samples tagged photographs from country categories, with perceptual deduplication and manual removal of repeated scene series. The Cloud Appreciation Society supplies photo-specific gallery tags and photographer credits. Its gallery sample is enriched by cloud-type filters; it contains many sky-dominant photographs. Neither gallery records a consistent phone-camera protocol. Synthetic positives in the paired building test reuse five reviewed real cloud donors. The no-sky controls come from the earlier held-out ADE partition.

These data were obtained from the Zooniverse online platform, the GLOBE Program and NASA Langley Research Center. [CLOUD GAZE dataset](https://zenodo.org/records/7853217) · [NASA documentation](https://observer.globe.gov/en/get-data/clouds-data) · [Annual GLOBE observations](https://zenodo.org/records/18763550) · [Wikimedia Commons](https://commons.wikimedia.org/wiki/Category:Stratocumulus_clouds) · [Cloud Appreciation Society gallery](https://cloudappreciationsociety.org/gallery/) · [ADE20K](https://huggingface.co/datasets/zhoubolei/scene_parse_150)

## Reproduce the model and browser measurements

In the working directory from the [parent training recipe](../README.md), prepare its images, sky model and transformer features through `train_transformer.py`. Keep its `ade.parquet`, `dino.onnx`, `sky.onnx`, `dino_scores.csv` and `dino_features.npy`. Set `recipe` to the absolute path of this cloud_gaze directory. The new trainer also uses SciPy. The evaluation runner uses Playwright for Python.

```bash
for name in gaze-clean gaze-independent commons-train; do
    gzip -dc "$recipe/$name.csv.gz" > "$name.csv"
    python "$recipe/prepare_data.py" "$recipe/$name.csv.gz"
    python "$recipe/gaze_embed.py" "$name.csv"
done
python "$recipe/gaze_synthetic.py"
python "$recipe/gaze_embed.py" gaze-synthetic.csv
OPENBLAS_NUM_THREADS=3 python "$recipe/fit_basis.py"
OPENBLAS_NUM_THREADS=1 python "$recipe/fit_urban.py"
python "$recipe/export_urban.py" urban-head.npz urban-release.npz gaze-head.onnx
python "$recipe/summarize.py"
```

The reference export has SHA-256 `96c0ec10018a7c9d1d35223ea684879a24c4a46e840d4807d1e2a0a849f39b92`. For a new browser run, pass a CSV with local `file` paths and labels to `python "$recipe/run_browser.py" input.csv results.jsonl`. Use `--chromium /path/to/chromium` for an installed browser. For geographic cross-validation, export each `urban-fold-N.npz` using the same export command and supply its ONNX file via `--fold-head` for the matching held-out rows. Pass `--high-sensitivity` to evaluate that operating point. The archived scores include each image reference, baseline and candidate decisions, decision margin, elapsed time and fold when applicable. The annual sampling manifests include all downloaded candidates, including images screened out for framing. [Browser results](browser-results.jsonl.gz) · [Urban folds](urban-sites.csv.gz) · [2019 sample](annual-field.csv.gz) · [2021 sample](annual21-field.csv.gz) · [2023 sample](annual23-field.csv.gz)
