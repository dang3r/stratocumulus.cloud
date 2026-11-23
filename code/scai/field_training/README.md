# Field evaluation of the browser cloud classifier

The deployment target is a ground observer pointing a camera toward the sky. A positive means the image provides evidence of stratocumulus. A building can appear in either class. A separate response asks the user to reframe when there is too little usable sky.

## Evaluation data

- CCSN v2 contains 2,543 square photographs. All 340 stratocumulus images are 400 × 400. This is a useful genus benchmark with limited framing diversity. [CCSN](https://www.kaggle.com/datasets/mmichelli/cirrus-cumulus-stratus-nimbus-ccsn-database)
- Howard Cloud-X has more varied framing, although only seven of its 132 stratocumulus images are portrait. Its published test split was prioritized during duplicate removal. A seeded sample from both archives was visually inspected. [Howard Cloud-X](https://www.kaggle.com/datasets/imbikramsaha/howard-cloudx)
- GLOBE Observer supplies actual directional phone photos, including roofs, trees, haze and predominantly clear views. Development and test photos were separated by observer and site. Each observation contributes one horizontal direction. The primary test uses a sole reported genus or clear-sky observation. It covers 27 countries, with 74 of 119 photos from the United States. A separate test selects multi-genus observations at other sites. Cloud reports describe the entire observation and sometimes disagree with what is visible in a particular direction. [GLOBE API](https://api.globe.gov/) · [GLOBE Observer Clouds](https://observer.globe.gov/do-globe-observer/clouds)
- The WMO International Cloud Atlas provides expert examples. Of 190 selected images, 30 had a thumbnail hash within five bits of an archive image. Twenty-nine matches were to Howard. Visual inspection confirmed repeated photographs, and one matched cumulus example was in Howard's stratocumulus folder. The remaining 160 images were reserved for evaluation. Cloud genera in complex scenes may be incompletely specified by gallery tags or captions. [WMO Cloud Atlas](https://cloudatlas.wmo.int/en/search-image-gallery.html)
- ADE20K supplies building and sky masks. No-sky examples require zero labeled sky pixels and less than 15% unlabeled area. Image-name hashes split its 2,000 validation images into experiment-specific training, development and test partitions. The reported ADE results refer to the held-out partition. [ADE20K Scene Parsing](https://huggingface.co/datasets/zhoubolei/scene_parse_150)
- For the façade stress test, each held-out Howard image contributes a top-region donor. It is resized and placed in the annotated sky of a held-out ADE20K scene with 12–55% sky and more than 15% building. Polygon boundaries and donor labels are approximate. Several donors reuse a façade. For the compression stress test, real GLOBE photos were reduced to 240 pixels on the long side, blurred and JPEG-compressed. Each perturbation reuses an original photo.
- Real video includes 80 one-second samples of a static camera facing a courtyard and 37 samples at two-second intervals from a mixed-cloud mountain time-lapse. Frame counts express behavior within each clip. The mountain clip has a sequence-level description and no frame-level reference labels. [Courtyard video](https://github.com/opencv/opencv/blob/4.x/samples/data/vtest.avi) · [Mountain time-lapse](https://cloudatlas.wmo.int/en/imgviewer-5664.txt)

The supplied split holds image-level near-duplicates apart using a 64-bit grayscale difference hash with a Hamming distance cutoff of five. The filter can remove genuinely different smooth skies and miss transformed copies. Photographer or event identities were unavailable for many archive images. Stratocumulus prevalence in these panels is deliberately enriched and differs from the prevalence in everyday camera use.

## Reproduce the candidates

Use a fresh working directory. Set `recipe` to the absolute path of this directory in the checkout. The scripts read input archives from the working directory. The split manifest contains the public image references, local destination names, binary labels (2 means no annotated sky), and partition for each retained example. Source JPEGs stay outside the repository.

The CPU experiment used Python 3.12, PyTorch 2.10.0, torchvision 0.25.0, Pillow 13.0.0, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0, PyArrow 26.0.1, ONNX 1.23.2 and ONNX Runtime 1.32.0. The browser uses ONNX Runtime Web 1.22.0. Installation of the research dependencies is independent of the static site.

```bash
curl -fL https://www.kaggle.com/api/v1/datasets/download/mmichelli/cirrus-cumulus-stratus-nimbus-ccsn-database -o mmichelli.zip
curl -fL https://www.kaggle.com/api/v1/datasets/download/imbikramsaha/howard-cloudx -o imbikramsaha.zip
curl -fL https://huggingface.co/datasets/zhoubolei/scene_parse_150/resolve/e660d866a1351c70bcf07925d2600b60bd6e3bc3/scene_parsing/validation/0000.parquet -o ade.parquet
curl -fL https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth -o pretrained.pth
curl -fL https://huggingface.co/Xenova/dinov2-small/resolve/c2bb04a51fab207c420665f1946016107bffc701/onnx/model_quantized.onnx -o dino.onnx
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3
python "$recipe/prepare_images.py"
python "$recipe/train_sky.py"
python "$recipe/train_transformer.py"
python "$recipe/export_clouds.py"
python "$recipe/train_sky.py" --detail
python "$recipe/train_mobile.py"
python "$recipe/prepare_facades.py"
python "$recipe/train_mobile.py" --obstructed
python "$recipe/package_models.py" exported-browser-models
```

The photo model combines the DINOv2 classification token with a sky-weighted average of spatial patch tokens. The video model combines MobileNetV3-Small embeddings from the full frame and a rectangular sky crop when available. Both classifiers have three outputs: other cloud or sky, stratocumulus, and background. A strong class score can override a missed sky mask. Outputs are thresholded model scores.

Both masks are trained on sampled ADE20K segmentation pixels. The deeper video mask adds 69 visually inspected upper-sky rectangles from training photos to cover cloud textures underrepresented in ADE20K. Held-out sampled-pixel sky IoU was 0.686 for the photo mask and 0.804 for the video mask. The validation objective weights GLOBE F1 twice as heavily as each archive's F1 and subtracts the false-positive rate on ADE no-sky images. The photo model scored 0.549 and the video model 0.439 on that objective.

The exported photo assets correspond to `sky.onnx`, `dino_clouds.onnx`, and the threshold and override values in `dino_head.json`. The video assets correspond to `sky_cloud.onnx`, `compact.onnx`, and both `compact_head.json` and `augmented_head.json`. Each mode fetches its own assets lazily. Consecutive camera estimates must agree before the interface displays a cloud classification. A request to reframe takes effect immediately. Reported image and video-frame metrics use the individual estimates.

The browser models derive from public ImageNet MobileNetV3-Small and the int8 conversion of DINOv2-Small. [torchvision MobileNetV3-Small](https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html) · [DINOv2-Small](https://huggingface.co/facebook/dinov2-small) · [ONNX base conversion](https://huggingface.co/Xenova/dinov2-small)

For live video, the app uses the obstruction-specific head when less than 15% of the 28 × 28 grid is predicted to be sky. Both heads reuse the same embeddings. The additional head was fitted on 880 synthetic training views plus the original training data. A further 180 synthetic validation views used separate donors and backgrounds. On the four-domain validation objective including these views, the general head achieved 0.376, the obstruction-specific head alone achieved 0.382 and switching at 35% sky achieved 0.420. That first switch produced 93 false alarms on 218 negative façade composites. The final development search capped the synthetic validation false-positive rate at 10%. The selected switch uses a 15% sky-fraction cutoff and a 0.475 positive threshold in the obstructed branch. It achieved 0.408 on the combined objective and 7.6% false alarms on the synthetic validation negatives. When this branch has weak evidence, the camera asks the user to reframe. The synthetic panel motivated this experiment after the first candidate was inspected, so its improvement is exploratory.

[Browser results, foreground tests, and limitations](BENCHMARK.md)
