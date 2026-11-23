# Browser results

Each result cell gives true positives / false positives. Positive counts in the second column give the recall denominator. A request to reframe counts as a missed detection on a positive example. Both browser candidates and the shipped model were scored on the same files in Chromium 156.

| Panel | Positives / total | Original | Photo | Live |
|---|---:|---:|---:|---:|
| ADE | 0 / 182 | 0 / 0 | 0 / 0 | 0 / 0 |
| CCSN | 53 / 472 | 1 / 27 | 36 / 46 | 45 / 156 |
| GLOBE | 26 / 119 | 2 / 3 | 11 / 17 | 14 / 18 |
| GLOBE mixed | 49 / 95 | 0 / 1 | 22 / 11 | 22 / 14 |
| Howard | 24 / 242 | 1 / 8 | 9 / 21 | 12 / 42 |
| New cloudy phone photos | 53 / 122 | 9 / 5 | 27 / 12 | 29 / 18 |
| WMO | 35 / 160 | 3 / 4 | 8 / 8 | 12 / 26 |
| Synthetic sky behind a façade | 24 / 242 | 0 / 2 | 1 / 3 | 2 / 28 |
| Blurred and compressed GLOBE photos | 26 / 119 | 4 / 5 | 16 / 31 | 19 / 41 |
| Courtyard video frames | 0 / 80 | 0 / 0 | 0 / 0 | 0 / 0 |
| GLOBE with a visible building | 9 / 40 | 0 / 1 | 6 / 10 | 4 / 6 |

The new cloudy phone panel was opened after both decision rules were frozen. It contains 122 directional GLOBE Observer photos from eight additional observation dates. Eligible observations reported at least one cloud genus, and the sampling deliberately enriched stratocumulus. Contributors and sites were kept separate from the training data and earlier phone panels. A perceptual hash filter removed further near matches. The observation tags can miss what is present in a particular camera direction.

The real-building subset is a conservative visual selection within the 119-photo GLOBE panel. Building visibility was marked from contact sheets before reviewing scores for the slice. It overlaps the parent panel. Both models were evaluated independently frame by frame. The live interface also requires repeated estimates as described in the training notes.

The photo model called 2 of 37 mountain time-lapse frames possible stratocumulus. The clip-level description includes several cloud types, so this is a temporal observation.
The live model called 8 of 37 mountain time-lapse frames possible stratocumulus. The clip-level description includes several cloud types, so this is a temporal observation.

The per-image table includes source references and both abstention flags. The building flag identifies the manually marked single-genus GLOBE slice; other panels were not annotated for building presence.

[Source and training details](README.md) · [Per-image browser results](browser-evaluation.csv.gz)


## What the building tests show

A foreground building can coexist with a positive sky. The photo model detected 6 of 9 reported positives in the building slice, with 10 false positives among 31 negatives. In 36 of these views an analyst selected a sky rectangle without viewing the scores. On the same 36 images, using the whole photo produced 6 detections and 9 false positives. Cropping produced 3 detections and 4 false positives. Automatic cropping alone can discard cloud scale and context.

The constructed façade scenes are a harder endpoint. The photo model found 1 of 24 positive donors and live mode found 2 of 24. Building-only or indoor rejection is a different capability. Live mode requested reframing on all 182 held-out ADE no-sky images and every sampled frame of the single courtyard clip. These counts do not establish performance on an arbitrary building or camera feed.

## Browser checks

Both models ran in Chromium 156 and in Playwright WebKit 26.5 with a mobile-sized viewport. The full table uses Chromium. WebKit inference was smoke-tested on 60 examples per mode. Chromium also ran the upload UI and a prerecorded stream through the webcam interface, including a transition from courtyard to clouds and back, followed by stopping the camera. Physical iOS and Android hardware were unavailable.

Model downloads are approximately 25.0 MB for photos and 7.36 MB for video, before transfer compression. These sizes exclude the shared ONNX Runtime Web library. On this Linux host a separate warm run measured 30 predictions of the same phone image in each mode. Browser decoding, model download and startup time are excluded.

Warm median and 95th-percentile inference times were 396 ms and 448 ms for photos, and 29 ms and 42 ms for video. Browser runtime and device hardware affect both results.

A follow-up browser experiment evaluates up to three native-resolution sky crops per photo on additional GLOBE Observer images and matched building composites. [Region labels, measurements and reproduction](regions/README.md)
