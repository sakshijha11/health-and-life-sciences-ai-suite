# Microscopy POC — agentic cellular-analysis skills

Interim, inference-only proof-of-concept for the HLS 2027.0 microscopy effort.
Three composable agent skills — **detection → segmentation → classification** —
plus a one-command end-to-end runner. The pipeline and I/O contracts are frozen;
each skill has a single `load_*()` swap point where Geti-trained weights drop in
later. Today it runs on COCO-pretrained RF-DETR to prove the plumbing.

> Read `MICROSCOPY_2027_SITUATION.md` for the why, the results, and the open items.

---

## 1. Prerequisites

- Linux, Python **3.11**
- ~1 GB free disk for the auto-downloaded model weights (see §3)
- No GPU required (CPU inference is fine for the POC)

## 2. Setup

```bash
cd microscopy-poc
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Model weights — nothing to download by hand

The detector uses **RF-DETR**. On the **first** run, `rfdetr` automatically
downloads the COCO-pretrained checkpoint (`rf-detr-base.pth`, ~356 MB) to:

```
~/.roboflow/models/
```

So the model is **not** committed to this repo (it exceeds GitHub's file-size
limit and is fetched automatically). Just run a command below and the first
invocation pulls it once; later runs reuse the cache.

> The COCO weights don't know what a cell is — on real microscopy data they
> detect **0 cells**, which is expected. This POC proves the pipeline; the
> cell-aware weights are the `load_*()` swap point for Geti.

## 4. Run it — smoke test (no dataset needed)

A tiny `sample.jpg` is bundled so you can verify the whole chain end-to-end:

```bash
# End-to-end: detection -> segmentation -> classification
python skills/run_pipeline.py --image sample.jpg
# -> writes pipeline_out/{pipeline_result.json, cells.csv, pipeline_overlay.png}
```

Run the skills individually:

```bash
# Detection
python skills/detection/detect_skill.py --image sample.jpg --out detections.json

# Segmentation (auto-runs detection if --detections not given)
python skills/segmentation/segment_skill.py --image sample.jpg \
    --overlay segmentation_overlay.png --out instances.json

# Classification (auto-runs the chain if --instances not given)
python skills/classification/classify_skill.py --image sample.jpg \
    --csv cells.csv --overlay classified_overlay.png --out classifications.json
```

## 5. Run the tests

```bash
# Each skill ships with fast fake-model tests + a real integration test
python skills/detection/test_detect_skill.py
python skills/segmentation/test_segment_skill.py
python skills/classification/test_classify_skill.py
# or, if you have pytest:  pytest skills/
```

## 6. Run on real microscopy data (S-BIAD800)

The public reference set is **S-BIAD800** — *Imaging Mass Cytometry of human
tonsil* (CC BY 4.0). It is **not** committed here (hundreds of MB of binary data);
download it directly from the EBI BioImage Archive:

```bash
# ~96 MB, 27-channel OME-Zarr (zipped)
wget -O "Tonsil 1.ome.zarr.zip" \
  "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/800/S-BIAD800/Files/idr0054/Tonsil%201.ome.zarr.zip"

mkdir -p sbiad800
unzip -q "Tonsil 1.ome.zarr.zip" -d sbiad800/
```

Study landing page: https://www.ebi.ac.uk/biostudies/studies/S-BIAD800

S-BIAD800 is 27-channel uint8 (not RGB). Project 3 channels to an 8-bit RGB
tile, then run the pipeline on it:

```bash
# 27 channels -> pick 3 -> per-channel 1-99% stretch -> RGB tile
python prep_sbiad800_tile.py --zarr "sbiad800/Tonsil 1.ome.zarr/0" \
    --channels 0 1 2 --level 1 --out tonsil1_rgb.png

python skills/run_pipeline.py --image tonsil1_rgb.png
```

Expect **0 detections** with the COCO baseline — that's the documented result and
the exact gap the Geti-trained weights fill.

## 7. Layout

```
microscopy-poc/
├── requirements.txt
├── sample.jpg                     # bundled smoke-test image
├── prep_sbiad800_tile.py          # OME-Zarr (27ch) -> RGB tile
├── infer_rfdetr.py                # standalone RF-DETR inference / export
├── rfdetr_io_contract*.json       # frozen I/O contracts
├── MICROSCOPY_2027_SITUATION.md   # context, results, open items
├── MEETING_TALKING_POINTS.md
└── skills/
    ├── run_pipeline.py            # one command: det -> seg -> cls
    ├── detection/                 # SKILL.md + detect_skill.py + tests
    ├── segmentation/              # SKILL.md + segment_skill.py + tests
    └── classification/            # SKILL.md + classify_skill.py + tests
```
