# Microscopy (Geti/Greenstone) — 2027.0 Situation & Interim Plan

_Last updated: 2026-11-16 · Owner: Sakshi Jha_

## Plain-English primer (read this first)

**What the product is.** A biologist types a plain question about a microscope
image ("count and classify the cells in this tile"). A **planner agent** (an AI
orchestrator) reads the request, picks the right **agent skills** (small,
self-contained tools), chains them into a pipeline, runs it **locally on a Xeon
CPU**, and returns a results table + a marked-up image. No cloud, no GPU required.

**What the PPT asked.** Deliver that automated cellular analysis for **2027.0
(March)** as **agent skills + a planner DAG** running **Geti models**. Guidance:
use **RF-DETR** for detection (avoid YOLO26 for licensing), **OpenVINO or PyTorch**
are both fine, target **Xeon / Ubuntu 24.04**. Because Geti's models aren't ready,
we were told to **test RF-DETR in our own pipeline now** and swap Geti's models in
later behind the same interface.

**What the JIRA tickets say.** 27 tickets split into two ownership layers:
- **Geti layer (data + models, 96788–96793):** ingest OME-Zarr/OME-TIFF images,
  generate annotation datasets, train models at 16-bit. _Owned by Geti/Avinash._
- **Agent/skills layer (96794–96815, ours):** the pipeline demo (96794), the
  **skills registry** (segmentation 96796/96797, classification 96798, readers
  96803/96804, OpenCV 96802, DAG-as-skill 96799…), and the **planner agent**
  (96805–96815). _Note: there is **no standalone detection-skill ticket** — a gap
  to raise._

**Glossary (the technical terms, simply).**
| Term | Plain meaning |
|------|---------------|
| Agent skill | A small tool with a label (`SKILL.md`) + code. The planner calls it. |
| Planner agent | The AI that picks skills and chains them into a workflow. |
| DAG / pipeline | The ordered chain of steps: detect → segment → classify → report. |
| Detection | Find **boxes** around objects (where are the cells?). |
| Segmentation | Trace the exact **outline/mask** of each object (its precise shape). |
| Classification | Put a **label** on each object (what kind of cell is it?). |
| Bounding box | 4 numbers `[x1,y1,x2,y2]` = a rectangle in pixels. |
| Mask | A per-pixel yes/no map of "is this pixel part of the object?". |
| OME-Zarr | A scientific microscopy image format: huge, multi-channel, pyramidal. |
| 8-bit vs 16-bit | 8-bit = 0–255 brightness (for display/AI); 16-bit = fine scientific intensity. |
| I/O contract | The fixed shape of data in/out of a skill — the stable boundary. |
| ONNX / OpenVINO IR | Portable model formats; OpenVINO runs fast on Intel Xeon CPU. |

## What we commit for 2027.0 (March)
Automated cellular analysis delivered as **agent skills + a planner-agent DAG**,
running **Geti models**. Division of labor: **Geti owns the models and OME-Zarr
data support** (their December 2026.3 preview); **we own the agent skills and the
pipeline that chains them**.

## The pipeline (ITEP-96794)
`image -> detect (bounding boxes) -> segment each region (Mask-RCNN-SwinT) ->
classify (ResNet) -> marked-up image + results dataframe`
Target platform: **Xeon CPU, Ubuntu 24.04**.

## Interim approach (per the OME-Zarr sync + architect)
Geti's models are **not ready yet**. In the interim we run a model (**RF-DETR**)
in **our own framework/pipeline** to learn its input/output contract and build the
skills against it now. When Geti's official models land (~December), we **swap
them in behind the same skill interface**. This is **bootstrap/learning only** —
Geti remains the source of the shipped models.

## Model guidance
- **RF-DETR** recommended (Geti sizes nano–2XL).
- **YOLO26 avoided** (licensing).
- **OpenVINO or PyTorch** both acceptable (no hard perf requirement).

## Open items (NOT yet confirmed — we proceed with the interim test anyway)
1. **Detection model:** whether RF-DETR replaces the earlier D-FINE-M is **not
   confirmed**. Architect's direction: **just test RF-DETR in our pipeline for now
   and see if it makes sense.**
2. **Boundary:** whether Geti provides the full inference pipeline (which would
   shrink/drop our OME-Zarr/OME-TIFF reader skills 96803/96804) is **not
   confirmed**. Interim: **test RF-DETR with our pipeline.**
3. **Awaiting Friday:** raw multi-channel vs RGB input (Radwan's PoC).
   RGB = nothing blocks us; raw/16-bit = added preprocessing work.

## Dependencies & risks
- **External long pole is Geti:** first model ~WW50 + OME-Zarr ingestion, both due
  as the **December preview**. We cannot fully scale until that lands.
- **No real Greenstone data yet;** PoCs use public **S-BIAD800**. Greenstone is on
  **TIFF today, moving to OME-Zarr** — the format we will receive is unconfirmed.

## Commit view
We can commit the 2027.0 scope now, contingent on (a) Geti delivering models +
OME-Zarr in the December preview and (b) the RGB input answer. With those,
**March 2027.0 is achievable.** The interim RF-DETR work **starts immediately and
is not blocked by Geti.**

## Validated results (interim test, 2026-11-10)
Ran the full chain on both a sanity image and the **real S-BIAD800** reference set.

- **Reference data reality (open-item #3):** S-BIAD800 = *Imaging Mass Cytometry of
  human tonsil* in **OME-Zarr**, 5D `(t,c,z,y,x)` = `(1, 27, 1, 2700, 2702)`, **uint8,
  27 channels — NOT RGB.** `prep_sbiad800_tile.py` makes the channel→RGB mapping
  explicit (pick 3 channels → per-channel 1–99% stretch → 8-bit RGB tile).
- **End-to-end runs:** `prep → infer_rfdetr` works on the real tile (1351×1350 RGB).
- **Domain gap confirmed:** COCO-pretrained RF-DETR finds **0 real cells** (34 spurious
  matches only below 0.05). Expected — the interim weights are COCO. **Cell detection
  needs Geti's domain-trained weights (or fine-tuning on S-BIAD800).** Pipeline/I/O are
  proven; only the weights are the gap.
- **Deployment chain proven (PPT ask: OpenVINO/PyTorch, Xeon CPU):**
  PyTorch RF-DETR → **ONNX** (`output/rfdetr-base.onnx`) → **OpenVINO IR FP16**
  (`rfdetr-base_ov.xml/.bin`, 52 MB) → **compiles & runs on CPU** (OpenVINO 2026.4.1).

### I/O contract (two levels)
- **Skill-level (what the Agent Skill exposes):** in = 8-bit RGB `H×W×3`; out = list of
  `{bbox_xyxy_px, class_id, confidence}` (supervision `Detections`).
- **Model tensor-level (OpenVINO IR):** in `dets_input [1,3,560,560] f32` (NCHW, fixed
  560²); out `dets [1,300,4]` + `labels [1,300,91]` (300 queries, 91 COCO classes).

## Immediate next steps
1. [DONE] Run RF-DETR in our pipeline on a sample image and on real S-BIAD800 data.
2. [DONE] Capture the **I/O contract** (skill-level + OpenVINO tensor-level) and export
   ONNX + OpenVINO IR, verified on CPU.
3. [DONE] Build the detection **Agent Skill** (`SKILL.md` + script + test) against this
   contract; template it for Mask-RCNN-SwinT (96796) and ResNet (96798), chain the DAG.
4. [PENDING Geti] Swap in Geti's domain-trained weights (~Dec) behind the same interface.

## What we built — COMPLETED (2026-11-16)

Three pipeline nodes scaffolded as real agent skills, plus a one-command runner.
All live under `microscopy_poc/skills/`. **Every test passes.**

| Skill | Ticket | Interim model (swappable) | Output contract | Tests |
|-------|--------|---------------------------|-----------------|-------|
| `detection/` | (no ticket — gap) | RF-DETR (COCO) | `{bbox_xyxy_px, class_id, confidence}` | 3 ✓ |
| `segmentation/` | 96796 / 96797 | classical Otsu, COCO-RLE masks | `{bbox, class_id, area_px, mask_rle}` | 4 ✓ |
| `classification/` | 96798 | feature rules (measure + label) | `{bbox, label, class_id, score, features}` | 4 ✓ |

- **One swap point per skill:** `load_detector()` / `load_segmenter()` /
  `load_classifier()`. Replacing only that function swaps in the Geti model; every
  downstream step is untouched.
- **Composition works (ITEP-96811):** segmentation calls detection; classification
  calls segmentation. `run_pipeline.py` runs all three in one command and writes a
  results **dataframe** (`cells.csv`) + a labeled overlay image — exactly 96794's output.
- **Demo:** `cd skills && python run_pipeline.py --image ../sample.jpg`.

## What we found

- Pipeline, composition, and I/O contracts are **real and tested today**; the only
  missing piece is **domain-trained weights** (Geti's, ~Dec).
- On real S-BIAD800, COCO weights detect **0 cells** — this is expected and cleanly
  **isolates the one true dependency** (weights), not the plumbing.
- Masks use the **COCO-RLE** standard, so Geti's segmenter drops in with no format work.
- The **16-bit** requirement (96793) has a designed home: the classification step
  measures intensity under the mask — that is where 16-bit source is read in production.

## Conclusion

We have de-risked everything we own. The interim RF-DETR detection, the
segmentation and classification skills, their composition, the dataframe output,
and the OpenVINO/Xeon deploy path are **built and proven**. What remains is
gated on Geti's December preview (models + OME-Zarr). **March 2027.0 is
achievable** if Geti delivers in December and the raw-vs-RGB input question is
answered; our non-gated skills are already moving.
