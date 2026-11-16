# Microscopy 2027.0 — Meeting Talking Points (keep open during call)

**Me:** Sakshi · **With:** architect + manager · **Topic:** individual insights on microscopy requirements + my commitments & timeline

---

## 1. 30-second opener
> "Went through all 27 microscopy tickets. Clear split: **Geti owns ingestion + model training (96788–96793)**; **we own the planner agent (96805–96815) + the agent-skills registry (96796–96804)**, with **96794** as the reference DAG. I prototyped the interim detection path **end-to-end on the real S-BIAD800 OME-Zarr data** — RF-DETR in our pipeline, exported to OpenVINO, running on Xeon CPU. Pipeline and I/O contract hold; the only missing piece is Geti's domain weights. I have **3 decisions** for you and a set of **tickets I'm ready to own**."

## 2. The pipeline (say it in one breath)
read image → OpenCV preprocess → **detect boxes** → crop → **segment masks** → apply mask → **classify** → dataframe + marked-up image. (ITEP-96794, Xeon/Ubuntu 24.04, S-BIAD800 commercial-OK.)

## 3. Three decisions I need from the architect
1. **Detection model:** ticket 96794 says **D-FINE-M**; guidance says **RF-DETR**. I validated RF-DETR end-to-end. Your call — and can I open a **dedicated detection-skill ticket** (none exists today)?
2. **16-bit vs 8-bit:** Geti trains 16-bit (96793). Detect/segment can run 8-bit RGB; **which steps must stay 16-bit** (intensity/classify)?
3. **Monolith vs composed:** March default = single **Geti-DAG skill (96799)** or **agent-composed individual skills (96811)**?

## 4. Tickets I propose to own (non-Geti-gated, already de-risked)
- **96803 / 96804** OME-Zarr + OME-TIFF reader skills
- **96802** OpenCV preprocess (grid-split + normalize)
- **Detection skill** (new) — RF-DETR interim, swap Geti/D-FINE later
- **96799** HCI DAG-execution skill
- **96814** benchmark skill (reuses my perf-tools/live-metrics work)

Geti keeps model training/ingestion (96788–96793) + the model internals behind 96796–96798; I consume their outputs behind a stable skill interface.

## 5. What I already proved (credibility)
- RF-DETR in **our** pipeline on real S-BIAD800 (27-ch IMC, `(1,27,1,2700,2702)` uint8).
- Pipeline + I/O **work**; COCO weights → **0 cells** = gap is purely Geti weights/fine-tune (dependency quantified).
- **PyTorch → ONNX → OpenVINO IR FP16 → runs on CPU/Xeon** (deploy path proven for every model skill).
- Froze the skill **I/O contract** `{bbox_xyxy_px, class_id, confidence}` + built a **detection Agent Skill** (SKILL.md + script + passing tests).

## 6. Timeline
- **Dec (2026.3 preview):** Geti model training + ingestion (Cartox, Moldev Micro4) + OME-Zarr preview — **hard dependency**.
- **March (2027.0):** automated cellular analysis via agentic workflows + skills — **our commit**.
- **June (2027.1):** morphology classification via more skills.
- **My line:** non-gated skills (readers, OpenCV, interim detection, planner scaffold, benchmarks) start **now**; first real end-to-end ~**January** after Geti's Dec model; **March achievable if Geti delivers Dec + the 16-bit/input answers close now.**

## 7. One-line answer bank (if challenged)
- *Detection found nothing?* → Interim COCO weights, not cells. Pipeline/I/O proven; isolates the one real dependency (Geti weights).
- *OpenVINO on Xeon proven?* → Yes, IR FP16 compiled + ran on CPU.
- *RF-DETR vs D-FINE risk?* → Low — both DETR-family, identical skill interface, weights-only swap.
- *Biggest risk?* → Geti long pole (models + ingestion = Dec preview). Everything we own that isn't gated is already moving.

## 8. Priority #2 — Radwan / Geti models (pivot after requirements)
- Awaiting raw-multichannel vs RGB answer — grounded now: S-BIAD800 is 27-ch, ties to 96793 (16-bit).
- Geti models land in the Dec preview; interim RF-DETR is the bridge.
- **Qwen2VL** = planner's local VLM (96808) — confirm intended build/footprint on Xeon.
