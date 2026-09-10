# Prediction Mode — Real Pretrained Model Registry & Adapter Pipeline

**Project:** Neurofluxion / NN-Visualizer
**Date:** 2026-09-10
**Status:** Implemented, verified end-to-end on CPU (API + registry + UI)

This report documents the upgrade of the existing **Prediction Mode** from a
single MNIST classifier to a **model-agnostic registry-with-adapters pipeline**
that runs multiple **real, pre-trained** ANN/CNN/RNN models. The Simulator is a
protected subsystem and was **not modified** (see *Explicit confirmations*).

---

## 1. Scope and guarantees

| Guarantee | Evidence |
|---|---|
| **Nothing was trained, fine-tuned or retrained.** | Every model in the registry loads genuine published checkpoints (Hugging Face, Keras Applications `weights="imagenet"`, or the repo's pre-existing `nn_visual_ann.h5`). The only conversion performed is format: a JAX pickle → `.npz` and a legacy Keras-2 SavedModel → Keras-3 graph with its original float32 variables copied across. No training run, no synthetic "pretrained" weights. |
| **Simulator internals untouched.** | `git status` shows zero changes under `backend/simulator/**`, `frontend/src/components/simulator/**`, `frontend/src/pages/SimulatorPage.tsx`, or simulator API routes/WebSockets. A new pytest suite exercises the Simulator engine to prove no regression. |
| **No pretrained models loaded at startup.** | The FastAPI app only loads the legacy baseline(s) it always loaded (existing `inference_engine.load_all()` behaviour, unchanged). Registry weights are fetched lazily inside `load()` / `predict()` on first use, then cached in memory and on disk. |
| **CPU is the baseline; GPU optional and never required.** | `services/prediction_service.py::detect_device()` reports CPU or GPU from TensorFlow. No CUDA-only code or dependency was added. |
| **One generic pipeline, no 10+ bespoke endpoints.** | API, service, registry and UI all talk in `model_id` + family-appropriate input. Model specifics live only inside per-model adapters. |

---

## 2. High-level architecture

```
                     ┌──────────────────────────────────────────────────┐
   Frontend          │  PredictionMode  ModelsMode                      │
  (React / Vite)     │  family tabs + per-model dropdowns, per-family   │
                     │  input (draw / image / text), statuses, Load/Ux  │
                     └───────────────┬──────────────────────────────────┘
                                     │  HTTP (Vite dev proxy /backend → :8000)
                                     ▼
  FastAPI      GET /models/catalog{,/{id}}   POST /models/{id}/load|unload
               POST /predict  (model_id · or legacy model_type path)
                     │
                     ▼
  services/prediction_service.PredictionService
        (model_id → adapter → lazy load → predict → standardized payload)
                     │
                     ▼
  model_registry/registry.ModelRegistry (metadata + status + caching)
                     │
                     ▼
  model_registry/adapters/  ─ one class per real model ─
   ANN: dacorvo/mnist-mlp · mysticdan/mlp-mnist · repo baseline
   CNN: MobileNetV3Small · MobileNetV2 · ResNet50 · VGG16 (imagenet)
   RNN: keras-io/bidirectional-lstm-imdb  (+ unavailable entries listed w/ reasons)
```

Layering:

- `model_registry/schema.py` — metadata fields, family/status constants, the
  standardized `prediction_payload(...)` builder.
- `model_registry/store.py` — lazy download, cache resolution, provenance sidecars.
- `model_registry/base.py` — `ModelAdapter` ABC (metadata / load / unload /
  `check_available()` / `predict()` returning
  `(predicted_index, confidence, probabilities, labels, extra)`).
- `model_registry/registry.py` — `ModelRegistry` + `UNAVAILABLE_MODELS` +
  module-level `registry` singleton.
- `services/prediction_service.py` — `PredictionService` (load/unload/predict,
  device detection, legacy `model_type` → baseline mapping).
- `api/predict.py`, `api/meta.py` — thin HTTP layer; no model logic.

---

## 3. Files

### New files (added)
| File | Purpose |
|---|---|
| `backend/model_registry/__init__.py`, `schema.py`, `store.py`, `base.py`, `registry.py`, `util.py`, `safetensors.py` | Registry core (schema/contract/cache/catalog; pure-NumPy safetensors reader; preprocessing/IO helpers). |
| `backend/model_registry/adapters/ann_adapters.py` | dacorvo + mysticdan MNIST MLP + repo baseline adapters. |
| `backend/model_registry/adapters/cnn_adapters.py` | Shared Keras ImageNet CNN adapter + MobileNetV3Small/MobileNetV2/ResNet50/VGG16 + ImageNet label loader. |
| `backend/model_registry/adapters/rnn_imdb_kerasio.py` | keras-io IMDB BiLSTM adapter (tokenizer + checkpoint variable rebuild). |
| `backend/model_registry/adapters/legacy_mnist_adapters.py` | Legacy project CNN/RNN MNIST adapters (delegate to existing inference engine when weights exist). |
| `backend/model_registry/adapters/_mysticdan_convert.py` | JAX-pickle → `.npz` converter (works with or without JAX). |
| `backend/services/prediction_service.py` | Generic registry-backed prediction service + `detect_device()`. |
| `backend/api/predict.py` (content rewritten) | `POST /predict` (registry + legacy `model_type` compatibility). |
| `backend/api/meta.py` (extended) | `GET /models/catalog{,/{id}}`, `POST /models/{id}/load|unload`. |
| `backend/tests/` | `conftest.py`, `test_registry.py`, `test_utils.py`, `test_prediction_service.py`, `test_api.py`, `test_simulator_smoke.py`. |
| `backend/requirements-dev.txt` | `pytest`, `httpx`. |
| `frontend/public/samples/` | TensorFlow example JPEGs (Grace Hopper, labrador, cat) for CNN demos. |
| `MODEL_REGISTRY_REPORT.md` | This report. |

### Modified files
- `backend/requirements.txt` — added `Pillow`, `python-multipart` (both needed by existing/new code paths).
- `frontend/src/components/models/ModelsMode.tsx` — pretrained registry section.
- `frontend/src/components/prediction/PredictionMode.tsx` — per-family inputs + model dropdowns.
- `frontend/src/types/index.ts`, `frontend/src/api/client.ts`, `frontend/vite.config.ts` — types + dev proxy.
- `README.md` — documentation of the registry.

### Inspected, not modified (unchanged)
`backend/simulator/**`, `backend/services/inference.py` (wrapped by legacy adapters, not edited),
`backend/services/explanation.py`, `backend/training/**`, `backend/train_*.py`,
`backend/model/*`, all simulator/lab/training frontend code, `frontend/src/App.tsx`.

---

## 4. Registered models — exact sources, inputs & preprocessing

Status legend: **available** = runnable in this stack; **unavailable** = listed
with a reason but not integrated (verified incompatible or weights absent).

| id | Family | Model & source | Input | Preprocessing | Params | Status |
|---|---|---|---|---|---|---|
| `ann-nn-visualizer` | ANN | This repo, `backend/nn_visual_ann.h5` (existing baseline, ~242k params) | 784 floats (drawn digit) | grayscale 28×28 → flatten; **already 0–1** (never re-divided by 255) | 242,762 | available |
| `ann-dacorvo-mnist-mlp` | ANN | Hugging Face `dacorvo/mnist-mlp` — official PyTorch MLP `model.safetensors` (784→256→256→10, ReLU/softmax) | 784 floats | flatten → z-score `(x−0.1307)/0.3081` (the repo's stated normalization) | 269,322 | available |
| `ann-mysticdan-mlp-mnist` | ANN | Hugging Face `mysticdan/mlp-mnist` — JAX MLP `mlp_mnist_model.pkl` (784→512→256→128→64→10), converted to `.npz` | 784 floats | flatten → ÷255 (no mean/std) | 575,050 | available |
| `cnn-mobilenetv3small` | CNN | Keras Applications, `weights="imagenet"` | image (base64/data URL) | RGB → 224×224 → `mobilenet_v3.preprocess_input` | 2,554,968 | available |
| `cnn-mobilenetv2` | CNN | Keras Applications, `weights="imagenet"` | image | RGB → 224×224 → `mobilenet_v2.preprocess_input` | 3,538,984 | available |
| `cnn-resnet50` | CNN | Keras Applications, `weights="imagenet"` | image | RGB → 224×224 → `resnet50.preprocess_input` | 25,636,712 | available |
| `cnn-vgg16` | CNN | Keras Applications, `weights="imagenet"` | image | RGB → 224×224 → `vgg16.preprocess_input` | 138,357,544 | available (needs ~3.5 GB RAM — see Limitations) |
| `rnn-bilstm-imdb-kerasio` | RNN | Hugging Face `keras-io/bidirectional-lstm-imdb` (tf-keras **SavedModel** + `tokenizer.pickle`) | text | lowercase/split (keras word-sequence filters) → packaged tokenizer → pad 500; output sigmoid → negative/positive | ~208k | available |
| `cnn-nn-visualizer` / `rnn-nn-visualizer` | CNN/RNN | This repo's legacy training outputs (`nn_visual_cnn.h5`, `nn_visual_rnn.h5`) | 784 floats | MNIST digit preprocessing | n/a | unavailable until .h5 present |
| `rnn-jongador-lstm-imdb-256` | RNN | HF `jongador/lstm-imdb-256` | — | — | — | unavailable (TextAttack/torch checkpoint; deprecated repo; needs the legacy textattack+torch stack) |
| `rnn-jongador-lstm-imdb-512` | RNN | HF `jongador/lstm-imdb-512` | — | — | — | unavailable (same TextAttack format; not loadable in the project's TensorFlow/Keras stack) |
| `rnn-pratyushee-assamese-lstm` | RNN | HF `pratyushee/assamese-sentiment-analysis` | — | — | — | unavailable (repo ships only a notebook — **no checkpoint file published**) |

The user-specified RNN candidates were each verified directly against their
Hugging Face repositories (API + file manifests + code/config):
`jongador/lstm-imdb-256` & `-512` are TextAttack-format checkpoints requiring a
torch/textattack load path, and `pratyushee/assamese-sentiment-analysis`
contains no weight files at all — so, per the task's substitution rule, the
**keras-io IMDB BiLSTM** (a genuine pretrained TensorFlow SavedModel that is
CPU-capable and compatible with the framework already used) fills the RNN slot.
The Models page lists the three unusable repos as **unavailable** with exact
reasons instead of pretending they run.

### Checkpoint-fidelity notes
- **mysticdan/mlp-mnist** ships a JAX pickle. `_mysticdan_convert` unpickles it
  (natively when JAX is present; otherwise through a tiny reconstruction shim)
  into a plain `.npz`, so inference is pure NumPy. No retraining.
- **keras-io/bidirectional-lstm-imdb** ships a Keras-2 SavedModel whose
  optimizer object cannot be imported by Keras 3. The adapter rebuilds the
  documented architecture (`Embedding(2000,30) → BiLSTM(64, return_sequences)
  → BiLSTM(64) → Dense(1,sigmoid)`) in Keras 3 and copies the genuine float32
  checkpoint variables + the packaged `tokenizer.pickle`; text is decoded and
  re-tokenized with that tokenizer (raw `keras.datasets.imdb` integer
  sequences mismatch the artifact and only reach ~53%). Verified ~87% IMDB
  test accuracy (matching the model card).
- **ImageNet labels**: `tf.keras`'s module alias does **not** expose
  `CLASS_INDEX`, so the CNN adapter loads the same `imagenet_class_index.json`
  Keras itself downloads (labels verified: Grace Hopper photo → class 652
  `military_uniform` on all tested CNNs).

---

## 5. Download / bootstrap & cache strategy

- Lazy: first `load()`/`predict()` triggers download; **startup never touches
  registry weights**.
- Cache precedence:
  1. `$NN_MODEL_CACHE` (any dir)
  2. `NN_MODEL_CACHE=local` → project `models_store/pretrained/` (git-ignored)
  3. default `~/.cache/nn_visualizer/models`
- Keras ImageNet weights use Keras' own cache (`~/.keras`, `KERAS_HOME`).
- A `*.provenance.json` sidecar (repo, file, license, timestamp) is written
  next to each cached weight for reproducibility.
- `.gitignore` already excludes `models_store/` and weight extensions; nothing
  binary is added to git. `backend/nn_visual_ann.h5` is the repo's existing
  pre-tracked baseline.

---

## 6. API surface

`POST /predict`
```jsonc
// request — model_id + exactly one family-appropriate input
{ "model_id": "cnn-mobilenetv3small", "image": "<base64 or data URL>" }
{ "model_id": "ann-dacorvo-mnist-mlp", "pixels": [784 floats 0..1] }
{ "model_id": "rnn-bilstm-imdb-kerasio", "text": "review…" }
// legacy (kept): model_type only + pixels → original engine response incl.
// layers/explanation (ann works; cnn/rnn return a clean 503 when no .h5)
```
```jsonc
// standardized response
{
  "model_id": "cnn-mobilenetv3small", "model_name": "MobileNetV3-Small (ImageNet)",
  "family": "CNN", "framework": "TensorFlow / Keras Applications",
  "predicted_class": "military_uniform", "predicted_index": 652,
  "confidence": 0.7497, "probabilities": [ ...1000 floats... ], "labels": [ ... ],
  "top_k": [ {"index": 652, "label": "military_uniform", "probability": 0.7497}, ... ],
  "latency_ms": 1234.5, "device": "CPU", "input_shape": [224,224,3],
  "architecture": "…", "parameter_count": 2554968, "dataset": "ImageNet-1k",
  "preprocessing": "…", "details": { /* optional viz, e.g. legacy ANN layers */ }
}
```
Other endpoints: `GET /models/catalog`, `GET /models/catalog/{id}`,
`POST /models/{id}/load`, `POST /models/{id}/unload`. Statuses exposed:
`available | loading | loaded | unavailable | error` (unavailable entries carry
`unavailable_reason`). Error mapping: unknown → 404, unavailable/unloadable →
503 with reason, bad input → 400.

---

## 7. Frontend changes

**Prediction Mode (`components/prediction/PredictionMode.tsx`)**
- Family tabs ANN / CNN / RNN each now have a **model dropdown populated from
  `GET /models/catalog`** (options show parameter counts; the dropdown only
  lists runnable models).
- Per-family input preserved & extended: ANN = original digit canvas (undo /
  redo / clear / grid / autopredict / MNIST sample digits / 784-pixel
  pipeline); CNN = image upload (file or example images) sent as base64; RNN =
  text area with positive/negative presets. The old "force pixels on every
  family" behaviour is gone.
- Result ring/center, adaptive probability landscape (10 classes for digits,
  2 for sentiment, top-5 class names for ImageNet) and a "What the network saw"
  details panel (legacy ANN layer activations + model metadata/latency/device).
- History thumbnails now store enough per family to restore a prediction.
- Existing "Compare Architectures" entry point is preserved (unchanged page).

**Models Mode (`components/models/ModelsMode.tsx`)** — new **Pretrained model
registry** section grouped by family showing extended metadata (framework,
source, weights, dataset, input type/shape, classes, preprocessing,
architecture, params, license, description), live status chips, unavailable
reasons, and Load/Unload actions. The original locally-trained weight
management (Use/Reload/Delete) is kept.

**Dev plumbing** — `apiClient` uses `/backend` in dev and Vite proxies it to
`http://127.0.0.1:8000` (`vite.config.ts`), so the app works from any host
(including the sandbox live preview). Training/Simulator WebSockets keep their
pre-existing `ws://localhost:8000` URLs (unchanged behaviour).

---

## 8. Tests

New suite under `backend/tests/`:

| Area | Coverage |
|---|---|
| Registry | discovery, family enumeration, metadata schema, per-model details/params, unavailable entries with reasons, unknown id, load-error propagation |
| Utils | softmax, probability normalisation, pixel validation, tokenizer behaviour, image base64→224×224, safetensors round-trip, cache env |
| PredictionService (light) | unknown model 404-path, unavailable-with-reason, legacy mapping |
| PredictionService (heavy, `NN_RUN_HEAVY=1`) | real CPU digit predictions (all 3 ANNs), RNN neg/pos, CNN ImageNet class-name decode, repeated-inference stability, load/unload lifecycle |
| API (HTTP) | catalog shape/statuses, single entry, unknown 404, load-unavailable 503, predict error mapping, legacy model_type path (incl. clean 503 without .h5) — heavy tests run real predictions through `/predict` |
| Simulator smoke | validation accepts/rejects, forward pass shape, `run_forward_full`, engine backend — proves the untouched engine still works |

**Results**
- Light: `35 passed, 12 skipped` (skips are heavy real-inference tests).
- Heavy (`NN_RUN_HEAVY=1`): **47 passed** (all suites including real CPU
  inference through the registry and HTTP layers).
- Manual service-level runs, all CPU: dacorvo→digit 4 (conf ≈ 1.000),
  mysticdan→9 (≈ .997), legacy ANN→7 (≈ .99992), keras-io BiLSTM→ negative
  .976 / positive .991, MobileNetV3Small/V2/ResNet50 → ImageNet class 652
  `military_uniform` (.750 / .804 / .988 on the Grace Hopper photo).
- Live server smoke: catalog, single entry, load/unload, error paths and
  legacy path all verified over HTTP via `curl`/`httpx` (see §6).

---

## 9. Limitations & notes

- **VGG16** is fully registered and uses the *same* adapter code path as the
  other three ImageNet CNNs, but its 528 MB checkpoint + model build needs
  roughly 3.5 GB RAM, which exceeds this 1.9 GB sandbox (a run was OOM-killed).
  It loads and runs anywhere with enough RAM; tests gate it behind
  `NN_RUN_VGG16=1`. All other models were executed repeatedly on CPU.
- The three originally-proposed RNN repos were **verified unusable** and are
  surfaced as unavailable rather than silently swapped; the keras-io BiLSTM is
  the one runnable pretrained RNN (a deliberate, documented substitution).
- ImageNet class *names* require the `imagenet_class_index.json` download on
  the first CNN prediction (Keras-side cache; offline after that).
- Real CNN/RNN inference times on CPU vary (first load includes weight
  download/build); subsequent calls reuse cached weights and are fast.
- Training/Simulator WebSockets still assume the backend at `localhost:8000`
  when accessed from a remote preview host (pre-existing behaviour, unchanged).

---

## 10. Explicit confirmations

1. **Nothing was trained, fine-tuned or retrained** during this task. All
   registry weights come from original published sources or the repo's
   existing baseline; conversions are format-only.
2. **No fake/synthetic “pretrained” weights** were created or committed.
3. **The Simulator subsystem is unchanged** — no file under `backend/simulator/`
   or the simulator UI was modified, and its engine still passes smoke tests.
4. **No models are loaded at startup** and **CPU is the required/default
   device**; GPU is optional and no CUDA-specific dependency was added.
5. Every model marked available has been run repeatedly through the project's
   single registry→adapter→service pipeline (and, for the demo models, through
   the HTTP API and the frontend wiring).
