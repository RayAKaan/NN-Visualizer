# Neurofluxion

A full-stack neural-network visualization, simulation, prediction, training, and model-exploration platform built with FastAPI and React.

Neurofluxion lets you **draw**, **predict**, **inspect**, **simulate**, and **train** neural networks in real time — with live layer-by-layer activations, gradient flows, model metadata, and task-appropriate explanations.

## Features

- **Draw-to-Predict** — Draw a digit on the canvas and see instant predictions with confidence and probability distributions.
- **Real Pretrained Models** — Prediction Mode now runs genuine pretrained checkpoints (not simulations) through one shared registry: MNIST MLPs, ImageNet CNNs (MobileNetV3-Small, MobileNetV2, ResNet50, VGG16) and an IMDB sentiment BiLSTM. Weights are downloaded lazily on first use and cached — never bundled into the repo.
- **Per-family inputs** — ANN keeps the 28×28 digit canvas, CNN takes an uploaded image (224×224), RNN takes free text; every family tab has its own model dropdown populated from registry metadata.
- **Model Registry page** — the Models tab shows extended metadata (framework, source, architecture, parameters, preprocessing, license…) plus live status: available / loading / loaded / unavailable / error, with Load/Unload controls.
- **Architecture Comparison** — Explore the existing same-input legacy architecture path; this is not a cross-task benchmark of unrelated pretrained registry models.
- **Live Training** — Start, pause, resume, and stop training from the UI with real-time loss/accuracy/gradient telemetry over WebSockets.
- **Simulator Mode** — Build custom neural networks from scratch: add layers, set activation functions, choose optimizers, load datasets, run forward/backward passes, and inspect every weight and gradient.
- **3D Visualization** — Interactive 3D scatter plots and network topology views via Three.js.
- **Weight Inspector** — Drill into individual neuron connections with per-layer weight distributions.
- **Import / Export** — Generate PyTorch or Keras code from your custom architectures.
- **Math Equations** — Rendered inline with KaTeX for each layer's forward/backward math.

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python, TypeScript |
| Frontend | React 18, Vite, Tailwind CSS, Zustand |
| Backend | FastAPI, Uvicorn, WebSockets |
| AI/ML | TensorFlow / Keras, NumPy |
| 3D & Charts | Three.js, Recharts |
| Math | KaTeX |
| Version Control | Git & GitHub |

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+ and npm

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

For a production verification pass:

```bash
npm run lint
npm run build
```

The Vite development server proxies `/backend/*` requests and WebSockets to
FastAPI, so the browser does not need a hard-coded backend host.

### Train Models (optional)

> Training produces the *legacy* `nn_visual_*.h5` baselines used by the old
> prediction flow. None of the pretrained registry models are trained here —
> they are downloaded from their original sources.

```bash
cd backend
python train_ann.py
python train_cnn.py
python train_rnn.py
```

## Pretrained Model Registry (Prediction Mode)

Prediction Mode classifies real inputs (drawn digits, images, text) with real
**pretrained** models. Nothing is trained or fabricated; every checkpoint is
downloaded from its original source on first use and cached on disk.

### Supported models

| Registry id | Family | Model / source | Weights | Input | Preprocessing | Params |
|---|---|---|---|---|---|---|
| `ann-nn-visualizer` | ANN | This repo's existing `backend/nn_visual_ann.h5` baseline | bundled .h5 | 784 floats (drawn digit) | grayscale 28×28 → flatten (already 0–1) | ~243k |
| `ann-dacorvo-mnist-mlp` | ANN | `dacorvo/mnist-mlp` (Hugging Face) | `model.safetensors` | 784 floats | flatten → z-score (mean 0.1307, std 0.3081) | ~269k |
| `ann-mysticdan-mlp-mnist` | ANN | `mysticdan/mlp-mnist` (Hugging Face) | `mlp_mnist_model.pkl` (converted to `.npz`) | 784 floats | flatten → ÷255 (no mean/std) | ~575k |
| `cnn-mobilenetv3small` | CNN | Keras Applications, `weights="imagenet"` | Keras cache (lazy) | image (base64/upload) | resize 224×224 → `mobilenet_v3.preprocess_input` | 2.55M |
| `cnn-mobilenetv2` | CNN | Keras Applications, `weights="imagenet"` | Keras cache (lazy) | image | resize 224×224 → `mobilenet_v2.preprocess_input` | 3.54M |
| `cnn-resnet50` | CNN | Keras Applications, `weights="imagenet"` | Keras cache (lazy) | image | resize 224×224 → `resnet50.preprocess_input` | 25.6M |
| `cnn-vgg16` | CNN | Keras Applications, `weights="imagenet"` | Keras cache (lazy) | image | resize 224×224 → `vgg16.preprocess_input` | 138M |
| `rnn-bilstm-imdb-kerasio` | RNN | `keras-io/bidirectional-lstm-imdb` (Hugging Face, tf-keras SavedModel) | SavedModel variables + `tokenizer.pickle` | text | lowercase/split → packaged IMDB tokenizer → pad to 500 | ~208k |

**Listed but unavailable** (shown on the Models page with a reason, never
integrated): `rnn-jongador-lstm-imdb-256`, `rnn-jongador-lstm-imdb-512`
(TextAttack/torch-format checkpoints incompatible with the project's
TensorFlow/Keras stack) and `rnn-pratyushee-assamese-lstm` (the repository
publishes only a notebook — no weights).

### How downloads & caching work

- Weights are fetched **lazily** — on the first `load`/`predict` for that model,
  never at server startup.
- Cached in `~/.cache/nn_visualizer/models` by default. Set
  `NN_MODEL_CACHE=local` to cache inside the git-ignored `models_store/pretrained/`,
  or point `NN_MODEL_CACHE` at any directory.
- Keras ImageNet weights use Keras' own cache (`~/.keras`, `KERAS_HOME`).
- A small `*.provenance.json` records repo/file/license next to each cached
  weight file for reproducibility.
- `models_store/` and all weight files are git-ignored — the repo never commits
  large binaries.

### CPU / GPU

CPU is the required baseline and the default: model loading, preprocessing and
inference are CPU-first, and responses report the executing device. If
TensorFlow detects a GPU it is used automatically; no CUDA-specific code or
dependency was added, and a GPU is never required.

### Architecture

```
model_registry/            central catalog + contracts
├── schema.py              metadata fields, family/status constants, response payload
├── store.py               lazy download + cache + provenance helpers
├── base.py                ModelAdapter contract (load/unload/predict/check_available)
├── registry.py            ModelRegistry + unavailable-model list + catalog/status
├── adapters/              one class per real model (all load weights lazily)
└── util.py, safetensors.py
services/prediction_service.py   generic pipeline: model_id → adapter → predict → payload
api/predict.py             POST /predict (registry mode + legacy model_type mode)
api/meta.py                GET /models/catalog..., POST /models/{id}/load|unload
```

The UI never hard-codes model logic: the Prediction and Models pages consume
`GET /models/catalog` metadata and send only `model_id` + family-appropriate
input to `POST /predict`.

### Adding a model

1. Subclass `ModelAdapter` under `backend/model_registry/adapters/`.
2. Implement `_describe()` (metadata), `load()` (lazy weight load) and
   `predict(raw_input)` returning `(index, confidence, probabilities, labels, extra)`.
3. Override `check_available()` when weights live on disk.
4. Register the class in `ADAPTER_CLASSES` in `backend/model_registry/registry.py`.

### Testing

```bash
cd backend
python -m pytest tests -q                 # registry, metadata, status, API + simulator smoke
NN_RUN_HEAVY=1 python -m pytest tests -q  # + real CPU inference through /predict
```

## Repository Layout

```
NN-Visualizer/
├── backend/
│   ├── api/                 # REST + WebSocket routes (predict, meta, simulator, …)
│   ├── model/               # ANN/CNN/RNN model builders (training)
│   ├── model_registry/      # Pretrained catalog: schema, store, base, registry, adapters
│   ├── simulator/           # Custom NN simulator engine (unchanged, protected)
│   ├── services/            # Inference engine + prediction_service (registry pipeline)
│   ├── tests/               # pytest suite (registry/status/API/simulator smoke + heavy CPU)
│   ├── training/            # Training manager + gradient engine
│   ├── app.py               # FastAPI entrypoint
│   └── config.py            # Paths, CORS, defaults
├── frontend/
│   ├── public/samples/      # TF example images used by the CNN demo
│   └── src/
│       ├── components/      # UI modules (simulator, lab, training, models, prediction, …)
│       ├── design-system/   # Tokens, hooks, reusable components
│       ├── store/           # Zustand state stores
│       ├── pages/           # Page-level composition
│       └── index.css        # Global styles + design system
├── LICENSE
└── README.md
```

## API Overview

| Endpoint | Description |
|---|---|
| `GET /health` | Backend health + loaded models |
| `POST /predict` | Prediction + explanation (registry `model_id`, or legacy `model_type`) |
| `GET /models/catalog` | Pretrained registry: metadata + status for every model |
| `GET /models/catalog/{id}` | Single registry entry |
| `POST /models/{id}/load` | Lazy-load a registry model into memory |
| `POST /models/{id}/unload` | Release a loaded registry model |
| `GET /model/info` | Architecture details |
| `GET /models/available` | Loaded model list |
| `POST /model/switch` | Set active model |
| `GET /samples` | MNIST digit samples for the drawing UI |
| `WS /train` | Live training stream |
| `WS /stream` | Topology + metrics stream |
| `POST /api/simulator/architecture/build` | Build a custom network |
| `POST /api/simulator/forward/full` | Run forward + backward pass |
| `GET /api/device/info` | GPU/CPU device detection |

## License

[MIT](LICENSE)
