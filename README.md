# Neurofluxion (NN-Visualizer)

Neurofluxion is a full-stack neural-network visualization platform for MNIST that combines:

- live handwritten digit inference,
- ANN/CNN/RNN architecture comparison,
- layer/activation explainability views,
- real-time training telemetry over WebSockets.

It is built for both demo-first usage (quick prediction and comparison) and deeper inspection (training + internals).

---

## Table of Contents

1. [What You Get](#what-you-get)
2. [Architecture](#architecture)
3. [Tech Stack](#tech-stack)
4. [Repository Layout](#repository-layout)
5. [Prerequisites](#prerequisites)
6. [Run Locally (Bash)](#run-locally-bash)
7. [Run Locally (PowerShell)](#run-locally-powershell)
8. [How to Use the App](#how-to-use-the-app)
9. [API and WebSocket Reference](#api-and-websocket-reference)
10. [Troubleshooting](#troubleshooting)
11. [Notes for Development](#notes-for-development)

---

## What You Get

- Three model families:
  - ANN (dense)
  - CNN (convolutional)
  - RNN/LSTM
- Prediction workspace:
  - draw-to-predict canvas
  - confidence + probability visualization
  - architecture comparison mode
- Training workspace:
  - start/pause/resume/stop training
  - live batch + epoch metrics
  - confusion matrix and class-level precision/recall/F1
- Model management:
  - inspect registry
  - switch active model
  - reload/delete/save model files

---

## Architecture

Frontend (React + TypeScript + Vite) talks to a FastAPI backend.

- HTTP APIs are used for:
  - prediction,
  - model metadata/state,
  - model lifecycle actions,
  - historical training snapshots.
- WebSockets are used for:
  - live training stream (`/train`),
  - live topology/metrics stream (`/stream`).

Backend startup loads available saved models from disk (`*.h5` paths configured in `backend/config.py`).

---

## Tech Stack

- Backend: FastAPI, TensorFlow, NumPy, Uvicorn
- Frontend: React 18, TypeScript, Zustand, Recharts, Tailwind, Vite
- Visualization helpers: `@react-three/fiber`, `three` (where applicable)

---

## Repository Layout

```text
NN-Visualizer/
|- backend/
|  |- api/                 # REST + WS routes
|  |- model/               # ANN/CNN/RNN model builders
|  |- services/            # inference + explanation logic
|  |- training/            # training manager + gradient utilities
|  |- app.py               # FastAPI app entrypoint
|  |- config.py            # model paths, CORS, defaults
|  |- train_ann.py         # standalone ANN training script
|  |- train_cnn.py         # standalone CNN training script
|  `- train_rnn.py         # standalone RNN training script
|- frontend/
|  |- src/
|  |  |- components/       # UI modules
|  |  |- hooks/            # socket/data hooks
|  |  |- store/            # Zustand stores
|  |  |- pages/            # page-level composition
|  |  `- App.tsx
|  `- package.json
`- README.md
```

---

## Prerequisites

- Python 3.10+ (recommended)
- Node.js 18+ and npm
- Enough disk/RAM for TensorFlow + MNIST training

Optional but recommended:
- GPU-ready TensorFlow environment if you want faster training

---

## Run Locally (Bash)

### 1) Backend setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Train models (first run or when you need fresh checkpoints)

```bash
python train_ann.py
python train_cnn.py
python train_rnn.py
```

This generates model files used by inference at startup.

### 3) Start backend

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 4) Frontend setup and run

```bash
cd ../frontend
npm install
npm run dev
```

Open: `http://localhost:5173`

---

## Run Locally (PowerShell)

### 1) Backend setup

```powershell
Set-Location backend
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If script execution is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
. .\.venv\Scripts\Activate.ps1
```

### 2) Train models (first run or refresh)

```powershell
python .\train_ann.py
python .\train_cnn.py
python .\train_rnn.py
```

### 3) Start backend

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 4) Frontend setup and run

```powershell
Set-Location ..\frontend
npm install
npm run dev
```

Open: `http://localhost:5173`

---

## How to Use the App

### Prediction

- Draw a digit on the canvas.
- Auto/manual predict shows:
  - predicted class,
  - confidence,
  - class probability distribution.
- Use comparison mode to run ANN/CNN/RNN on the same input in parallel.

### Training

- Configure model and training parameters.
- Start training from UI.
- Observe:
  - live batch loss/accuracy/gradient norm,
  - epoch-level train/validation curves,
  - class metrics and confusion matrix.

### Models

- Inspect model registry (disk + memory state).
- Switch active model.
- Reload or delete existing model files.
- Save newly trained in-memory model checkpoints.

---

## API and WebSocket Reference

### Health and model metadata

- `GET /health` - backend health + loaded models
- `GET /model/info?type=ann|cnn|rnn` - architecture details
- `GET /models/available` - loaded model list + active model
- `GET /models/registry` - full registry (disk + loaded + active)

### Inference and visualization

- `POST /predict` - prediction + explanation
  - body: `{ "pixels": [784 floats], "model_type": "ann|cnn|rnn" }`
- `POST /state` - visualization state payload
- `GET /weights?model_type=...` - cached model weights
- `GET /samples` - one MNIST sample per digit

### Model lifecycle

- `POST /model/switch` - set active model
- `POST /models/{model_type}/reload` - reload model from disk
- `DELETE /models/{model_type}` - delete model file + unload
- `POST /models/{model_type}/save` - save latest trained in-memory model

### Training history

- `GET /training/history` - batch history
- `GET /training/epochs` - epoch summaries
- `GET /training/status` - current training state/config

### WebSockets

- `WS /train`
  - control commands: `configure`, `start`, `pause`, `resume`, `stop`, `status`
  - stream events: batch updates, epoch updates, completion/errors
- `WS /stream`
  - continuous topology + synthetic metrics snapshots for live visualization mode

---

## Troubleshooting

### Backend starts but no models are available

- Train models first (`train_ann.py`, `train_cnn.py`, `train_rnn.py`).
- Verify files exist at paths defined in `backend/config.py`.

### Frontend can't reach backend

- Confirm backend is running on `http://localhost:8000`.
- Check `frontend/src/api/client.ts` base URLs.
- Confirm CORS origins in `backend/config.py`.

### Prediction fails after switching model

- Ensure selected model is loaded (`/models/available`).
- If needed, call reload endpoint or retrain/save.

### Training socket disconnects

- Ensure backend process remains active during training.
- Check terminal logs for TensorFlow runtime errors.

### PowerShell activation blocked

- Use process-scoped policy bypass:
  - `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

---

## Notes for Development

- Frontend build:

```bash
cd frontend
npm run build
```

- The current frontend bundle is large due to visualization dependencies and math rendering packages; consider route-level code splitting for optimization.
- Keep backend and frontend contracts synchronized when adding new activation/explanation payload fields.

---

## License

No license file is currently declared in this repository. Add one if distribution/public usage is intended.


