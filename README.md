# Neurofluxion

A full-stack neural network visualization and training platform for MNIST, built with FastAPI and React.

Neurofluxion lets you **draw**, **predict**, **compare**, and **train** neural networks in real time — with live layer-by-layer activations, gradient flows, and architecture comparison across ANN, CNN, and RNN/LSTM models.

## Features

- **Draw-to-Predict** — Draw a digit on the canvas and see instant predictions with confidence and probability distributions.
- **Architecture Comparison** — Run ANN, CNN, and RNN on the same input side-by-side.
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

### Train Models (optional)

```bash
cd backend
python train_ann.py
python train_cnn.py
python train_rnn.py
```

## Repository Layout

```
NN-Visualizer/
├── backend/
│   ├── api/              # REST + WebSocket routes
│   ├── model/            # ANN/CNN/RNN model builders
│   ├── simulator/        # Custom NN simulator engine
│   ├── services/         # Inference + explanation logic
│   ├── training/         # Training manager + gradient engine
│   ├── app.py            # FastAPI entrypoint
│   └── config.py         # Paths, CORS, defaults
├── frontend/
│   ├── src/
│   │   ├── components/   # UI modules (simulator, lab, training, etc.)
│   │   ├── design-system/# Tokens, hooks, and reusable components
│   │   ├── store/        # Zustand state stores
│   │   ├── pages/        # Page-level composition
│   │   └── index.css     # Global styles + design system
│   └── package.json
├── LICENSE
└── README.md
```

## API Overview

| Endpoint | Description |
|---|---|
| `GET /health` | Backend health + loaded models |
| `POST /predict` | Prediction + explanation |
| `GET /model/info` | Architecture details |
| `GET /models/available` | Loaded model list |
| `POST /model/switch` | Set active model |
| `WS /train` | Live training stream |
| `WS /stream` | Topology + metrics stream |
| `POST /api/simulator/architecture/build` | Build a custom network |
| `POST /api/simulator/forward/full` | Run forward + backward pass |
| `GET /api/device/info` | GPU/CPU device detection |

## License

[MIT](LICENSE)
