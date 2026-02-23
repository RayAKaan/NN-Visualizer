# Neural Network Visualizer

Interactive system for exploring ANN, CNN, and RNN/LSTM models on MNIST with prediction, model comparison, and training telemetry.

## Features
- Real-time drawing + prediction
- 3 model support: ANN, CNN, RNN/LSTM
- 2D / 3D / Compare view modes (Phase 1 includes working placeholders for upcoming deep visualizations)
- Explanation panel per model type
- Training WebSocket pipeline with batch + epoch metrics
- Confusion matrix + per-class precision/recall/F1 APIs
- Export prediction snapshot as PNG
- Presentation and UI polish scaffolds ready for future phases

## Architecture
Frontend (React + TypeScript + Vite) → FastAPI backend → TensorFlow inference/training services.

## Model comparison
| Model | Input | Core Layers | Best For |
|---|---|---|---|
| ANN | 784 flat pixels | Dense 256→128→64→10 | Baseline interpretability |
| CNN | 28×28×1 | Conv32/Pool/Conv64/Pool/Dense128/10 | Spatial feature extraction |
| RNN | 28 timesteps × 28 features | LSTM128/Dense64/10 | Sequential row-wise reasoning |

## Quick start
### Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train_ann.py
python train_cnn.py
python train_rnn.py
uvicorn app:app --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## Keyboard shortcuts
| Key | Action |
|---|---|
| C | Clear canvas |
| 0-9 | Load sample digit |
| D/T/K | 2D / 3D / Compare |
| M | Cycle model |
| P | Open presentation |
| ? | Shortcut help |
| Esc | Close modal |

## API reference
| Method | Route | Purpose |
|---|---|---|
| GET | `/health` | Backend status + loaded models |
| POST | `/predict` | Inference + explanation |
| POST | `/state` | Visualization state |
| GET | `/weights` | Model weights |
| GET | `/model/info` | Architecture details |
| GET | `/models/available` | Loaded model list |
| POST | `/model/switch` | Set active model |
| GET | `/samples` | One MNIST sample per digit |
| GET | `/training/history` | Batch history |
| GET | `/training/epochs` | Epoch summaries |
| GET | `/training/status` | Training status |
| WS | `/train` | Training control + streaming updates |

## Project structure
- `backend/`: FastAPI API, TensorFlow models, inference engine, training manager
- `frontend/`: React app, hooks, layout/prediction components, future-phase stubs

## Technology stack
FastAPI, TensorFlow, NumPy, React, TypeScript, Vite, Three.js.
