# NN-Visualizer

Interactive neural network visualization for MNIST digits. Draw on a 28×28 grid, watch neuron activations light up, and see live output probabilities.

## Project roadmap (implemented)

1. **Model (TensorFlow)**
   - MNIST dataset normalization, flatten to 784.
   - Fully connected network: 784 → 128 → 64 → 10 (softmax).
2. **Activation extraction**
   - Activation model returns hidden layer activations per inference.
3. **Backend API (FastAPI)**
   - `/predict` returns prediction, probabilities, and hidden activations.
   - `/weights` exposes weights for visualization.
4. **Frontend (React + TypeScript + Canvas/SVG)**
   - 28×28 drawing grid and debounced inference.
   - Hidden layer activations as intensity blocks.
   - Output probabilities as bars.
   - SVG connection lines with positive/negative influence.
   - Decision explanation panel using real activations and weights.
   - Optional 3D visualization using react-three-fiber.
   - Training mode with live WebSocket controls, charts, and timeline replay scrubber.
5. **Performance**
   - Debounced requests (~30ms).
   - Normalized activations for display.

## Backend setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
uvicorn app:app --reload --port 8000
```

## Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Then open `http://localhost:5173`.

## Notes

- If the backend reports that the model file is missing, run `python train.py`.
- The `/weights` endpoint returns hidden-layer weights for connection visualization.


## Streaming

- Training WebSocket endpoint: `ws://localhost:8000/train`
- Prediction API endpoint: `POST /predict`
- Training status endpoint: `GET /training/status`
- Model manager endpoints: `GET /models`, `POST /models/save`, `POST /models/load`, `DELETE /models/{name}`
