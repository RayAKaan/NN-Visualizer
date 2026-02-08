- **Training Details:**
- Optimizer: Adam
- Loss: Sparse categorical cross-entropy
- Early stopping with validation monitoring

---

## System Architecture

### 1. TensorFlow Model
- Dense MNIST classifier trained locally
- Separate activation model exposes intermediate layer outputs
- Weights extracted for visualization and explainability

### 2. Backend API (FastAPI)
Provides a lightweight inference and inspection interface:

- **`POST /predict`**
- Returns:
  - Predicted digit
  - Class probabilities
  - Hidden layer activations
  - Explanation data (top contributing neurons)
- **`GET /weights`**
- Returns trained weight matrices for:
  - Hidden1 → Hidden2
  - Hidden2 → Output

All inference runs on the trained TensorFlow model loaded at startup.

### 3. Frontend (React + TypeScript)
- Canvas-based digit input
- SVG-based neural network visualization
- Pruned, activation-weighted connection rendering
- Smooth temporal interpolation of activations
- Optional 2D / 3D visualization modes

---

## Implemented Roadmap

1. **Model Training**
 - MNIST normalization and dense architecture
2. **Activation Extraction**
 - Dedicated activation model for intermediate layers
3. **Backend API**
 - Inference, weights, and explanation endpoints
4. **Frontend Visualization**
 - Drawing grid
 - Activation heatmaps
 - Output probability bars
 - Signed connection paths
 - Decision explanation panel
5. **Performance Optimizations**
 - Request throttling (~30 ms)
 - Activation smoothing
 - Connection pruning and global caps

---

## Backend Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate   # Windows
# or
py -m venv .venv
.\.venv\Scripts\Activate.ps1
# or
source .venv/bin/activate  # For Linux / macOS

pip install -r requirements.txt
python train.py
uvicorn app:app --reload --port 8000

---

## Frontend Setup
cd frontend
npm install
npm run dev
