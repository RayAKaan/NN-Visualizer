from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.history import router as history_router
from api.meta import router as meta_router
from api.models import router as models_router
from api.predict import router as predict_router
from api.state import router as state_router
from api.train_ws import router as train_ws_router
from api.weights import router as weights_router
from config import CORS_ORIGINS
from services.inference import inference_engine

app = FastAPI(title="NN Visualizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(meta_router)
app.include_router(predict_router)
app.include_router(state_router)
app.include_router(weights_router)
app.include_router(models_router)
app.include_router(history_router)
app.include_router(train_ws_router)


@app.on_event("startup")
def load_model_on_startup() -> None:
    inference_engine.load()
