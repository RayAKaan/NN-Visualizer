from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config
from api.history import router as history_router
from api.lab import router as lab_router
from api.meta import router as meta_router
from api.predict import router as predict_router
from api.simulator_routes import router as simulator_router
from api.simulator_landscape import router as simulator_landscape_router
from api.simulator_embeddings import router as simulator_embeddings_router
from api.simulator_interpret import router as simulator_interpret_router
from api.simulator_adversarial import router as simulator_adversarial_router
from api.simulator_compress import router as simulator_compress_router
from api.simulator_generative import router as simulator_generative_router
from api.simulator_augmentation import router as simulator_augmentation_router
from api.simulator_experiments import router as simulator_experiments_router
from api.simulator_assistant import router as simulator_assistant_router
from api.simulator_train_ws import router as simulator_train_ws_router
from api.state import router as state_router
from api.stream_ws import router as stream_ws_router
from api.train_ws import router as train_ws_router
from api.weights import router as weights_router
from services.inference import inference_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    inference_engine.load_all()
    yield


app = FastAPI(title="Neurofluxion", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(predict_router)
app.include_router(state_router)
app.include_router(weights_router)
app.include_router(meta_router)
app.include_router(history_router)
app.include_router(lab_router)
app.include_router(simulator_router)
app.include_router(simulator_landscape_router)
app.include_router(simulator_embeddings_router)
app.include_router(simulator_interpret_router)
app.include_router(simulator_adversarial_router)
app.include_router(simulator_compress_router)
app.include_router(simulator_generative_router)
app.include_router(simulator_augmentation_router)
app.include_router(simulator_experiments_router)
app.include_router(simulator_assistant_router)
app.include_router(simulator_train_ws_router)
app.include_router(train_ws_router)
app.include_router(stream_ws_router)

