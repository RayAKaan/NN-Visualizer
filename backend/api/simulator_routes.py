from __future__ import annotations

from typing import List, Optional

import base64
import json

import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from simulator.dataset_engine import custom_dataset, generate_dataset
from simulator.datasets import (
    generate_sequence_dataset,
    generate_text_tokens,
    load_cifar10,
    load_fashion_mnist,
    load_mnist,
    process_image,
)
from simulator.dataset_manager import dataset_manager
from simulator.equation_engine import layer_equations
from simulator.forward_engine import run_forward_full, run_forward_step
from simulator.inspector import activation_inspection, weight_inspection
from simulator.backward_engine import backward_full, backward_step
from simulator.sequence_engine import sequence_step, sequence_full
from simulator.debugger import diagnose
from simulator.performance_estimator import estimate_performance
from simulator.snapshot_manager import snapshot_manager
from simulator.training_engine import TrainingConfig, training_sessions
from simulator.layers import LayerConfig, validate_layers
from simulator.session_manager import session_manager
from simulator.comparison.comparison_engine import setup_comparison
from simulator.comparison.comparison_metrics import compute_comparison_results
from simulator.profiler.flops_counter import compute_flops
from simulator.profiler.memory_estimator import estimate_memory
from simulator.profiler.throughput_estimator import estimate_throughput
from simulator.profiler.bottleneck_detector import detect_bottlenecks
from simulator.import_export.import_manager import import_manager
from simulator.import_export.architecture_serializer import parse_architecture, serialize_architecture
from simulator.import_export.code_generator import generate_pytorch, generate_keras
from simulator.import_export.image_exporter import export_svg
from simulator.templates.architecture_templates import list_templates

router = APIRouter(prefix="/api/simulator", tags=["simulator"])


class LayerIn(BaseModel):
    type: str
    neurons: int
    activation: Optional[str] = None
    init: Optional[str] = None
    input_shape: Optional[List[int]] = None
    kernel_size: Optional[int] = None
    stride: Optional[int] = None
    padding: Optional[str] = None
    filters: Optional[int] = None
    pool_size: Optional[int] = None
    pool_stride: Optional[int] = None
    hidden_size: Optional[int] = None
    sequence_length: Optional[int] = None
    return_sequences: Optional[bool] = None
    embedding_dim: Optional[int] = None
    vocab_size: Optional[int] = None
    num_heads: Optional[int] = None


class ArchitectureRequest(BaseModel):
    layers: List[LayerIn]


class ForwardRequest(BaseModel):
    graph_id: str
    input: List[float]


class ForwardStepRequest(ForwardRequest):
    step_index: int


class EquationRequest(BaseModel):
    graph_id: str
    layer_index: int
    include_numeric: bool = True


class DatasetRequest(BaseModel):
    type: str
    n_samples: int = Field(200, ge=10, le=5000)
    noise: float = Field(0.1, ge=0.0, le=1.0)
    train_split: float = Field(0.8, ge=0.5, le=0.95)
    seed: Optional[int] = None


class SequenceDatasetRequest(BaseModel):
    type: str
    n_samples: int = Field(200, ge=10, le=5000)
    seq_length: int = Field(20, ge=2, le=200)
    n_features: int = Field(1, ge=1, le=128)
    vocab_size: int = Field(50, ge=5, le=5000)
    n_classes: int = Field(3, ge=2, le=50)
    noise: float = Field(0.1, ge=0.0, le=1.0)
    train_split: float = Field(0.8, ge=0.5, le=0.95)


class CustomDatasetRequest(BaseModel):
    points: List[dict]
    train_split: float = Field(0.8, ge=0.5, le=0.95)


class StandardDatasetRequest(BaseModel):
    name: str
    n_samples: int = Field(500, ge=50, le=5000)
    train_split: float = Field(0.8, ge=0.5, le=0.95)
    seed: Optional[int] = None


class InspectWeightsRequest(BaseModel):
    graph_id: str
    layer_index: int


class InspectActivationsRequest(BaseModel):
    graph_id: str
    layer_index: int
    input: List[float]


class BackwardRequest(BaseModel):
    graph_id: str
    input: List[float]
    target: List[float]
    loss_function: str = "bce"
    l2_lambda: float = 0.0


class BackwardStepRequest(BackwardRequest):
    step_index: int


class GradientsRequest(BaseModel):
    graph_id: str
    layer_index: int


class GradientFlowRequest(BaseModel):
    graph_id: str


class DebugRequest(BaseModel):
    graph_id: str
    check_dead_neurons: bool = True
    check_gradients: bool = True
    check_overfitting: bool = True
    check_saturation: bool = True
    n_diagnostic_samples: int = 100


class ApplyFixRequest(BaseModel):
    graph_id: str
    fix_action: dict


class ReplayLoadRequest(BaseModel):
    graph_id: str
    snapshot_index: int


class PerformanceRequest(BaseModel):
    graph_id: str
    n_samples: int
    batch_size: int
    optimizer: str = "adam"


class FeatureMapRequest(BaseModel):
    graph_id: str
    input_image: List[float]
    input_shape: List[int]


class SaliencyRequest(BaseModel):
    graph_id: str
    input: List[float]
    input_shape: List[int]
    target_class: int = 0
    method: str = "gradient"


class FilterResponseRequest(BaseModel):
    graph_id: str
    dataset_id: str
    layer_index: int
    filter_index: int
    n_samples: int = 50


class NeuronAtlasRequest(BaseModel):
    graph_id: str
    dataset_id: str
    layer_index: int
    n_samples: int = 100


class SequenceStepRequest(BaseModel):
    graph_id: str
    sequence: List[List[float]]
    timestep: int


class SequenceFullRequest(BaseModel):
    graph_id: str
    sequence: List[List[float]]


class CompareSetupRequest(BaseModel):
    models: List[dict]
    dataset_id: str
    epochs: int = 50


class CompareResultsRequest(BaseModel):
    comparison_id: str


class ProfileRequest(BaseModel):
    graph_id: str
    batch_sizes: List[int] = [1, 8, 16]

class ImportBuildRequest(BaseModel):
    import_id: str


class ExportCodeRequest(BaseModel):
    graph_id: str
    format: str


class ExportImageRequest(BaseModel):
    graph_id: str
    format: str = "svg"
    width: int = 1200
    height: int = 600

def _parse_layers(req_layers: List[LayerIn]) -> List[LayerConfig]:
    return [
        LayerConfig(
            layer_type=l.type,
            neurons=l.neurons,
            activation=l.activation,
            init=l.init,
            input_shape=l.input_shape,
            kernel_size=l.kernel_size,
            stride=l.stride,
            padding=l.padding,
            filters=l.filters,
            pool_size=l.pool_size,
            pool_stride=l.pool_stride,
            hidden_size=l.hidden_size,
            sequence_length=l.sequence_length,
            return_sequences=l.return_sequences,
            embedding_dim=l.embedding_dim,
            vocab_size=l.vocab_size,
            num_heads=l.num_heads,
        )
        for l in req_layers
    ]


@router.post("/architecture/validate")
def architecture_validate(req: ArchitectureRequest):
    layers = _parse_layers(req.layers)
    result = validate_layers(layers)
    return {
        "valid": result.valid,
        "architecture": result.architecture,
        "activations": result.activations,
        "total_params": result.total_params,
        "flops_per_sample": result.flops_per_sample,
        "layer_params": result.layer_params,
        "errors": result.errors,
        "warnings": result.warnings,
    }


@router.post("/architecture/build")
def architecture_build(req: ArchitectureRequest):
    layers = _parse_layers(req.layers)
    result = validate_layers(layers)
    if not result.valid:
        raise HTTPException(status_code=400, detail=result.errors)
    graph_id = session_manager.create_graph(layers)
    graph = session_manager.get_graph(graph_id)
    weight_stats = []
    for idx, w in enumerate(graph.weights):
        weight_stats.append(
            {
                "layer": idx,
                "mean": float(w.mean()),
                "std": float(w.std()),
                "min": float(w.min()),
                "max": float(w.max()),
            }
        )
    return {
        "graph_id": graph_id,
        "weights": [w.tolist() for w in graph.weights],
        "biases": [b.tolist() for b in graph.biases],
        "weight_stats": weight_stats,
    }


@router.post("/forward/full")
def forward_full(req: ForwardRequest):
    try:
        graph = session_manager.get_graph(req.graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    steps, final_output, layer_outputs = run_forward_full(graph, req.input)
    return {
        "steps": steps,
        "final_output": final_output,
        "total_steps": len(steps),
        "layer_outputs": {str(k): v for k, v in layer_outputs.items()},
    }


@router.post("/forward/step")
def forward_step(req: ForwardStepRequest):
    try:
        graph = session_manager.get_graph(req.graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        return run_forward_step(graph, req.input, req.step_index)
    except IndexError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/equations/layer")
def equations_layer(req: EquationRequest):
    try:
        graph = session_manager.get_graph(req.graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        return layer_equations(graph, req.layer_index, req.include_numeric)
    except IndexError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/dataset/generate")
def dataset_generate(req: DatasetRequest):
    try:
        data = generate_dataset(req.type, req.n_samples, req.noise, req.train_split, req.seed)
        dataset_manager.add(data["dataset_id"], data)
        return data
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/dataset/load_standard")
def dataset_load_standard(req: StandardDatasetRequest):
    name = (req.name or "").lower()
    if name == "mnist":
        data = load_mnist(req.n_samples, req.train_split, req.seed)
    elif name in {"fashion_mnist", "fashion-mnist"}:
        data = load_fashion_mnist(req.n_samples, req.train_split, req.seed)
    elif name in {"cifar10", "cifar-10"}:
        data = load_cifar10(req.n_samples, req.train_split, req.seed)
    else:
        raise HTTPException(status_code=400, detail="Unknown standard dataset")
    dataset_manager.add(data["dataset_id"], data)
    return data


@router.post("/dataset/custom")
def dataset_custom(req: CustomDatasetRequest):
    try:
        data = custom_dataset(req.points, req.train_split)
        dataset_manager.add(data["dataset_id"], data)
        return data
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/dataset/generate_sequence")
def dataset_generate_sequence(req: SequenceDatasetRequest):
    dtype = (req.type or "sine_wave").lower()
    if dtype == "text_tokens":
        data = generate_text_tokens(req.n_samples, req.seq_length, req.vocab_size, req.n_classes, req.train_split)
    else:
        data = generate_sequence_dataset(
            dtype,
            req.n_samples,
            req.seq_length,
            req.n_features,
            req.n_classes,
            req.noise,
            req.train_split,
        )
    dataset_manager.add(data["dataset_id"], data)
    return data


@router.post("/dataset/upload_image")
async def dataset_upload_image(file: UploadFile = File(...), target_size: int = 28):
    data = await file.read()
    return process_image(data, target_size)


@router.post("/inspect/weights")
def inspect_weights(req: InspectWeightsRequest):
    try:
        graph = session_manager.get_graph(req.graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        return weight_inspection(graph, req.layer_index)
    except IndexError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/inspect/activations")
def inspect_activations(req: InspectActivationsRequest):
    try:
        graph = session_manager.get_graph(req.graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        return activation_inspection(graph, req.layer_index, req.input)
    except IndexError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/backward/full")
def backward_full_route(req: BackwardRequest):
    try:
        graph = session_manager.get_graph(req.graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    result = backward_full(graph, req.input, req.target, req.loss_function, req.l2_lambda)
    session = training_sessions.get_or_create(req.graph_id, graph)
    session.last_gradients = {
        "dW": [np.asarray(g) for g in result["gradients_W"]],
        "db": [np.asarray(g) for g in result["gradients_b"]],
    }
    return result


@router.post("/backward/step")
def backward_step_route(req: BackwardStepRequest):
    try:
        graph = session_manager.get_graph(req.graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        result = backward_step(graph, req.input, req.target, req.loss_function, req.step_index)
        if "partial_gradients" in result:
            session = training_sessions.get_or_create(req.graph_id, graph)
            if result["partial_gradients"]:
                dW = []
                db = []
                for i in range(len(graph.weights)):
                    entry = result["partial_gradients"].get(str(i))
                    if entry:
                        dW.append(np.asarray(entry["dW"]))
                        db.append(np.asarray(entry["db"]))
                if dW:
                    session.last_gradients = {"dW": dW, "db": db}
        return result
    except IndexError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/inspect/gradients")
def inspect_gradients(req: GradientsRequest):
    try:
        session = training_sessions.get(req.graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not session.last_gradients["dW"]:
        raise HTTPException(status_code=400, detail="No gradients available. Run training or backward pass.")
    dW = session.last_gradients["dW"][req.layer_index]
    db = session.last_gradients["db"][req.layer_index]
    vals = dW.flatten()
    counts, edges = np.histogram(vals, bins=12)
    return {
        "layer_index": req.layer_index,
        "dW_matrix": dW.tolist(),
        "db_vector": db.tolist(),
        "dW_shape": [int(dW.shape[0]), int(dW.shape[1])],
        "stats": {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "l2_norm": float(np.linalg.norm(vals)),
        },
        "histogram": {"bins": [float(e) for e in edges.tolist()], "counts": [int(c) for c in counts.tolist()]},
    }


@router.post("/inspect/gradient_flow")
def inspect_gradient_flow(req: GradientFlowRequest):
    graph_id = req.graph_id
    try:
        session = training_sessions.get(graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not session.last_gradients["dW"]:
        raise HTTPException(status_code=400, detail="No gradients available.")
    norms = [float(np.linalg.norm(dw)) for dw in session.last_gradients["dW"]]
    per_layer = [{"layer": i, "gradient_norm": n, "status": "healthy"} for i, n in enumerate(norms)]
    return {
        "per_layer": per_layer,
        "total_gradient_norm": float(np.linalg.norm(norms)),
        "flow_health": "healthy",
        "flow_ratio": norms[-1] / max(norms[0], 1e-8) if norms else 0.0,
    }


@router.get("/inspect/weight_history/{graph_id}")
def inspect_weight_history(graph_id: str):
    try:
        session = training_sessions.get(graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    epochs = [h["epoch"] for h in session.weight_history]
    per_layer = []
    if session.weight_history:
        layers = len(session.weight_history[0]["weight_norms"])
        for li in range(layers):
            per_layer.append(
                {
                    "layer": li,
                    "norms": [h["weight_norms"][li] for h in session.weight_history],
                    "means": [],
                    "stds": [],
                    "deltas": [h["weight_deltas"][li] for h in session.weight_history],
                }
            )
    return {"epochs": epochs, "per_layer": per_layer}


@router.post("/debug/diagnose")
def debug_diagnose(req: DebugRequest):
    try:
        session = training_sessions.get(req.graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    grads = session.last_gradients["dW"]
    gradient_norms = [float(np.linalg.norm(dw)) for dw in grads] if grads else []
    loss_history = [m.train_loss for m in session.history]
    train_loss = session.history[-1].train_loss if session.history else None
    test_loss = session.history[-1].test_loss if session.history else None
    dead_neurons = session.history[-1].dead_neurons if session.history else [0] * len(gradient_norms)
    total_neurons = [layer.neurons for layer in session.graph.layers[1:]]
    return diagnose(gradient_norms, loss_history, train_loss, test_loss, dead_neurons, total_neurons)


@router.post("/debug/apply_fix")
def debug_apply_fix(req: ApplyFixRequest):
    try:
        graph = session_manager.get_graph(req.graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    fix = req.fix_action or {}
    if fix.get("type") == "change_activation":
        idx = int(fix.get("layer_index", -1))
        if idx < 0 or idx >= len(graph.layers) - 1:
            raise HTTPException(status_code=400, detail="Invalid layer_index")
        graph.layers[idx + 1].activation = fix.get("new_activation", "relu")
        summary = " -> ".join(str(l.neurons) for l in graph.layers)
        return {
            "success": True,
            "message": f"Layer {idx + 1} activation changed to {graph.layers[idx + 1].activation}",
            "architecture_updated": True,
            "new_arch_summary": summary,
        }
    raise HTTPException(status_code=400, detail="Unsupported fix action")


@router.get("/replay/snapshots/{graph_id}")
def replay_snapshots(graph_id: str):
    snaps = snapshot_manager.list_snapshots(graph_id)
    summaries = []
    for idx, snap in enumerate(snaps):
        summaries.append(
            {
                "index": idx,
                "epoch": snap.epoch,
                "train_loss": snap.metrics.get("train_loss"),
                "test_loss": snap.metrics.get("test_loss"),
                "train_accuracy": snap.metrics.get("train_accuracy"),
                "test_accuracy": snap.metrics.get("test_accuracy"),
                "thumbnail_boundary": snap.boundary,
            }
        )
    return {"total_snapshots": len(snaps), "epochs": [s.epoch for s in snaps], "summaries": summaries}


@router.post("/replay/load")
def replay_load(req: ReplayLoadRequest):
    try:
        graph = session_manager.get_graph(req.graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        snap = snapshot_manager.get_snapshot(req.graph_id, req.snapshot_index)
    except IndexError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    snapshot_manager.apply_snapshot(graph, snap)
    return {
        "epoch": snap.epoch,
        "weights": [w.tolist() for w in snap.weights],
        "biases": [b.tolist() for b in snap.biases],
        "metrics": snap.metrics,
        "boundary_grid": None,
        "layer_activations_sample": None,
    }


@router.post("/performance/estimate")
def performance_estimate(req: PerformanceRequest):
    try:
        graph = session_manager.get_graph(req.graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return estimate_performance(graph, req.n_samples, req.batch_size, req.optimizer)


@router.post("/activations/feature_maps")
def activations_feature_maps(req: FeatureMapRequest):
    try:
        graph = session_manager.get_graph(req.graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    from simulator.visualization.feature_maps import compute_feature_maps

    return compute_feature_maps(graph, req.input_image, req.input_shape)


@router.post("/activations/saliency")
def activations_saliency(req: SaliencyRequest):
    try:
        graph = session_manager.get_graph(req.graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    method = (req.method or "gradient").lower()
    if method == "grad_cam":
        from simulator.visualization.grad_cam import compute_grad_cam

        return compute_grad_cam(graph, req.input, req.input_shape, req.target_class)
    from simulator.visualization.saliency import compute_saliency

    return compute_saliency(graph, req.input, req.input_shape, req.target_class)


@router.post("/activations/filter_response")
def activations_filter_response(req: FilterResponseRequest):
    try:
        graph = session_manager.get_graph(req.graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    from simulator.visualization.filter_response import compute_filter_response

    return compute_filter_response(graph, req.dataset_id, req.layer_index, req.filter_index, req.n_samples)


@router.post("/activations/neuron_atlas")
def activations_neuron_atlas(req: NeuronAtlasRequest):
    try:
        graph = session_manager.get_graph(req.graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    from simulator.visualization.neuron_atlas import compute_neuron_atlas

    return compute_neuron_atlas(graph, req.dataset_id, req.layer_index, req.n_samples)


@router.post("/sequence/step")
def sequence_step_route(req: SequenceStepRequest):
    try:
        graph = session_manager.get_graph(req.graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        return sequence_step(graph, req.sequence, req.timestep)
    except (ValueError, IndexError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/sequence/full")
def sequence_full_route(req: SequenceFullRequest):
    try:
        graph = session_manager.get_graph(req.graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        return sequence_full(graph, req.sequence)
    except (ValueError, IndexError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/compare/setup")
def compare_setup(req: CompareSetupRequest):
    return setup_comparison(req.models, req.dataset_id, req.epochs)


@router.get("/compare/results/{comparison_id}")
def compare_results(comparison_id: str):
    return compute_comparison_results(comparison_id)


@router.post("/profile/full")
def profile_full(req: ProfileRequest):
    try:
        graph = session_manager.get_graph(req.graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    flops = compute_flops(graph)
    mem = estimate_memory(graph)
    batch = estimate_throughput(req.batch_sizes)
    bottleneck = detect_bottlenecks()
    return {
        "summary": {
            "total_params": sum(w.size + b.size for w, b in zip(graph.weights, graph.biases)),
            "total_flops_forward": flops.get("total_flops_forward", 0),
        },
        "per_layer": flops.get("per_layer", []),
        "batch_analysis": batch,
        "bottleneck": bottleneck,
        "memory": mem,
    }







@router.get("/templates/list")
def templates_list():
    return {"templates": list_templates()}


@router.post("/import/upload")
async def import_upload(file: UploadFile = File(...), format: str | None = None):
    data = await file.read()
    filename = file.filename or ""
    ext = filename.split(".")[-1].lower()
    detected = format or ext
    warnings: list[str] = []
    if ext == "json" or detected == "json":
        try:
            payload = json.loads(data.decode("utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc
        artifact = import_manager.add("json", payload, warnings)
        return {
            "import_id": artifact.import_id,
            "status": "success",
            "format_detected": artifact.format,
            "architecture": artifact.architecture,
            "warnings": warnings,
        }
    raise HTTPException(status_code=400, detail="Unsupported import format. Use JSON architecture for now.")


@router.post("/import/build")
def import_build(req: ImportBuildRequest):
    try:
        artifact = import_manager.get(req.import_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    layers = parse_architecture(artifact.architecture)
    graph_id = session_manager.create_graph(layers)
    return {"graph_id": graph_id, "ready": True}


@router.post("/export/code")
def export_code(req: ExportCodeRequest):
    try:
        graph = session_manager.get_graph(req.graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    fmt = (req.format or "json").lower()
    if fmt == "pytorch":
        return {"format": "pytorch", "code": generate_pytorch(graph.layers), "filename": "custom_model.py"}
    if fmt == "keras":
        return {"format": "keras", "code": generate_keras(graph.layers), "filename": "custom_model.py"}
    payload = serialize_architecture("custom", graph.layers)
    return {"format": "json", "code": json.dumps(payload, indent=2), "filename": "architecture.json"}


@router.post("/export/image")
def export_image(req: ExportImageRequest):
    try:
        graph = session_manager.get_graph(req.graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    export = export_svg(graph.layers, req.width, req.height)
    svg = export["image_base64"]
    encoded = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return {"image_base64": f"data:image/svg+xml;base64,{encoded}", "filename": "network_graph.svg"}

