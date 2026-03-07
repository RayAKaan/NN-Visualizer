import asyncio
import math
from typing import Dict, List, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Neurofluxion Stream Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _relu(x: float) -> float:
    return x if x > 0 else 0.0


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _build_snapshot(epoch: int) -> dict:
    input_ids = ["I_1", "I_2", "I_3"]
    hidden_ids = ["H1_1", "H1_2", "H1_3", "H1_4"]
    output_ids = ["O_1", "O_2"]

    # deterministic changing inputs so stream animates smoothly
    in_acts = {
        "I_1": 0.55 + 0.35 * math.sin(epoch * 0.12),
        "I_2": 0.50 + 0.30 * math.cos(epoch * 0.11),
        "I_3": 0.45 + 0.28 * math.sin(epoch * 0.09 + 1.4),
    }

    # synthetic target to drive gradients
    target = [1.0, 0.0] if (epoch // 12) % 2 == 0 else [0.0, 1.0]

    # base weights for 3->4 and 4->2
    w1_base: Dict[Tuple[str, str], float] = {
        ("I_1", "H1_1"): 0.42,
        ("I_1", "H1_2"): -0.18,
        ("I_1", "H1_3"): 0.31,
        ("I_1", "H1_4"): -0.11,
        ("I_2", "H1_1"): -0.27,
        ("I_2", "H1_2"): 0.51,
        ("I_2", "H1_3"): -0.08,
        ("I_2", "H1_4"): 0.22,
        ("I_3", "H1_1"): 0.16,
        ("I_3", "H1_2"): -0.36,
        ("I_3", "H1_3"): 0.44,
        ("I_3", "H1_4"): 0.29,
    }
    w2_base: Dict[Tuple[str, str], float] = {
        ("H1_1", "O_1"): 0.37,
        ("H1_2", "O_1"): -0.22,
        ("H1_3", "O_1"): 0.41,
        ("H1_4", "O_1"): -0.13,
        ("H1_1", "O_2"): -0.34,
        ("H1_2", "O_2"): 0.28,
        ("H1_3", "O_2"): -0.19,
        ("H1_4", "O_2"): 0.33,
    }
    b_hidden = {"H1_1": 0.03, "H1_2": -0.06, "H1_3": 0.02, "H1_4": 0.01}
    b_out = {"O_1": -0.02, "O_2": 0.04}

    # animate params as if they are being trained over time
    def wiggle(v: float, i: int) -> float:
        return v + 0.06 * math.sin(epoch * 0.08 + i * 0.7)

    w1 = {}
    for i, k in enumerate(w1_base.keys()):
        w1[k] = wiggle(w1_base[k], i)
    w2 = {}
    for i, k in enumerate(w2_base.keys()):
        w2[k] = wiggle(w2_base[k], i + 11)

    hidden_z = {}
    hidden_a = {}
    for h in hidden_ids:
        z = sum(in_acts[i] * w1[(i, h)] for i in input_ids) + b_hidden[h]
        hidden_z[h] = z
        hidden_a[h] = _relu(z)

    out_z = {}
    out_a = {}
    for o in output_ids:
        z = sum(hidden_a[h] * w2[(h, o)] for h in hidden_ids) + b_out[o]
        out_z[o] = z
        out_a[o] = _sigmoid(z)

    # simple gradients
    delta_out = {o: out_a[o] - target[idx] for idx, o in enumerate(output_ids)}
    delta_hidden = {}
    for h in hidden_ids:
        dz = sum(delta_out[o] * w2[(h, o)] for o in output_ids)
        delta_hidden[h] = dz * (1.0 if hidden_z[h] > 0 else 0.0)

    edges: List[dict] = []
    # input->hidden
    for i in input_ids:
        for h in hidden_ids:
            w = w1[(i, h)]
            g = in_acts[i] * delta_hidden[h]
            edges.append(
                {
                    "id": f"{i}_to_{h}",
                    "from": i,
                    "to": h,
                    "weight": float(w),
                    "gradient": float(g),
                    "contribution": float(w * in_acts[i]),
                }
            )
    # hidden->output
    for h in hidden_ids:
        for o in output_ids:
            w = w2[(h, o)]
            g = hidden_a[h] * delta_out[o]
            edges.append(
                {
                    "id": f"{h}_to_{o}",
                    "from": h,
                    "to": o,
                    "weight": float(w),
                    "gradient": float(g),
                    "contribution": float(w * hidden_a[h]),
                }
            )

    incoming_map = {nid: [] for nid in [*input_ids, *hidden_ids, *output_ids]}
    outgoing_map = {nid: [] for nid in [*input_ids, *hidden_ids, *output_ids]}
    for e in edges:
        outgoing_map[e["from"]].append(e)
        incoming_map[e["to"]].append(e)

    neurons: List[dict] = []
    for nid in input_ids:
        neurons.append(
            {
                "id": nid,
                "layerType": "input",
                "activation": float(in_acts[nid]),
                "bias": 0.0,
                "gradient": 0.0,
                "incomingEdges": incoming_map[nid],
                "outgoingEdges": outgoing_map[nid],
            }
        )
    for nid in hidden_ids:
        neurons.append(
            {
                "id": nid,
                "layerType": "dense",
                "activation": float(hidden_a[nid]),
                "bias": float(b_hidden[nid]),
                "gradient": float(delta_hidden[nid]),
                "incomingEdges": incoming_map[nid],
                "outgoingEdges": outgoing_map[nid],
            }
        )
    for nid in output_ids:
        neurons.append(
            {
                "id": nid,
                "layerType": "dense",
                "activation": float(out_a[nid]),
                "bias": float(b_out[nid]),
                "gradient": float(delta_out[nid]),
                "incomingEdges": incoming_map[nid],
                "outgoingEdges": outgoing_map[nid],
            }
        )

    # synthetic training metrics for dashboard
    loss = max(0.02, 1.25 * math.exp(-epoch * 0.045) + 0.04 * math.sin(epoch * 0.21))
    acc = min(0.995, 0.28 + 0.72 * (1 - math.exp(-epoch * 0.05)) + 0.02 * math.sin(epoch * 0.17))

    return {
        "epoch": epoch,
        "metrics": {
            "epoch": epoch,
            "loss": float(loss),
            "accuracy": float(max(0.0, min(1.0, acc))),
        },
        "topology": {
            "neurons": neurons,
            "edges": edges,
        },
    }


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    epoch = 0
    try:
        while True:
            payload = _build_snapshot(epoch)
            await ws.send_json(payload)
            epoch += 1
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return


@app.get("/health")
def health():
    return {"status": "ok", "service": "neurofluxion-stream"}
