import React, {
  useEffect,
  useMemo,
  useRef,
  useState,
  useCallback,
} from "react";

import CanvasGrid from "./components/CanvasGrid";
import LayerView from "./components/LayerView";
import ConnectionView from "./components/ConnectionView";
import Network3D from "./components/Network3D";

import TrainingMode from "./components/training/TrainingMode";
import { useTrainingSocket, TrainingConfig } from "./hooks/useTrainingSocket";

/* =====================================================
   Config

const API_BASE =
  import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

const GRID_PIXELS = 28 * 28;
const BLANK_PIXELS = Array(GRID_PIXELS).fill(0);

const FRAME_INTERVAL = 33;
const HIDDEN_ALPHA = 0.35;
const OUTPUT_ALPHA = 0.18;

const CONFIDENCE_THRESHOLD = 0.75;
const STABLE_FRAMES = 6;

/* =====================================================
   Default Training Config

const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  learningRate: 0.001,
  batchSize: 128,
  epochs: 10,
  optimizer: "adam",
  weightDecay: 0.0001,
  activation: "relu",
  initializer: "glorot_uniform",
  dropout: 0,
};

/* =====================================================
   Helpers

const smooth = (
  prev: number[],
  next?: number[],
  alpha = 0.3
): number[] => {
  if (!Array.isArray(next) || prev.length === 0) return prev;
  const len = Math.min(prev.length, next.length);
  const out = prev.slice();
  for (let i = 0; i < len; i++) {
    out[i] = prev[i] + alpha * (next[i] - prev[i]);
  }
  return out;
};

const normalize = (layer: number[]): number[] => {
  const max = Math.max(...layer, 1e-6);
  return layer.map((v) => v / max);
};

/* =====================================================
   Types

type Explanation = {
  prediction: number;
  confidence: number;
  top_hidden2_neurons?: Array<{
    neuron: number;
    activation: number;
    contribution: number;
  }>;
};

/* =====================================================
   App

const App: React.FC = () => {
  /* ---------------- mode ---------------- */

  const [mode, setMode] =
    useState<"inference" | "training">("inference");

  const [viewMode, setViewMode] =
    useState<"2d" | "3d">("2d");

  /* ---------------- inference state ---------------- */

  const [pixels, setPixels] =
    useState<number[]>(BLANK_PIXELS);

  const [hidden1, setHidden1] =
    useState<number[]>(Array(128).fill(0));
  const [hidden2, setHidden2] =
    useState<number[]>(Array(64).fill(0));
  const [probs, setProbs] =
    useState<number[]>(Array(10).fill(0));

  const [stablePrediction, setStablePrediction] =
    useState<number | null>(null);
  const [isConfident, setIsConfident] =
    useState(false);

  const [explanation, setExplanation] =
    useState<Explanation | null>(null);
  const [error, setError] =
    useState<string | null>(null);

  const [w12, setW12] =
    useState<number[][] | null>(null);
  const [w2o, setW2o] =
    useState<number[][] | null>(null);

  /* ---------------- refs ---------------- */

  const rafRef = useRef<number | null>(null);
  const lastFrameRef = useRef(0);
  const pixelsRef = useRef(pixels);
  const lastPredictionRef = useRef<number | null>(null);
  const stableCountRef = useRef(0);

  /* ---------------- training ---------------- */

  const training = useTrainingSocket(DEFAULT_TRAINING_CONFIG);

  /* =====================================================
     Training command dispatcher
  ===================================================== */

  const handleTrainingCommand = (command: string) => {
    switch (command) {
      case "configure":
        training.configure();
        break;
      case "start":
        training.start();
        break;
      case "pause":
        training.pause();
        break;
      case "resume":
        training.resume();
        break;
      case "stop":
        training.stop();
        break;
      case "step_batch":
        training.stepBatch();
        break;
      case "step_epoch":
        training.stepEpoch();
        break;
      default:
        console.warn("Unknown training command:", command);
    }
  };

  /* =====================================================
     Sync refs
  ===================================================== */

  useEffect(() => {
    pixelsRef.current = pixels;
  }, [pixels]);

  /* =====================================================
     Load inference weights
  ===================================================== */

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/weights`);
        const data = await res.json();
        setW12(data.hidden1_hidden2);
        setW2o(data.hidden2_output);
      } catch {
        setError("Model weights unavailable");
      }
    })();
  }, []);

  /* =====================================================
     Inference loop
  ===================================================== */

  const infer = useCallback(
    async (time: number) => {
      if (mode !== "inference") return;

      if (time - lastFrameRef.current < FRAME_INTERVAL) {
        rafRef.current = requestAnimationFrame(infer);
        return;
      }

      lastFrameRef.current = time;

      try {
        const res = await fetch(`${API_BASE}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ pixels: pixelsRef.current }),
        });

        const data = await res.json();

        setHidden1((p) =>
          smooth(p, data.layers.hidden1, HIDDEN_ALPHA)
        );
        setHidden2((p) =>
          smooth(p, data.layers.hidden2, HIDDEN_ALPHA)
        );
        setProbs((p) =>
          smooth(p, data.probabilities, OUTPUT_ALPHA)
        );

        const pred = data.prediction;
        const conf = data.probabilities[pred];

        if (lastPredictionRef.current === pred) {
          stableCountRef.current++;
        } else {
          stableCountRef.current = 1;
          lastPredictionRef.current = pred;
        }

        setIsConfident(
          stableCountRef.current >= STABLE_FRAMES &&
            conf >= CONFIDENCE_THRESHOLD
        );
        setStablePrediction(pred);
        setExplanation(data.explanation);
      } catch {
        setError("Prediction failed");
      }

      rafRef.current = requestAnimationFrame(infer);
    },
    [mode]
  );

  useEffect(() => {
    rafRef.current = requestAnimationFrame(infer);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [infer]);

  /* =====================================================
     Normalized visuals
  ===================================================== */

  const nHidden1 = useMemo(() => normalize(hidden1), [hidden1]);
  const nHidden2 = useMemo(() => normalize(hidden2), [hidden2]);
  const nOutput = useMemo(() => normalize(probs), [probs]);

  const maxDigit = useMemo(
    () => probs.indexOf(Math.max(...probs)),
    [probs]
  );

  /* =====================================================
     Render
  ===================================================== */

  return (
    <div className="app landscape">
      <header className="hero">
        <h1>NN Visualizer</h1>
        <p>Inference & Training Introspection</p>

        <div className="mode-toggle">
          <button onClick={() => setMode("inference")}>
            Inference
          </button>
          <button onClick={() => setMode("training")}>
            Training
          </button>
        </div>
      </header>

      {mode === "training" ? (
        training.ready ? (
          <TrainingMode
            view={viewMode}
            onToggleView={() =>
              setViewMode((v) => (v === "2d" ? "3d" : "2d"))
            }
            state3D={training.state3D}
            hidden1={training.hidden1}
            hidden2={training.hidden2}
            output={training.output}
            weightsHidden1Hidden2={training.weightsHidden1Hidden2}
            weightsHidden2Output={training.weightsHidden2Output}
            lossHistory={training.lossHistory}
            accuracyHistory={training.accuracyHistory}
            valLossHistory={training.valLossHistory}
            valAccuracyHistory={training.valAccuracyHistory}
            controlState={training.controlState}
            onControlChange={training.updateControl}
            onCommand={handleTrainingCommand}
          />
        ) : (
          <div className="panel loading">
            <p>Training not started</p>
          </div>
        )
      ) : (
        <section className="visualizer landscape-layout">
          <aside className="panel panel-left">
            <CanvasGrid pixels={pixels} onChange={setPixels} />
            <button onClick={() => setPixels([...BLANK_PIXELS])}>
              Clear
            </button>
            <p>
              Prediction:{" "}
              {isConfident ? stablePrediction : "uncertain"}
            </p>
          </aside>

          <main className="panel panel-center">
            {viewMode === "2d" && w12 && w2o ? (
              <>
                <ConnectionView
                  hidden1={hidden1}
                  hidden2={hidden2}
                  output={probs}
                  weightsHidden1Hidden2={w12}
                  weightsHidden2Output={w2o}
                />
                <LayerView
                  title="Hidden Layer 1"
                  activations={nHidden1}
                  columns={32}
                />
                <LayerView
                  title="Hidden Layer 2"
                  activations={nHidden2}
                  columns={16}
                />
              </>
            ) : (
              <Network3D
                hidden1={nHidden1}
                hidden2={nHidden2}
                output={nOutput}
                weightsHidden1Hidden2={w12}
                weightsHidden2Output={w2o}
              />
            )}
          </main>

          <aside className="panel panel-right">
            <div className="output-row">
              {probs.map((_, i) => (
                <div
                  key={i}
                  className={i === maxDigit ? "active" : ""}
                >
                  {i}
                </div>
              ))}
            </div>
          </aside>
        </section>
      )}
    </div>
  );
};

export default App;
