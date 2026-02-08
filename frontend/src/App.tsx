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

/* =====================================================
   Config
===================================================== */

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
   Helpers (SAFE)
===================================================== */

const smooth = (
  prev: number[],
  next?: number[],
  alpha = 0.3
) => {
  if (!Array.isArray(next)) return prev;
  const len = Math.min(prev.length, next.length);
  const out = prev.slice();
  for (let i = 0; i < len; i++) {
    out[i] = prev[i] + alpha * (next[i] - prev[i]);
  }
  return out;
};

const normalize = (layer: number[]) => {
  const max = Math.max(...layer, 1e-6);
  return layer.map((v) => v / max);
};

/* =====================================================
   Types
===================================================== */

type Explanation = {
  prediction: number;
  confidence: number;
  margin: number;
  top_hidden2_neurons?: Array<{
    neuron: number;
    activation: number;
    weight: number;
    contribution: number;
  }>;
  uncertainty_notes?: string[];
};

/* =====================================================
   App
===================================================== */

const App: React.FC = () => {
  /* ---------------- view ---------------- */

  const [viewMode, setViewMode] =
    useState<"2d" | "3d">("2d");

  /* ---------------- input ---------------- */

  const [pixels, setPixels] =
    useState<number[]>(BLANK_PIXELS);

  /* ---------------- activations (RAW) ---------------- */

  const [hidden1, setHidden1] =
    useState<number[]>(Array(128).fill(0));
  const [hidden2, setHidden2] =
    useState<number[]>(Array(64).fill(0));
  const [probs, setProbs] =
    useState<number[]>(Array(10).fill(0));

  /* ---------------- prediction stability ---------------- */

  const [stablePrediction, setStablePrediction] =
    useState<number | null>(null);
  const [isConfident, setIsConfident] =
    useState(false);

  /* ---------------- explanation & error ---------------- */

  const [explanation, setExplanation] =
    useState<Explanation | null>(null);
  const [error, setError] =
    useState<string | null>(null);

  /* ---------------- weights ---------------- */

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

  /* =====================================================
     Sync refs
  ===================================================== */

  useEffect(() => {
    pixelsRef.current = pixels;
  }, [pixels]);

  /* =====================================================
     Load weights (ONCE)
  ===================================================== */

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const res = await fetch(`${API_BASE}/weights`);
        if (!res.ok) throw new Error();

        const data = await res.json();
        if (cancelled) return;

        setW12(
          Array.isArray(data.hidden1_hidden2)
            ? data.hidden1_hidden2
            : null
        );
        setW2o(
          Array.isArray(data.hidden2_output)
            ? data.hidden2_output
            : null
        );
      } catch {
        setError("Model weights unavailable");
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  /* =====================================================
     Inference loop (RAF + throttle)
  ===================================================== */

  const infer = useCallback(async (time: number) => {
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

      if (!res.ok) throw new Error();

      const data = await res.json();

      setHidden1((p) =>
        smooth(p, data.layers?.hidden1, HIDDEN_ALPHA)
      );
      setHidden2((p) =>
        smooth(p, data.layers?.hidden2, HIDDEN_ALPHA)
      );
      setProbs((p) =>
        smooth(p, data.probabilities, OUTPUT_ALPHA)
      );

      const current = data.prediction;
      const confidence = data.probabilities?.[current] ?? 0;

      if (lastPredictionRef.current === current) {
        stableCountRef.current++;
      } else {
        lastPredictionRef.current = current;
        stableCountRef.current = 1;
      }

      if (
        stableCountRef.current >= STABLE_FRAMES &&
        confidence >= CONFIDENCE_THRESHOLD
      ) {
        setStablePrediction(current);
        setIsConfident(true);
      } else {
        setIsConfident(false);
      }

      setExplanation(data.explanation ?? null);
      setError(null);
    } catch {
      setError("Prediction failed");
    }

    rafRef.current = requestAnimationFrame(infer);
  }, []);

  useEffect(() => {
    rafRef.current = requestAnimationFrame(infer);
    return () => {
      if (rafRef.current)
        cancelAnimationFrame(rafRef.current);
    };
  }, [infer]);

  /* =====================================================
     Normalized visuals (UI ONLY)
  ===================================================== */

  const nHidden1 = useMemo(() => normalize(hidden1), [hidden1]);
  const nHidden2 = useMemo(() => normalize(hidden2), [hidden2]);
  const nOutput = useMemo(() => normalize(probs), [probs]);

  const maxDigit = useMemo(() => {
    let m = 0;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > probs[m]) m = i;
    }
    return m;
  }, [probs]);

  /* =====================================================
     Render
  ===================================================== */

  return (
    <div className="app landscape">
      <header className="hero">
        <h1>Neural Network Visualization</h1>
        <p>How a neural network interprets your input</p>
      </header>

      <section className="visualizer landscape-layout">
        {/* LEFT */}
        <aside className="panel panel-left">
          <CanvasGrid pixels={pixels} onChange={setPixels} />

          <button
            className="clear-button"
            onClick={() => setPixels([...BLANK_PIXELS])}
          >
            Clear
          </button>

          {stablePrediction !== null && isConfident ? (
            <p className="prediction">
              Prediction: <strong>{stablePrediction}</strong>
            </p>
          ) : (
            <p className="prediction">Prediction: uncertain</p>
          )}

          {error && <p className="error">{error}</p>}
        </aside>

        {/* CENTER */}
        <main className="panel panel-center">
          <button
            className="clear-button"
            onClick={() =>
              setViewMode((v) => (v === "2d" ? "3d" : "2d"))
            }
          >
            Switch to {viewMode === "2d" ? "3D" : "2D"}
          </button>

          {/* ---------- 2D VIEW ---------- */}
          {viewMode === "2d" && (
            <>
              {w12 && w2o ? (
                <>
                  <ConnectionView
                    hidden1={hidden1}
                    hidden2={hidden2}
                    output={probs}
                    weightsHidden1Hidden2={w12}
                    weightsHidden2Output={w2o}
                    width={900}
                    height={320}
                  />

                  <LayerView
                    title="Hidden Layer 1"
                    activations={nHidden1}
                    columns={32}
                    showTitle={false}
                  />

                  <LayerView
                    title="Hidden Layer 2"
                    activations={nHidden2}
                    columns={16}
                    showTitle={false}
                  />
                </>
              ) : (
                <p className="loading">Loading network weights…</p>
              )}
            </>
          )}

          {/* ---------- 3D VIEW ---------- */}
          {viewMode === "3d" && (
            <Network3D
              hidden1={nHidden1}
              hidden2={nHidden2}
              output={nOutput}
              weightsHidden1Hidden2={w12}
              weightsHidden2Output={w2o}
            />
          )}
        </main>

        {/* RIGHT */}
        <aside className="panel panel-right">
          <div className="output-row">
            {probs.map((v, i) => (
              <div
                key={i}
                className={`digit-cell ${
                  i === maxDigit ? "active" : ""
                }`}
              >
                <span>{i}</span>
                <div className="digit-box">
                  <div
                    className="digit-fill"
                    style={{
                      opacity: Math.min(1, v * 2 + 0.1),
                    }}
                  />
                </div>
              </div>
            ))}
          </div>

          {explanation?.top_hidden2_neurons?.length ? (
            <div className="explanation compact">
              <strong>Decision factors</strong>
              <ul>
                {explanation.top_hidden2_neurons
                  .slice(0, 4)
                  .map((n) => (
                    <li key={n.neuron}>
                      h2[{n.neuron}] → {n.contribution.toFixed(2)}
                    </li>
                  ))}
              </ul>
            </div>
          ) : null}
        </aside>
      </section>
    </div>
  );
};

export default App;