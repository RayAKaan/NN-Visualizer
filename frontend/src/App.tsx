import React, { useEffect, useMemo, useRef, useState } from "react";
import CanvasGrid from "./components/CanvasGrid";
import LayerView from "./components/LayerView";
import ConnectionView from "./components/ConnectionView";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

const BLANK_PIXELS = Array.from({ length: 28 * 28 }, () => 0);

/* ---------------- tuning constants ---------------- */

const FRAME_INTERVAL = 33;          // ~30 FPS
const HIDDEN_ALPHA = 0.35;          // fast response
const OUTPUT_ALPHA = 0.18;          // slower, more stable
const CONFIDENCE_THRESHOLD = 0.75;  // confidence gate
const STABLE_FRAMES = 6;            // hysteresis frames

/* ---------------- helpers ---------------- */

const smooth = (prev: number[], next: number[], alpha: number) =>
  next.map((v, i) => prev[i] + alpha * (v - prev[i]));

const normalize = (layer: number[]) => {
  const max = Math.max(...layer, 1e-6);
  return layer.map((v) => v / max);
};

/* ---------------- types ---------------- */

type Explanation = {
  prediction: number;
  confidence: number;
  margin: number;
  top_hidden2_neurons: Array<{
    neuron: number;
    activation: number;
    weight: number;
    contribution: number;
  }>;
  quadrant_intensity: Record<string, number>;
  competing_digits: Array<{
    digit: number;
    probability: number;
    positive_support: number;
  }>;
  uncertainty_notes: string[];
  hidden1_summary: {
    mean_activation: number;
    max_activation: number;
  };
};

/* ---------------- app ---------------- */

const App: React.FC = () => {
  /* input */
  const [pixels, setPixels] = useState<number[]>(BLANK_PIXELS);

  /* raw activations */
  const [hidden1, setHidden1] = useState<number[]>(Array(128).fill(0));
  const [hidden2, setHidden2] = useState<number[]>(Array(64).fill(0));
  const [probs, setProbs] = useState<number[]>(Array(10).fill(0));

  /* semantic outputs */
  const [stablePrediction, setStablePrediction] = useState<number | null>(null);
  const [isConfident, setIsConfident] = useState(false);

  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [error, setError] = useState<string | null>(null);

  /* weights */
  const [w12, setW12] = useState<number[][] | null>(null);
  const [w2o, setW2o] = useState<number[][] | null>(null);

  /* refs */
  const rafRef = useRef<number | null>(null);
  const pixelsRef = useRef<number[]>(pixels);
  const lastTimeRef = useRef(0);

  const lastPredictionRef = useRef<number | null>(null);
  const stableCountRef = useRef(0);

  useEffect(() => {
    pixelsRef.current = pixels;
  }, [pixels]);

  /* -------- load weights -------- */

  useEffect(() => {
    const load = async () => {
      try {
        const res = await fetch(`${API_BASE}/weights`);
        if (!res.ok) throw new Error();
        const data = await res.json();
        setW12(data.hidden1_hidden2);
        setW2o(data.hidden2_output);
      } catch {
        setError("Model weights unavailable.");
      }
    };
    load();
  }, []);

  /* -------- realtime inference loop -------- */

  useEffect(() => {
    const loop = async (t: number) => {
      if (t - lastTimeRef.current >= FRAME_INTERVAL) {
        lastTimeRef.current = t;

        try {
          const res = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ pixels: pixelsRef.current }),
          });

          if (res.ok) {
            const data = await res.json();

            /* smooth activations */
            setHidden1((p) => smooth(p, data.layers.hidden1, HIDDEN_ALPHA));
            setHidden2((p) => smooth(p, data.layers.hidden2, HIDDEN_ALPHA));
            setProbs((p) => smooth(p, data.probabilities, OUTPUT_ALPHA));

            /* prediction orchestration */
            const current = data.prediction;
            const confidence = data.probabilities[current] ?? 0;

            if (lastPredictionRef.current === current) {
              stableCountRef.current += 1;
            } else {
              stableCountRef.current = 1;
              lastPredictionRef.current = current;
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
          }
        } catch {
          setError("Prediction failed.");
        }
      }

      rafRef.current = requestAnimationFrame(loop);
    };

    rafRef.current = requestAnimationFrame(loop);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  /* -------- normalized visuals -------- */

  const nHidden1 = useMemo(() => normalize(hidden1), [hidden1]);
  const nHidden2 = useMemo(() => normalize(hidden2), [hidden2]);
  const nOutput = useMemo(() => normalize(probs), [probs]);

  const maxDigit = probs.reduce(
    (m, v, i) => (v > probs[m] ? i : m),
    0
  );

  /* ---------------- render ---------------- */

  return (
    <div className="app">
      <header className="hero">
        <h1>Neural network visualization.</h1>
        <p>Ever wondered how AI “sees”?</p>
      </header>

      <section className="visualizer">
        {/* output */}
        <div className="output-row">
          {probs.map((v, i) => (
            <div key={i} className="digit-cell">
              <span className="digit-label">{i}</span>
              <div className={`digit-box ${i === maxDigit ? "active" : ""}`}>
                <div
                  className="digit-fill"
                  style={{ opacity: Math.min(1, v * 2 + 0.1) }}
                />
              </div>
            </div>
          ))}
        </div>

        {/* network */}
        <div className="network-stage">
          <ConnectionView
            hidden1={nHidden1}
            hidden2={nHidden2}
            output={nOutput}
            weightsHidden1Hidden2={w12}
            weightsHidden2Output={w2o}
            width={830}
            height={255}
          />

          <div className="layer-row">
            <LayerView
              title="Hidden Layer 1"
              activations={nHidden1}
              columns={32}
              showTitle={false}
            />
          </div>

          <div className="layer-row">
            <LayerView
              title="Hidden Layer 2"
              activations={nHidden2}
              columns={16}
              showTitle={false}
            />
          </div>
        </div>

        {/* input */}
        <div className="input-panel">
          <CanvasGrid pixels={pixels} onChange={setPixels} />
          <button
            className="clear-button"
            onClick={() => setPixels([...BLANK_PIXELS])}
          >
            Clear
          </button>

          {stablePrediction !== null && isConfident && (
            <p className="prediction">
              Prediction: <strong>{stablePrediction}</strong>
            </p>
          )}

          {!isConfident && <p className="prediction">Prediction: uncertain</p>}

          {error && <p className="error">{error}</p>}
        </div>
      </section>

      <section className="explanation-panel">
        <h2>Decision explanation</h2>

        {explanation ? (
          <div className="explanation">
            <p>
              Primary digit <strong>{explanation.prediction}</strong> with{" "}
              {(explanation.confidence * 100).toFixed(1)}% confidence.
            </p>

            <ul>
              {explanation.top_hidden2_neurons.map((n) => (
                <li key={n.neuron}>
                  h2[{n.neuron}] → contribution {n.contribution.toFixed(3)}
                </li>
              ))}
            </ul>

            {explanation.uncertainty_notes.length > 0 && (
              <ul>
                {explanation.uncertainty_notes.map((n, i) => (
                  <li key={i}>{n}</li>
                ))}
              </ul>
            )}
          </div>
        ) : (
          <p>No explanation available yet.</p>
        )}
      </section>
    </div>
  );
};

export default App;