import React, { useEffect, useMemo, useState } from "react";
import CanvasGrid from "./components/CanvasGrid";
import LayerView from "./components/LayerView";
import ConnectionView from "./components/ConnectionView";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

const createBlankPixels = () => Array.from({ length: 28 * 28 }, () => 0);

const App: React.FC = () => {
  const [pixels, setPixels] = useState<number[]>(createBlankPixels());
  const [hidden1, setHidden1] = useState<number[]>(Array.from({ length: 128 }, () => 0));
  const [hidden2, setHidden2] = useState<number[]>(Array.from({ length: 64 }, () => 0));
  const [probabilities, setProbabilities] = useState<number[]>(
    Array.from({ length: 10 }, () => 0)
  );
  const [prediction, setPrediction] = useState<number | null>(null);
  const [weightsHidden1Hidden2, setWeightsHidden1Hidden2] = useState<number[][] | null>(
    null
  );
  const [weightsHidden2Output, setWeightsHidden2Output] = useState<number[][] | null>(null);
  const [explanation, setExplanation] = useState<{
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
    competing_digits: Array<{ digit: number; probability: number; positive_support: number }>;
    uncertainty_notes: string[];
    hidden1_summary: { mean_activation: number; max_activation: number };
  } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const normalizedHidden1 = useMemo(() => hidden1.map((value) => Math.min(1, value)), [hidden1]);
  const normalizedHidden2 = useMemo(() => hidden2.map((value) => Math.min(1, value)), [hidden2]);
  const normalizedOutput = useMemo(
    () => probabilities.map((value) => Math.min(1, value)),
    [probabilities]
  );

  useEffect(() => {
    const loadWeights = async () => {
      try {
        const response = await fetch(`${API_BASE}/weights`);
        if (!response.ok) {
          throw new Error(await response.text());
        }
        const data = await response.json();
        setWeightsHidden1Hidden2(data.hidden1_hidden2);
        setWeightsHidden2Output(data.hidden2_output);
      } catch (err) {
        setError("Model weights unavailable. Make sure the backend is running and trained.");
      }
    };

    loadWeights();
  }, []);

  useEffect(() => {
    const timeout = setTimeout(async () => {
      try {
        const response = await fetch(`${API_BASE}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ pixels }),
        });
        if (!response.ok) {
          throw new Error(await response.text());
        }
        const data = await response.json();
        setHidden1(data.layers.hidden1);
        setHidden2(data.layers.hidden2);
        setProbabilities(data.probabilities);
        setPrediction(data.prediction);
        setExplanation(data.explanation ?? null);
        setError(null);
      } catch (err) {
        setError("Prediction failed. Ensure the backend API is reachable.");
      }
    }, 30);

    return () => clearTimeout(timeout);
  }, [pixels]);

  const maxIndex = probabilities.reduce(
    (max, value, index) => (value > probabilities[max] ? index : max),
    0
  );

  return (
    <div className="app">
      <header className="hero">
        <h1>Neural network visualization.</h1>
        <p>Ever wondered how AI “sees”?</p>
      </header>

      <section className="visualizer">
        <div className="output-row">
          {probabilities.map((value, index) => (
            <div key={`digit-${index}`} className="digit-cell">
              <span className="digit-label">{index}</span>
              <div className={`digit-box ${index === maxIndex ? "active" : ""}`}>
                <div className="digit-fill" style={{ opacity: Math.min(1, value * 2 + 0.1) }} />
              </div>
            </div>
          ))}
        </div>

        <div className="network-stage">
          <ConnectionView
            hidden1={normalizedHidden1}
            hidden2={normalizedHidden2}
            output={normalizedOutput}
            weightsHidden1Hidden2={weightsHidden1Hidden2}
            weightsHidden2Output={weightsHidden2Output}
            width={720}
            height={220}
          />
          <div className="layer-row">
            <LayerView title="Hidden 1" activations={normalizedHidden1} columns={32} showTitle={false} />
          </div>
          <div className="layer-row">
            <LayerView title="Hidden 2" activations={normalizedHidden2} columns={16} showTitle={false} />
          </div>
        </div>

        <div className="input-panel">
          <CanvasGrid pixels={pixels} onChange={setPixels} />
          <button type="button" className="clear-button" onClick={() => setPixels(createBlankPixels())}>
            Clear
          </button>
          {prediction !== null && <p className="prediction">Prediction: {prediction}</p>}
          {error && <p className="error">{error}</p>}
        </div>
      </section>

      <section className="explanation-panel">
        <h2>Decision explanation</h2>
        {explanation ? (
          <div className="explanation">
            <p>
              Primary digit: <strong>{explanation.prediction}</strong> (confidence{" "}
              {(explanation.confidence * 100).toFixed(1)}%, margin{" "}
              {(explanation.margin * 100).toFixed(1)}%).
            </p>
            <div>
              <strong>Top contributing hidden neurons</strong>
              <ul>
                {explanation.top_hidden2_neurons.map((item) => (
                  <li key={`neuron-${item.neuron}`}>
                    h2[{item.neuron}] activation {item.activation.toFixed(3)} × weight{" "}
                    {item.weight.toFixed(3)} = {item.contribution.toFixed(3)}
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <strong>Supporting evidence</strong>
              <p>
                Quadrant intensity — UL {explanation.quadrant_intensity.upper_left.toFixed(3)},
                UR {explanation.quadrant_intensity.upper_right.toFixed(3)}, LL{" "}
                {explanation.quadrant_intensity.lower_left.toFixed(3)}, LR{" "}
                {explanation.quadrant_intensity.lower_right.toFixed(3)}.
              </p>
              <p>
                Hidden1 summary — mean {explanation.hidden1_summary.mean_activation.toFixed(3)},
                max {explanation.hidden1_summary.max_activation.toFixed(3)}.
              </p>
            </div>
            <div>
              <strong>Competing digits</strong>
              <ul>
                {explanation.competing_digits.map((digit) => (
                  <li key={`compete-${digit.digit}`}>
                    {digit.digit}: {(digit.probability * 100).toFixed(1)}% (positive support{" "}
                    {digit.positive_support.toFixed(3)})
                  </li>
                ))}
              </ul>
            </div>
            {explanation.uncertainty_notes.length > 0 && (
              <div>
                <strong>Uncertainty notes</strong>
                <ul>
                  {explanation.uncertainty_notes.map((note, index) => (
                    <li key={`note-${index}`}>{note}</li>
                  ))}
                </ul>
              </div>
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
