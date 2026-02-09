import React, { useEffect, useMemo, useState } from "react";
import CanvasGrid from "./components/CanvasGrid";
import LayerView from "./components/LayerView";
import ConnectionView from "./components/ConnectionView";
import Network3D from "./components/Network3D";
import TrainingMode from "./components/training/TrainingMode";
import { Edge, NeuralState } from "./types/NeuralState";
import { TrainingMessage } from "./types/TrainingMessages";

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
  const [mode, setMode] = useState<"predict" | "train">("predict");
  const [view, setView] = useState<"2d" | "3d">("2d");
  const [state3D, setState3D] = useState<NeuralState | null>(null);
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

  const [trainingSocket, setTrainingSocket] = useState<WebSocket | null>(null);
  const [trainingState3D, setTrainingState3D] = useState<NeuralState | null>(null);
  const [trainingHidden1, setTrainingHidden1] = useState<number[]>(
    Array.from({ length: 128 }, () => 0)
  );
  const [trainingHidden2, setTrainingHidden2] = useState<number[]>(
    Array.from({ length: 64 }, () => 0)
  );
  const [trainingOutput, setTrainingOutput] = useState<number[]>(
    Array.from({ length: 10 }, () => 0)
  );
  const [trainingWeightsHidden1Hidden2, setTrainingWeightsHidden1Hidden2] = useState<
    number[][] | null
  >(null);
  const [trainingWeightsHidden2Output, setTrainingWeightsHidden2Output] = useState<
    number[][] | null
  >(null);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [accuracyHistory, setAccuracyHistory] = useState<number[]>([]);
  const [valLossHistory, setValLossHistory] = useState<number[]>([]);
  const [valAccuracyHistory, setValAccuracyHistory] = useState<number[]>([]);
  const [trainingControls, setTrainingControls] = useState({
    learningRate: 0.001,
    batchSize: 64,
    epochs: 5,
    optimizer: "adam",
    weightDecay: 0,
    activation: "relu",
    initializer: "glorot_uniform",
    dropout: 0,
  });

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
    if (mode !== "predict") {
      return;
    }
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
  }, [pixels, mode]);

  useEffect(() => {
    if (view !== "3d" || mode !== "predict") {
      return;
    }

    const timeout = setTimeout(async () => {
      try {
        const response = await fetch(`${API_BASE}/state`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ pixels }),
        });
        if (!response.ok) {
          throw new Error(await response.text());
        }
        const data = await response.json();
        setState3D(data);
        setError(null);
      } catch (err) {
        setError("3D state fetch failed. Ensure the backend API is reachable.");
      }
    }, 40);

    return () => clearTimeout(timeout);
  }, [pixels, view, mode]);

  useEffect(() => {
    if (mode !== "train") {
      return;
    }

    const ws = new WebSocket(`${API_BASE.replace("http", "ws")}/ws/training`);
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data) as TrainingMessage;
      if (message.type === "batch") {
        setTrainingHidden1(message.activations.hidden1);
        setTrainingHidden2(message.activations.hidden2);
        setTrainingOutput(message.activations.output);
        setLossHistory((prev) => [...prev, message.loss]);
        setAccuracyHistory((prev) => [...prev, message.accuracy]);
      }
      if (message.type === "epoch") {
        setValLossHistory((prev) => [...prev, message.val_loss]);
        setValAccuracyHistory((prev) => [...prev, message.val_accuracy]);
      }
      if (message.type === "weights") {
        setTrainingWeightsHidden1Hidden2(message.weights.hidden1_hidden2);
        setTrainingWeightsHidden2Output(message.weights.hidden2_output);
      }
      if (message.type === "stopped") {
        setError(null);
      }
    };
    ws.onerror = () => setError("Training WebSocket failed.");
    setTrainingSocket(ws);

    return () => {
      ws.close();
      setTrainingSocket(null);
    };
  }, [mode]);

  const maxIndex = probabilities.reduce(
    (max, value, index) => (value > probabilities[max] ? index : max),
    0
  );

  const buildEdges = (weights: number[][], activations: number[], limit: number): Edge[] => {
    const edges: Edge[] = [];
    weights.forEach((row, from) => {
      row.forEach((weight, to) => {
        edges.push({ from, to, strength: weight * (activations[from] ?? 0) });
      });
    });
    return edges
      .sort((a, b) => Math.abs(b.strength) - Math.abs(a.strength))
      .slice(0, limit);
  };

  const handleTrainingCommand = (command: string) => {
    if (!trainingSocket) return;
    if (command === "configure") {
      trainingSocket.send(
        JSON.stringify({
          command: "configure",
          learning_rate: trainingControls.learningRate,
          batch_size: trainingControls.batchSize,
          epochs: trainingControls.epochs,
          optimizer: trainingControls.optimizer,
          weight_decay: trainingControls.weightDecay,
          activation: trainingControls.activation,
          initializer: trainingControls.initializer,
          dropout: trainingControls.dropout,
        })
      );
      return;
    }
    trainingSocket.send(JSON.stringify({ command }));
  };

  useEffect(() => {
    if (view !== "3d" || mode !== "train") {
      return;
    }
    if (!trainingWeightsHidden1Hidden2 || !trainingWeightsHidden2Output) {
      setTrainingState3D(null);
      return;
    }
    setTrainingState3D({
      input: [],
      layers: {
        hidden1: trainingHidden1,
        hidden2: trainingHidden2,
        output: trainingOutput,
      },
      prediction: trainingOutput.indexOf(Math.max(...trainingOutput)),
      confidence: Math.max(...trainingOutput),
      edges: {
        hidden1_hidden2: buildEdges(trainingWeightsHidden1Hidden2, trainingHidden1, 240),
        hidden2_output: buildEdges(trainingWeightsHidden2Output, trainingHidden2, 160),
      },
    });
  }, [
    view,
    mode,
    trainingHidden1,
    trainingHidden2,
    trainingOutput,
    trainingWeightsHidden1Hidden2,
    trainingWeightsHidden2Output,
  ]);

  return (
    <div className="app">
      <header className="hero">
        <h1>Neural network visualization.</h1>
        <p>Ever wondered how AI “sees”?</p>
      </header>

      <div className="mode-toggle">
        <button type="button" onClick={() => setMode((current) => (current === "predict" ? "train" : "predict"))}>
          Switch to {mode === "predict" ? "Training" : "Prediction"} mode
        </button>
      </div>

      {mode === "predict" ? (
        <section className="visualizer">
          <div className="view-toggle">
            <button type="button" onClick={() => setView((current) => (current === "2d" ? "3d" : "2d"))}>
              Switch to {view === "2d" ? "3D" : "2D"}
            </button>
          </div>
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

          {view === "2d" ? (
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
                <LayerView
                  title="Hidden 1"
                  activations={normalizedHidden1}
                  columns={32}
                  showTitle={false}
                />
              </div>
              <div className="layer-row">
                <LayerView
                  title="Hidden 2"
                  activations={normalizedHidden2}
                  columns={16}
                  showTitle={false}
                />
              </div>
            </div>
          ) : (
            <div className="network-stage">
              {state3D ? <Network3D state={state3D} /> : <p>Loading 3D scene…</p>}
            </div>
          )}

          <div className="input-panel">
            <CanvasGrid pixels={pixels} onChange={setPixels} />
            <button type="button" className="clear-button" onClick={() => setPixels(createBlankPixels())}>
              Clear
            </button>
            {prediction !== null && <p className="prediction">Prediction: {prediction}</p>}
            {error && <p className="error">{error}</p>}
          </div>
        </section>
      ) : (
        <TrainingMode
          view={view}
          onToggleView={() => setView((current) => (current === "2d" ? "3d" : "2d"))}
          state3D={trainingState3D}
          hidden1={trainingHidden1}
          hidden2={trainingHidden2}
          output={trainingOutput}
          weightsHidden1Hidden2={trainingWeightsHidden1Hidden2}
          weightsHidden2Output={trainingWeightsHidden2Output}
          lossHistory={lossHistory}
          accuracyHistory={accuracyHistory}
          valLossHistory={valLossHistory}
          valAccuracyHistory={valAccuracyHistory}
          controlState={trainingControls}
          onControlChange={(field, value) =>
            setTrainingControls((prev) => ({ ...prev, [field]: value }))
          }
          onCommand={handleTrainingCommand}
        />
      )}

      {mode === "predict" && (
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
      )}
    </div>
  );
};

export default App;
