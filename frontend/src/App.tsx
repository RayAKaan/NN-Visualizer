import React, { useEffect, useMemo, useState } from "react";
import PredictionMode from "./components/PredictionMode";
import TrainingMode from "./components/training/TrainingMode";
import { Edge, NeuralState } from "./types/NeuralState";
import { TrainingMessage } from "./types/TrainingMessages";
import { usePredictionSocket } from "./hooks/usePredictionSocket";
import { useTrainingSocket } from "./hooks/useTrainingSocket";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
const WS_BASE = API_BASE.replace("http", "ws");

const createBlankPixels = () => Array.from({ length: 28 * 28 }, () => 0);

type PredictionExplanation = {
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
};

type TrainingSnapshot = {
  hidden1: number[];
  hidden2: number[];
  output: number[];
  weightsHidden1Hidden2: number[][] | null;
  weightsHidden2Output: number[][] | null;
};

const App: React.FC = () => {
  const [mode, setMode] = useState<"predict" | "train">("predict");
  const [view, setView] = useState<"2d" | "3d">("2d");

  const [pixels, setPixels] = useState<number[]>(createBlankPixels());
  const [weightsHidden1Hidden2, setWeightsHidden1Hidden2] = useState<number[][] | null>(null);
  const [weightsHidden2Output, setWeightsHidden2Output] = useState<number[][] | null>(null);

  const [hidden1, setHidden1] = useState<number[]>(Array.from({ length: 128 }, () => 0));
  const [hidden2, setHidden2] = useState<number[]>(Array.from({ length: 64 }, () => 0));
  const [probabilities, setProbabilities] = useState<number[]>(Array.from({ length: 10 }, () => 0));
  const [prediction, setPrediction] = useState<number | null>(null);
  const [explanation, setExplanation] = useState<PredictionExplanation | null>(null);
  const [state3D, setState3D] = useState<NeuralState | null>(null);
  const [predictError, setPredictError] = useState<string | null>(null);

  const [trainingHidden1, setTrainingHidden1] = useState<number[]>(Array.from({ length: 128 }, () => 0));
  const [trainingHidden2, setTrainingHidden2] = useState<number[]>(Array.from({ length: 64 }, () => 0));
  const [trainingOutput, setTrainingOutput] = useState<number[]>(Array.from({ length: 10 }, () => 0));
  const [trainingWeightsHidden1Hidden2, setTrainingWeightsHidden1Hidden2] = useState<number[][] | null>(null);
  const [trainingWeightsHidden2Output, setTrainingWeightsHidden2Output] = useState<number[][] | null>(null);
  const [trainingState3D, setTrainingState3D] = useState<NeuralState | null>(null);
  const [trainingStatus, setTrainingStatus] = useState("idle");

  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [accuracyHistory, setAccuracyHistory] = useState<number[]>([]);
  const [valLossHistory, setValLossHistory] = useState<number[]>([]);
  const [valAccuracyHistory, setValAccuracyHistory] = useState<number[]>([]);
  const [gradientNormHistory, setGradientNormHistory] = useState<number[]>([]);
  const [learningRateHistory, setLearningRateHistory] = useState<number[]>([]);

  const [snapshots, setSnapshots] = useState<TrainingSnapshot[]>([]);
  const [timelineIndex, setTimelineIndex] = useState(0);
  const [timelineLive, setTimelineLive] = useState(true);

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

  const predictionStream = usePredictionSocket(API_BASE, pixels, mode === "predict");
  const trainingSocket = useTrainingSocket(mode === "train", WS_BASE);

  const normalizedHidden1 = useMemo(() => hidden1.map((value) => Math.min(1, Math.max(0, value))), [hidden1]);
  const normalizedHidden2 = useMemo(() => hidden2.map((value) => Math.min(1, Math.max(0, value))), [hidden2]);
  const normalizedOutput = useMemo(
    () => probabilities.map((value) => Math.min(1, Math.max(0, value))),
    [probabilities]
  );

  const maxIndex = probabilities.reduce((max, value, index) => (value > probabilities[max] ? index : max), 0);

  useEffect(() => {
    const loadWeights = async () => {
      const response = await fetch(`${API_BASE}/weights`);
      if (response.ok) {
        const data = await response.json();
        setWeightsHidden1Hidden2(data.hidden1_hidden2);
        setWeightsHidden2Output(data.hidden2_output);
      }
    };
    loadWeights().catch(() => setPredictError("Failed loading weights"));
  }, []);

  useEffect(() => {
    if (!predictionStream.data) {
      return;
    }
    setHidden1(predictionStream.data.layers.hidden1);
    setHidden2(predictionStream.data.layers.hidden2);
    setProbabilities(predictionStream.data.probabilities);
    setPrediction(predictionStream.data.prediction);
    setExplanation((predictionStream.data as { explanation?: PredictionExplanation }).explanation ?? null);
    setPredictError(predictionStream.error);
  }, [predictionStream.data, predictionStream.error]);

  useEffect(() => {
    if (mode !== "predict" || view !== "3d") {
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
          throw new Error();
        }
        setState3D(await response.json());
      } catch {
        setPredictError("Failed loading 3D prediction state");
      }
    }, 40);
    return () => clearTimeout(timeout);
  }, [mode, view, pixels]);

  const buildEdges = (weights: number[][], activations: number[], limit: number): Edge[] => {
    const edges: Edge[] = [];
    weights.forEach((row, from) => {
      row.forEach((weight, to) => edges.push({ from, to, strength: weight * (activations[from] ?? 0) }));
    });
    return edges.sort((a, b) => Math.abs(b.strength) - Math.abs(a.strength)).slice(0, limit);
  };

  useEffect(() => {
    if (mode !== "train" || trainingSocket.messages.length === 0) {
      return;
    }

    const typed = trainingSocket.messages[trainingSocket.messages.length - 1] as TrainingMessage;
    if (typed.type === "batch_update") {
      setTrainingHidden1(typed.activations.hidden1);
      setTrainingHidden2(typed.activations.hidden2);
      setTrainingOutput(typed.activations.output);
      setTrainingWeightsHidden1Hidden2(typed.weights.hidden1_hidden2);
      setTrainingWeightsHidden2Output(typed.weights.hidden2_output);

      setLossHistory((prev) => [...prev, typed.loss].slice(-1000));
      setAccuracyHistory((prev) => [...prev, typed.accuracy].slice(-1000));
      setGradientNormHistory((prev) => [...prev, typed.gradient_norm].slice(-1000));
      setLearningRateHistory((prev) => [...prev, typed.learning_rate].slice(-1000));

      setSnapshots((prev) => {
        const next = [
          ...prev,
          {
            hidden1: typed.activations.hidden1,
            hidden2: typed.activations.hidden2,
            output: typed.activations.output,
            weightsHidden1Hidden2: typed.weights.hidden1_hidden2,
            weightsHidden2Output: typed.weights.hidden2_output,
          },
        ].slice(-1000);
        if (timelineLive) {
          setTimelineIndex(next.length - 1);
        }
        return next;
      });
    }

    if (typed.type === "epoch_update") {
      setValLossHistory((prev) => [...prev, typed.val_loss].slice(-500));
      setValAccuracyHistory((prev) => [...prev, typed.val_accuracy].slice(-500));
    }

    if (typed.type === "weights_update") {
      setTrainingWeightsHidden1Hidden2(typed.weights.hidden1_hidden2);
      setTrainingWeightsHidden2Output(typed.weights.hidden2_output);
    }

    if (typed.type === "status") {
      setTrainingStatus(typed.status);
    }
    if (typed.type === "training_complete") {
      setTrainingStatus("completed");
    }
    if (typed.type === "training_stopped") {
      setTrainingStatus("stopped");
    }
    if (typed.type === "error") {
      setTrainingStatus(`error: ${typed.message}`);
    }
  }, [trainingSocket.messages, mode, timelineLive]);

  useEffect(() => {
    if (mode !== "train" || view !== "3d") {
      return;
    }
    if (!trainingWeightsHidden1Hidden2 || !trainingWeightsHidden2Output) {
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
  }, [mode, view, trainingHidden1, trainingHidden2, trainingOutput, trainingWeightsHidden1Hidden2, trainingWeightsHidden2Output]);

  useEffect(() => {
    if (mode !== "train") {
      return;
    }
    const current = snapshots[timelineIndex];
    if (!current) {
      return;
    }
    setTrainingHidden1(current.hidden1);
    setTrainingHidden2(current.hidden2);
    setTrainingOutput(current.output);
    setTrainingWeightsHidden1Hidden2(current.weightsHidden1Hidden2);
    setTrainingWeightsHidden2Output(current.weightsHidden2Output);
  }, [mode, snapshots, timelineIndex]);

  const sendTrainingCommand = (command: string) => {
    if (command === "configure") {
      trainingSocket.send({
        command: "configure",
        learning_rate: trainingControls.learningRate,
        batch_size: trainingControls.batchSize,
        epochs: trainingControls.epochs,
        optimizer: trainingControls.optimizer,
        weight_decay: trainingControls.weightDecay,
        activation: trainingControls.activation,
        initializer: trainingControls.initializer,
        dropout: trainingControls.dropout,
      });
      return;
    }
    trainingSocket.send({ command });
  };

  return (
    <div className="app">
      <header className="hero">
        <h1>Neural network visualization.</h1>
        <p>Training + prediction lab with 2D/3D introspection.</p>
      </header>

      <div className="mode-toggle">
        <button type="button" onClick={() => setMode((m) => (m === "predict" ? "train" : "predict"))}>
          Switch to {mode === "predict" ? "Training" : "Prediction"} mode
        </button>
        {mode === "train" && <span className="status-pill">Status: {trainingStatus || "idle"}</span>}
      </div>

      {mode === "predict" ? (
        <PredictionMode
          view={view}
          onToggleView={() => setView((v) => (v === "2d" ? "3d" : "2d"))}
          probabilities={probabilities}
          maxIndex={maxIndex}
          hidden1={normalizedHidden1}
          hidden2={normalizedHidden2}
          output={normalizedOutput}
          state3D={state3D}
          weightsHidden1Hidden2={weightsHidden1Hidden2}
          weightsHidden2Output={weightsHidden2Output}
          pixels={pixels}
          onPixelsChange={setPixels}
          prediction={prediction}
          error={predictError}
        />
      ) : (
        <TrainingMode
          view={view}
          onToggleView={() => setView((v) => (v === "2d" ? "3d" : "2d"))}
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
          gradientNormHistory={gradientNormHistory}
          learningRateHistory={learningRateHistory}
          timelineIndex={timelineIndex}
          timelineMax={Math.max(0, snapshots.length - 1)}
          onTimelineChange={(next) => {
            setTimelineLive(false);
            setTimelineIndex(next);
          }}
          onTimelineReplay={() => {
            setTimelineLive(true);
            setTimelineIndex(Math.max(0, snapshots.length - 1));
          }}
          controlState={trainingControls}
          onControlChange={(field, value) =>
            setTrainingControls((prev) => ({ ...prev, [field]: value }))
          }
          onCommand={sendTrainingCommand}
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
