import React, { useEffect, useState } from "react";
import TrainingMode from "./TrainingMode";
import { TrainingMessage } from "../../types/TrainingMessages";
import { NeuralState } from "../../types";
import { useTrainingSocket } from "../../hooks/useTrainingSocket";

interface Props {
  apiBase: string;
  view: "2d" | "3d";
  onToggleView: () => void;
}

type TrainingSnapshot = {
  hidden1: number[];
  hidden2: number[];
  output: number[];
  gradientsHidden1Hidden2: number[][] | null;
  gradientsHidden2Output: number[][] | null;
  weightsHidden1Hidden2: number[][] | null;
  weightsHidden2Output: number[][] | null;
};

const TrainingShell: React.FC<Props> = ({ apiBase, view, onToggleView }) => {
  const wsBase = apiBase.replace("http", "ws");
  const trainingSocket = useTrainingSocket(true, wsBase);

  const [trainingHidden1, setTrainingHidden1] = useState<number[]>(Array.from({ length: 128 }, () => 0));
  const [trainingHidden2, setTrainingHidden2] = useState<number[]>(Array.from({ length: 64 }, () => 0));
  const [trainingOutput, setTrainingOutput] = useState<number[]>(Array.from({ length: 10 }, () => 0));
  const [w12, setW12] = useState<number[][] | null>(null);
  const [w2o, setW2o] = useState<number[][] | null>(null);
  const [g12, setG12] = useState<number[][] | null>(null);
  const [g2o, setG2o] = useState<number[][] | null>(null);
  const [trainingState3D, setTrainingState3D] = useState<NeuralState | null>(null);

  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [accuracyHistory, setAccuracyHistory] = useState<number[]>([]);
  const [valLossHistory, setValLossHistory] = useState<number[]>([]);
  const [valAccuracyHistory, setValAccuracyHistory] = useState<number[]>([]);
  const [gradientNormHistory, setGradientNormHistory] = useState<number[]>([]);
  const [learningRateHistory, setLearningRateHistory] = useState<number[]>([]);
  const [weightHistory, setWeightHistory] = useState<number[]>([]);
  const [status, setStatus] = useState("idle");

  const [snapshots, setSnapshots] = useState<TrainingSnapshot[]>([]);
  const [timelineIndex, setTimelineIndex] = useState(0);
  const [timelineLive, setTimelineLive] = useState(true);
  const [controls, setControls] = useState({
    learningRate: 0.001,
    batchSize: 64,
    epochs: 5,
    optimizer: "adam",
    weightDecay: 0,
    activation: "relu",
    initializer: "glorot_uniform",
    dropout: 0,
  });

  useEffect(() => {
    if (trainingSocket.messages.length === 0) return;
    const typed = trainingSocket.messages[trainingSocket.messages.length - 1] as TrainingMessage;
    if (typed.type === "batch_update") {
      setTrainingHidden1(typed.activations.hidden1);
      setTrainingHidden2(typed.activations.hidden2);
      setTrainingOutput(typed.activations.output);
      setW12(typed.weights.hidden1_hidden2);
      setW2o(typed.weights.hidden2_output);
      setG12(typed.gradients.hidden1_hidden2);
      setG2o(typed.gradients.hidden2_output);
      setLossHistory((prev) => [...prev, typed.loss].slice(-1000));
      setAccuracyHistory((prev) => [...prev, typed.accuracy].slice(-1000));
      setGradientNormHistory((prev) => [...prev, typed.gradient_norm].slice(-1000));
      setLearningRateHistory((prev) => [...prev, typed.learning_rate].slice(-1000));
      setWeightHistory((prev) => [...prev, Math.abs(typed.weights.hidden2_output[0]?.[0] ?? 0)].slice(-1000));
      setSnapshots((prev) => {
        const next = [...prev, {
          hidden1: typed.activations.hidden1,
          hidden2: typed.activations.hidden2,
          output: typed.activations.output,
          gradientsHidden1Hidden2: typed.gradients.hidden1_hidden2,
          gradientsHidden2Output: typed.gradients.hidden2_output,
          weightsHidden1Hidden2: typed.weights.hidden1_hidden2,
          weightsHidden2Output: typed.weights.hidden2_output,
        }].slice(-1000);
        if (timelineLive) setTimelineIndex(next.length - 1);
        return next;
      });
    }
    if (typed.type === "epoch_update") {
      setValLossHistory((prev) => [...prev, typed.val_loss].slice(-500));
      setValAccuracyHistory((prev) => [...prev, typed.val_accuracy].slice(-500));
    }
    if (typed.type === "status") setStatus(typed.status);
  }, [trainingSocket.messages, timelineLive]);

  useEffect(() => {
    if (trainingSocket.latestState) setTrainingState3D(trainingSocket.latestState);
  }, [trainingSocket.latestState]);

  useEffect(() => {
    const current = snapshots[timelineIndex];
    if (!current) return;
    setTrainingHidden1(current.hidden1);
    setTrainingHidden2(current.hidden2);
    setTrainingOutput(current.output);
    setG12(current.gradientsHidden1Hidden2);
    setG2o(current.gradientsHidden2Output);
    setW12(current.weightsHidden1Hidden2);
    setW2o(current.weightsHidden2Output);
  }, [snapshots, timelineIndex]);

  const sendCommand = (command: string) => {
    if (command === "configure") {
      trainingSocket.send({
        command,
        learning_rate: controls.learningRate,
        batch_size: controls.batchSize,
        epochs: controls.epochs,
        optimizer: controls.optimizer,
        weight_decay: controls.weightDecay,
        activation: controls.activation,
        initializer: controls.initializer,
        dropout: controls.dropout,
      });
      return;
    }
    trainingSocket.send({ command });
  };

  return (
    <div>
      <div className="status-pill">Status: {status}</div>
      <TrainingMode
        view={view}
        onToggleView={onToggleView}
        state3D={trainingState3D}
        hidden1={trainingHidden1}
        hidden2={trainingHidden2}
        output={trainingOutput}
        weightsHidden1Hidden2={w12}
        weightsHidden2Output={w2o}
        gradientsHidden1Hidden2={g12}
        gradientsHidden2Output={g2o}
        lossHistory={lossHistory}
        accuracyHistory={accuracyHistory}
        valLossHistory={valLossHistory}
        valAccuracyHistory={valAccuracyHistory}
        gradientNormHistory={gradientNormHistory}
        learningRateHistory={learningRateHistory}
        weightHistory={weightHistory}
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
        onReplaySnapshot={(snapshot) => {
          setTrainingHidden1(snapshot.activations.hidden1);
          setTrainingHidden2(snapshot.activations.hidden2);
          setTrainingOutput(snapshot.activations.output);
          setG12(snapshot.gradients.hidden1_hidden2);
          setG2o(snapshot.gradients.hidden2_output);
          setW12(snapshot.weights.hidden1_hidden2);
          setW2o(snapshot.weights.hidden2_output);
        }}
        apiBase={apiBase}
        controlState={controls}
        onControlChange={(field, value) => setControls((prev) => ({ ...prev, [field]: value }))}
        onCommand={sendCommand}
      />
    </div>
  );
};

export default TrainingShell;
