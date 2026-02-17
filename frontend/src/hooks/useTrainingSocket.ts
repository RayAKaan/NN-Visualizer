import { useCallback, useEffect, useRef, useState } from "react";
import { NeuralState, Edge } from "../types/NeuralState";

export interface TrainingConfig {
  learningRate: number;
  batchSize: number;
  epochs: number;
  optimizer: string;
  weightDecay: number;
  activation: string;
  initializer: string;
  dropout: number;
}

type TrainingEvent =
  | {
      type: "batch";
      epoch: number;
      batch: number;
      loss: number;
      accuracy: number;
      activations: {
        hidden1: number[];
        hidden2: number[];
        output: number[];
      };
      gradients: {
        hidden1_hidden2: number[][];
        hidden2_output: number[][];
      };
    }
  | {
      type: "epoch";
      epoch: number;
      loss: number;
      accuracy: number;
      val_loss: number;
      val_accuracy: number;
    }
  | {
      type: "weights";
      epoch: number;
      weights: {
        hidden1_hidden2: number[][];
        hidden2_output: number[][];
      };
    }
  | { type: "stopped" };

/* =====================================================
   Helpers

/** Convert dense matrix → sparse edge list */
function matrixToEdges(
  matrix: number[][] | null,
  threshold = 0.02
): Edge[] {
  if (!matrix) return [];

  const edges: Edge[] = [];

  for (let from = 0; from < matrix.length; from++) {
    const row = matrix[from];
    if (!row) continue;

    for (let to = 0; to < row.length; to++) {
      const w = row[to];
      if (Math.abs(w) >= threshold) {
        edges.push({
          from,
          to,
          strength: w,
        });
      }
    }
  }

  return edges;
}

export function useTrainingSocket(initialConfig: TrainingConfig) {
  const socketRef = useRef<WebSocket | null>(null);

  /* ---------------- lifecycle ---------------- */

  const [ready, setReady] = useState(false);
  const [running, setRunning] = useState(false);

  /* ---------------- control state ---------------- */

  const [controlState, setControlState] =
    useState<TrainingConfig>(initialConfig);

  const updateControl = useCallback(
    (field: string, value: number | string) => {
      setControlState((prev) => ({
        ...prev,
        [field]: value,
      }));
    },
    []
  );

  /* ---------------- activations ---------------- */

  const [hidden1, setHidden1] = useState<number[]>(
    Array(128).fill(0)
  );
  const [hidden2, setHidden2] = useState<number[]>(
    Array(64).fill(0)
  );
  const [output, setOutput] = useState<number[]>(
    Array(10).fill(0)
  );

  /* ---------------- raw weights ---------------- */

  const [w12, setW12] = useState<number[][] | null>(null);
  const [w2o, setW2o] = useState<number[][] | null>(null);

  /* ---------------- metrics ---------------- */

  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [accuracyHistory, setAccuracyHistory] = useState<number[]>([]);
  const [valLossHistory, setValLossHistory] = useState<number[]>([]);
  const [valAccuracyHistory, setValAccuracyHistory] = useState<number[]>([]);

  /* =====================================================
     WebSocket lifecycle
  ===================================================== */

  const connect = useCallback(() => {
    if (socketRef.current) return;

    const ws = new WebSocket("ws://localhost:8000/train");
    socketRef.current = ws;

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data) as TrainingEvent;

      switch (msg.type) {
        case "batch":
          setHidden1(msg.activations.hidden1);
          setHidden2(msg.activations.hidden2);
          setOutput(msg.activations.output);
          setLossHistory((h) => [...h, msg.loss]);
          setAccuracyHistory((h) => [...h, msg.accuracy]);
          setReady(true);
          break;

        case "epoch":
          setValLossHistory((h) => [...h, msg.val_loss]);
          setValAccuracyHistory((h) => [...h, msg.val_accuracy]);
          break;

        case "weights":
          setW12(msg.weights.hidden1_hidden2);
          setW2o(msg.weights.hidden2_output);
          break;

        case "stopped":
          setRunning(false);
          setReady(false);
          break;
      }
    };

    ws.onclose = () => {
      socketRef.current = null;
      setRunning(false);
      setReady(false);
    };
  }, []);

  const disconnect = useCallback(() => {
    socketRef.current?.close();
    socketRef.current = null;
    setRunning(false);
    setReady(false);
  }, []);

  useEffect(() => disconnect, [disconnect]);

  /* =====================================================
     Commands
  ===================================================== */

  const send = useCallback((payload: object) => {
    socketRef.current?.send(JSON.stringify(payload));
  }, []);

  const configure = useCallback(() => {
    connect();
    setLossHistory([]);
    setAccuracyHistory([]);
    setValLossHistory([]);
    setValAccuracyHistory([]);
    setReady(false);

    send({
      command: "configure",
      config: controlState,
    });
  }, [connect, send, controlState]);

  const start = useCallback(() => {
    setRunning(true);
    send({ command: "start" });
  }, [send]);

  const pause = useCallback(() => send({ command: "pause" }), [send]);
  const resume = useCallback(() => send({ command: "resume" }), [send]);

  const stop = useCallback(() => {
    send({ command: "stop" });
    setRunning(false);
    setReady(false);
  }, [send]);

  const stepBatch = useCallback(() => send({ command: "step_batch" }), [send]);
  const stepEpoch = useCallback(() => send({ command: "step_epoch" }), [send]);

  /* =====================================================
     3D State (CORRECT)
  ===================================================== */

  const state3D: NeuralState = {
    input: [],
    layers: {
      hidden1,
      hidden2,
      output,
    },
    prediction: 0,
    confidence: 0,
    edges: {
      hidden1_hidden2: matrixToEdges(w12),
      hidden2_output: matrixToEdges(w2o),
    },
  };

  /* =====================================================
     API
  ===================================================== */

  return {
    ready,
    running,

    controlState,
    updateControl,

    hidden1,
    hidden2,
    output,

    weightsHidden1Hidden2: w12,
    weightsHidden2Output: w2o,

    lossHistory,
    accuracyHistory,
    valLossHistory,
    valAccuracyHistory,

    state3D,

    configure,
    start,
    pause,
    resume,
    stop,
    stepBatch,
    stepEpoch,
  };
}
