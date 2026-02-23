import { useState, useEffect, useCallback, useRef } from "react";
import {
  TrainingStatus,
  TrainingConfig,
  BatchUpdate,
  EpochUpdate,
  TrainingMessage,
  TrainingMetrics,
} from "../types";

const WS_URL = "ws://localhost:8000/train";

export function useTrainingSocket() {
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const [status, setStatus] = useState<TrainingStatus>("idle");
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [currentBatch, setCurrentBatch] = useState(0);
  const [totalBatches, setTotalBatches] = useState(0);
  const [totalEpochs, setTotalEpochs] = useState(0);

  const [latestBatch, setLatestBatch] = useState<BatchUpdate | null>(null);
  const [latestEpoch, setLatestEpoch] = useState<EpochUpdate | null>(null);

  const [metrics, setMetrics] = useState<TrainingMetrics>({
    losses: [],
    accuracies: [],
    gradientNorms: [],
    learningRates: [],
    valLosses: [],
    valAccuracies: [],
    epochBoundaries: [],
  });

  const [epochSummaries, setEpochSummaries] = useState<EpochUpdate[]>([]);
  const [messageCount, setMessageCount] = useState(0);

  const [completionInfo, setCompletionInfo] = useState<{
    epochs: number;
    snapshots: number;
    finalAccuracy: number;
  } | null>(null);

  const [error, setError] = useState<string | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      setConnected(true);
      setError(null);
    };

    ws.onclose = () => {
      setConnected(false);
      wsRef.current = null;
    };

    ws.onerror = () => {
      setConnected(false);
      setError("WebSocket connection failed");
    };

    ws.onmessage = (event) => {
      try {
        const msg: TrainingMessage = JSON.parse(event.data);
        setMessageCount((prev) => prev + 1);

        switch (msg.type) {
          case "batch_update": {
            const bu = msg as BatchUpdate;
            setLatestBatch(bu);
            setCurrentEpoch(bu.epoch);
            setCurrentBatch(bu.batch);
            setTotalBatches(bu.total_batches);
            setTotalEpochs(bu.total_epochs);
            setStatus("running");

            setMetrics((prev) => ({
              ...prev,
              losses: [...prev.losses.slice(-999), bu.loss],
              accuracies: [...prev.accuracies.slice(-999), bu.accuracy],
              gradientNorms: [...prev.gradientNorms.slice(-999), bu.gradient_norm],
              learningRates: [...prev.learningRates.slice(-999), bu.learning_rate],
            }));
            break;
          }

          case "epoch_update": {
            const eu = msg as EpochUpdate;
            setLatestEpoch(eu);
            setEpochSummaries((prev) => [...prev, eu]);

            setMetrics((prev) => ({
              ...prev,
              valLosses: [...prev.valLosses, eu.val_loss],
              valAccuracies: [...prev.valAccuracies, eu.val_accuracy],
              epochBoundaries: [...prev.epochBoundaries, prev.losses.length],
            }));
            break;
          }

          case "training_complete": {
            const tc = msg as any;
            setStatus("completed");
            setCompletionInfo({
              epochs: tc.epochs,
              snapshots: tc.total_snapshots,
              finalAccuracy: tc.final_accuracy,
            });
            break;
          }

          case "training_stopped": {
            setStatus("idle");
            break;
          }

          case "training_error": {
            setStatus("error");
            setError((msg as any).error);
            break;
          }

          case "status": {
            const sm = msg as any;
            if (sm.status) setStatus(sm.status);
            break;
          }

          case "command_response": {
            const cr = msg as any;
            if (cr.status) setStatus(cr.status);
            break;
          }
        }
      } catch (e) {
        console.error("Failed to parse training message", e);
      }
    };

    wsRef.current = ws;
  }, []);

  const disconnect = useCallback(() => {
    wsRef.current?.close();
  }, []);

  const send = useCallback((command: string, data?: Record<string, any>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ command, ...data }));
    }
  }, []);

  const configure = useCallback(
    (config: Partial<TrainingConfig>) => {
      send("configure", { config });
    },
    [send]
  );

  const start = useCallback(() => {
    setMetrics({
      losses: [],
      accuracies: [],
      gradientNorms: [],
      learningRates: [],
      valLosses: [],
      valAccuracies: [],
      epochBoundaries: [],
    });
    setEpochSummaries([]);
    setCompletionInfo(null);
    setError(null);
    setMessageCount(0);
    setLatestBatch(null);
    setLatestEpoch(null);
    send("start");
  }, [send]);

  const pause = useCallback(() => send("pause"), [send]);
  const resume = useCallback(() => send("resume"), [send]);
  const stop = useCallback(() => send("stop"), [send]);
  const stepBatch = useCallback(() => send("step_batch"), [send]);
  const stepEpoch = useCallback(() => send("step_epoch"), [send]);

  useEffect(() => {
    return () => {
      wsRef.current?.close();
    };
  }, []);

  return {
    connected,
    connect,
    disconnect,
    status,
    currentEpoch,
    currentBatch,
    totalBatches,
    totalEpochs,
    error,
    completionInfo,
    latestBatch,
    latestEpoch,
    metrics,
    epochSummaries,
    messageCount,
    configure,
    start,
    pause,
    resume,
    stop,
    stepBatch,
    stepEpoch,
  };
}
