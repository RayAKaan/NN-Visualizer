import { useCallback, useMemo, useRef, useState } from "react";
import { BatchUpdate, EpochUpdate, TrainingMessage, TrainingMetrics, TrainingStatus } from "../types";

export function useTrainingSocket() {
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState<TrainingStatus>("idle");
  const [messages, setMessages] = useState<TrainingMessage[]>([]);
  const [metrics, setMetrics] = useState<TrainingMetrics>({ losses: [], accuracies: [], gradientNorms: [], learningRates: [], valLosses: [], valAccuracies: [], epochBoundaries: [], precisionHistory: [], recallHistory: [], f1History: [], confusionMatrices: [] });
  const wsRef = useRef<WebSocket | null>(null);
  const queueRef = useRef<string[]>([]);

  const trainingWsUrl = (() => {
    const envUrl = (import.meta as ImportMeta & { env?: { VITE_TRAIN_WS_URL?: string } }).env?.VITE_TRAIN_WS_URL;
    if (envUrl) return envUrl;
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const host = window.location.hostname || "localhost";
    return `${proto}://${host}:8000/train`;
  })();

  const flushQueue = (ws: WebSocket) => {
    while (queueRef.current.length > 0 && ws.readyState === WebSocket.OPEN) {
      const payload = queueRef.current.shift();
      if (payload) ws.send(payload);
    }
  };

  const connect = useCallback(() => {
    const existing = wsRef.current;
    if (existing && (existing.readyState === WebSocket.OPEN || existing.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const ws = new WebSocket(trainingWsUrl);
    wsRef.current = ws;
    ws.onopen = () => {
      setConnected(true);
      flushQueue(ws);
    };
    ws.onclose = () => {
      setConnected(false);
      if (wsRef.current === ws) wsRef.current = null;
    };
    ws.onerror = () => setStatus("error");
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data) as TrainingMessage;
      setMessages((m) => [...m, msg]);
      if ((msg as BatchUpdate).type === "batch_update") {
        const b = msg as BatchUpdate;
        setMetrics((x) => ({ ...x, losses: [...x.losses, b.loss], accuracies: [...x.accuracies, b.accuracy], gradientNorms: [...x.gradientNorms, b.gradient_norm], learningRates: [...x.learningRates, b.learning_rate] }));
      } else if ((msg as EpochUpdate).type === "epoch_update") {
        const ep = msg as EpochUpdate;
        setMetrics((x) => ({ ...x, valLosses: [...x.valLosses, ep.val_loss], valAccuracies: [...x.valAccuracies, ep.val_accuracy], epochBoundaries: [...x.epochBoundaries, x.losses.length], precisionHistory: [...x.precisionHistory, ep.precision_per_class], recallHistory: [...x.recallHistory, ep.recall_per_class], f1History: [...x.f1History, ep.f1_per_class], confusionMatrices: [...x.confusionMatrices, ep.confusion_matrix] }));
      }
      if ((msg as { type: string }).type === "training_complete") setStatus("completed");
      if ((msg as { type: string }).type === "training_error") setStatus("error");
    };
  }, [trainingWsUrl]);

  const send = (command: string, config?: unknown) => {
    const payload = JSON.stringify({ command, config });
    const ws = wsRef.current;
    if (!ws || ws.readyState === WebSocket.CLOSED || ws.readyState === WebSocket.CLOSING) {
      queueRef.current.push(payload);
      connect();
      return;
    }
    if (ws.readyState === WebSocket.CONNECTING) {
      queueRef.current.push(payload);
      return;
    }
    ws.send(payload);
  };
  const api = useMemo(() => ({ configure: (c: unknown) => send("configure", c), start: () => { setStatus("running"); send("start"); }, pause: () => { setStatus("paused"); send("pause"); }, resume: () => { setStatus("running"); send("resume"); }, stop: () => { setStatus("stopping"); send("stop"); }, stepBatch: () => send("step_batch"), stepEpoch: () => send("step_epoch"), getStatus: () => send("get_status") }), []);
  return { connected, status, messages, metrics, connect, ...api };
}
