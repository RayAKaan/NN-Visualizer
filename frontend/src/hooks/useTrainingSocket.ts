import { useMemo, useRef, useState } from "react";
import { BatchUpdate, EpochUpdate, TrainingMessage, TrainingMetrics, TrainingStatus } from "../types";

export function useTrainingSocket() {
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState<TrainingStatus>("idle");
  const [messages, setMessages] = useState<TrainingMessage[]>([]);
  const [metrics, setMetrics] = useState<TrainingMetrics>({ losses: [], accuracies: [], gradientNorms: [], learningRates: [], valLosses: [], valAccuracies: [], epochBoundaries: [], precisionHistory: [], recallHistory: [], f1History: [], confusionMatrices: [] });
  const wsRef = useRef<WebSocket | null>(null);

  const connect = () => {
    const ws = new WebSocket("ws://localhost:8000/train");
    wsRef.current = ws;
    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
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
  };

  const send = (command: string, config?: unknown) => wsRef.current?.send(JSON.stringify({ command, config }));
  const api = useMemo(() => ({ configure: (c: unknown) => send("configure", c), start: () => { setStatus("running"); send("start"); }, pause: () => { setStatus("paused"); send("pause"); }, resume: () => { setStatus("running"); send("resume"); }, stop: () => { setStatus("stopping"); send("stop"); }, stepBatch: () => send("step_batch"), stepEpoch: () => send("step_epoch"), getStatus: () => send("get_status") }), []);
  return { connected, status, messages, metrics, connect, ...api };
}
