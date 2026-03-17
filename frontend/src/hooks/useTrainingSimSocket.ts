import { useCallback, useEffect, useRef, useState } from "react";
import { SIM_WS_URL } from "../api/client";
import { useTrainingSimStore } from "../store/trainingSimStore";

export function useTrainingSimSocket() {
  const ws = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const setStatus = useTrainingSimStore((s) => s.setStatus);
  const pushMetrics = useTrainingSimStore((s) => s.pushMetrics);
  const pushWarning = useTrainingSimStore((s) => s.pushWarning);

  const connect = useCallback(() => {
    if (ws.current?.readyState === WebSocket.OPEN || ws.current?.readyState === WebSocket.CONNECTING) return;
    ws.current = new WebSocket(SIM_WS_URL);
    ws.current.onopen = () => setIsConnected(true);
    ws.current.onclose = () => setIsConnected(false);
    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "status") {
        setStatus({ isTraining: data.status === "training", isPaused: data.status === "paused" });
      }
      if (data.type === "batch") {
        setStatus({ currentEpoch: data.epoch, currentBatch: data.batch, totalBatches: data.total_batches });
      }
      if (data.type === "epoch") {
        pushMetrics(data.epoch, data.metrics);
      }
      if (data.type === "warning") {
        pushWarning(data);
      }
      if (data.type === "complete") {
        setStatus({ isTraining: false, isPaused: false });
      }
    };
  }, [pushMetrics, pushWarning, setStatus]);

  const send = useCallback((payload: any) => {
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) {
      connect();
      setTimeout(() => ws.current?.send(JSON.stringify(payload)), 300);
      return;
    }
    ws.current.send(JSON.stringify(payload));
  }, [connect]);

  useEffect(() => {
    connect();
    return () => ws.current?.close();
  }, [connect]);

  return { isConnected, send };
}

