import { useEffect, useRef, useState, useCallback } from "react";
import { WS_URL } from "../api/client";
import { TrainingBatchMetrics, TrainingMetrics, TrainingStatus } from "../types";

export function useTrainingSocket() {
  const ws = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<number | null>(null);
  const commandQueue = useRef<Array<{ action: string; config?: any }>>([]);
  const disposedRef = useRef(false);
  const [isConnected, setIsConnected] = useState(false);
  const [status, setStatus] = useState<TrainingStatus>({
    status: "idle",
    current_epoch: 0,
    total_epochs: 0,
  });
  const [history, setHistory] = useState<TrainingMetrics[]>([]);
  const [batchHistory, setBatchHistory] = useState<TrainingBatchMetrics[]>([]);
  const [liveBatch, setLiveBatch] = useState<TrainingBatchMetrics | null>(null);
  const [logs, setLogs] = useState<string[]>([]);

  const flushQueue = useCallback(() => {
    if (ws.current?.readyState !== WebSocket.OPEN) return;
    while (commandQueue.current.length > 0) {
      const cmd = commandQueue.current.shift();
      if (!cmd) break;
      ws.current.send(JSON.stringify(cmd));
    }
  }, []);

  const connect = useCallback(() => {
    if (
      ws.current?.readyState === WebSocket.OPEN ||
      ws.current?.readyState === WebSocket.CONNECTING
    ) {
      return;
    }

    disposedRef.current = false;
    const socket = new WebSocket(WS_URL);
    ws.current = socket;

    socket.onopen = () => {
      if (ws.current !== socket || disposedRef.current) return;
      setIsConnected(true);
      setLogs((prev) => [...prev, "Connected to training server."]);
      socket.send(JSON.stringify({ action: "status" }));
      flushQueue();
    };

    socket.onmessage = (event) => {
      if (ws.current !== socket) return;
      const data = JSON.parse(event.data);

      switch (data.type) {
        case "status_response":
        case "status":
          if (data.status) {
             const normalizedStatus = data.status === "training_started" ? "training" : data.status === "stopping" ? "stopping" : data.status;
             setStatus(prev => ({
                 ...prev,
                 status: normalizedStatus,
                 current_epoch: data.current_epoch || prev.current_epoch,
                 total_epochs: data.total_epochs || prev.total_epochs
             }));
          }
          if (data.status === "training_started") {
             setHistory([]);
             setBatchHistory([]);
             setLiveBatch(null);
             setLogs(prev => [...prev, "Training started..."]);
          }
          break;

        case "batch": {
          const point: TrainingBatchMetrics = {
            epoch: data.epoch,
            batch: data.batch,
            total_batches: data.total_batches,
            total_epochs: data.total_epochs,
            model_type: data.model_type,
            loss: data.loss,
            accuracy: data.accuracy,
            learning_rate: data.learning_rate,
            gradient_norm: data.gradient_norm,
            timestamp: data.timestamp,
          };
          setLiveBatch(point);
          setBatchHistory((prev) => {
            // Reset visuals when a brand-new run starts over from epoch/batch 0.
            if (point.epoch <= 1 && point.batch <= 1 && prev.length > 0) {
              const last = prev[prev.length - 1];
              if (last.epoch > point.epoch || (last.epoch === point.epoch && last.batch > point.batch)) {
                return [point];
              }
            }
            const next = [...prev, point];
            return next.length > 500 ? next.slice(next.length - 500) : next;
          });
          setStatus((prev) => ({
            ...prev,
            status: "training",
            current_epoch: data.epoch || prev.current_epoch,
            total_epochs: data.total_epochs || prev.total_epochs,
          }));
          break;
        }

        case "epoch":
          setHistory((prev) => {
            // A newer run restarting from an earlier epoch means the old
            // history belongs to a previous session - start fresh.
            const last = prev[prev.length - 1];
            if (last && data.epoch < last.epoch) return [data];
            return [...prev, data];
          });
          setStatus((prev) => ({
            ...prev,
            current_epoch: data.epoch,
            total_epochs: data.total_epochs || prev.total_epochs,
          }));
          break;

        case "complete":
          setStatus((prev) => ({ ...prev, status: "completed" }));
          setLogs((prev) => [...prev, `Training complete. Test Acc: ${(data.test_accuracy * 100).toFixed(2)}%`]);
          break;

        case "error":
          setLogs((prev) => [...prev, `Error: ${data.message}`]);
          setStatus((prev) => ({ ...prev, status: "idle" }));
          break;
      }
    };

    socket.onclose = () => {
      const isCurrent = ws.current === socket;
      if (isCurrent) setIsConnected(false);
      // Stale sockets (already replaced) or intentional closes never reconnect.
      if (!isCurrent || disposedRef.current) return;
      if (reconnectTimer.current) {
        window.clearTimeout(reconnectTimer.current);
      }
      reconnectTimer.current = window.setTimeout(() => {
        if (!disposedRef.current) connect();
      }, 1500);
    };
  }, [flushQueue]);

  const sendCommand = (action: string, config?: any) => {
    if (action === "start") {
      setStatus((prev) => ({
        ...prev,
        status: "training",
        current_epoch: 0,
        total_epochs: config?.epochs ?? prev.total_epochs,
      }));
      setHistory([]);
      setBatchHistory([]);
      setLiveBatch(null);
      setLogs((prev) => [...prev, "Start requested..."]);
    }
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ action, config }));
      return;
    }
    commandQueue.current.push({ action, config });
    setLogs((prev) => [...prev, `Queued command: ${action} (socket not ready)`]);
    connect();
  };

  useEffect(() => {
    disposedRef.current = false;
    connect();
    return () => {
      disposedRef.current = true;
      if (reconnectTimer.current) {
        window.clearTimeout(reconnectTimer.current);
      }
      commandQueue.current = [];
      ws.current?.close();
    };
  }, [connect]);

  return { status, history, batchHistory, liveBatch, logs, sendCommand, isConnected, connect };
}
