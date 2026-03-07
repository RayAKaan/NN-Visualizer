import { useEffect, useMemo, useRef, useState } from "react";
import { useNeurofluxStore } from "../store/useNeurofluxStore";
import { EdgeState, NeuronState } from "../components/neurofluxion/types";

type ConnectionStatus = "connecting" | "connected" | "disconnected";

interface StreamPacket {
  epoch?: number;
  metrics?: {
    epoch: number;
    loss: number;
    accuracy: number;
  };
  topology?: {
    neurons: NeuronState[];
    edges: EdgeState[];
  };
}

export function useWebSocketStream(url = "ws://localhost:8000/stream") {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectRef = useRef<number | null>(null);
  const retryRef = useRef(0);
  const aliveRef = useRef(true);
  const [status, setStatus] = useState<ConnectionStatus>("connecting");
  const addTopologySnapshot = useNeurofluxStore((s) => s.addTopologySnapshot);
  const addMetricsSnapshot = useNeurofluxStore((s) => s.addMetricsSnapshot);

  useEffect(() => {
    aliveRef.current = true;
    const connect = () => {
      if (!aliveRef.current) return;
      setStatus("connecting");
      wsRef.current = new WebSocket(url);

      wsRef.current.onopen = () => {
        retryRef.current = 0;
        setStatus("connected");
      };

      wsRef.current.onmessage = (event) => {
        try {
          const parsed: StreamPacket = JSON.parse(event.data);
          if (parsed.topology?.neurons && parsed.topology?.edges) {
            addTopologySnapshot(parsed.topology);
          }
          if (parsed.metrics) {
            addMetricsSnapshot(parsed.metrics);
          }
        } catch {
          // Ignore malformed packets to keep stream resilient.
        }
      };

      wsRef.current.onclose = () => {
        setStatus("disconnected");
        if (reconnectRef.current) window.clearTimeout(reconnectRef.current);
        retryRef.current += 1;
        const delay = Math.min(10000, 1200 * Math.max(1, retryRef.current));
        reconnectRef.current = window.setTimeout(connect, delay);
      };

      wsRef.current.onerror = () => {
        setStatus("disconnected");
      };
    };

    connect();
    return () => {
      aliveRef.current = false;
      if (reconnectRef.current) window.clearTimeout(reconnectRef.current);
      wsRef.current?.close();
    };
  }, [addMetricsSnapshot, addTopologySnapshot, url]);

  const indicatorColor = useMemo(() => {
    if (status === "connected") return "bg-emerald-400";
    if (status === "connecting") return "bg-amber-400";
    return "bg-rose-400";
  }, [status]);

  return { status, indicatorColor };
}
