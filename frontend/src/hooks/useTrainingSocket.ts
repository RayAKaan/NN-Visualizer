import { useEffect, useRef, useState } from "react";
import { TrainingMessage } from "../types/TrainingMessages";

export const useTrainingSocket = (enabled: boolean, wsBase: string) => {
  const socketRef = useRef<WebSocket | null>(null);
  const [messages, setMessages] = useState<TrainingMessage[]>([]);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    if (!enabled) {
      return;
    }

    const ws = new WebSocket(`${wsBase}/train`);
    socketRef.current = ws;
    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data) as TrainingMessage;
      setMessages((prev) => [...prev, message].slice(-1000));
    };

    return () => {
      ws.close();
      socketRef.current = null;
      setConnected(false);
    };
  }, [enabled, wsBase]);

  const send = (payload: Record<string, unknown>) => {
    if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
      return;
    }
    socketRef.current.send(JSON.stringify(payload));
  };

  return { messages, connected, send };
};
