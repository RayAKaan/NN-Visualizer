import { useEffect, useRef, useState } from "react";
import { NeuralState } from "../types/NeuralState";
import { TrainingMessage } from "../types/TrainingMessages";
import { matrixToEdges } from "../utils/edgeConverter";

export const useTrainingSocket = (enabled: boolean, wsBase: string) => {
  const socketRef = useRef<WebSocket | null>(null);
  const [messages, setMessages] = useState<TrainingMessage[]>([]);
  const [connected, setConnected] = useState(false);
  const [latestState, setLatestState] = useState<NeuralState | null>(null);

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

      if (message.type === "batch_update") {
        const output = message.activations.output;
        const prediction = output.indexOf(Math.max(...output));
        setLatestState({
          input: [],
          layers: {
            hidden1: message.activations.hidden1,
            hidden2: message.activations.hidden2,
            output,
          },
          prediction,
          confidence: output[prediction] ?? 0,
          edges: {
            hidden1_hidden2: matrixToEdges(message.weights.hidden1_hidden2),
            hidden2_output: matrixToEdges(message.weights.hidden2_output),
          },
        });
      }
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

  return { messages, connected, send, latestState };
};
