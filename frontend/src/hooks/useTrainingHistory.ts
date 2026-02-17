import { useEffect, useState } from "react";

export type TrainingHistorySnapshot = {
  epoch: number;
  batch: number;
  timestamp: number;
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
  weights: {
    hidden1_hidden2: number[][];
    hidden2_output: number[][];
  };
};

export function useTrainingHistory() {
  const [history, setHistory] = useState<TrainingHistorySnapshot[]>([]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    fetch("http://localhost:8000/training/history")
      .then((res) => res.json())
      .then((data) => {
        setHistory(Array.isArray(data) ? data : []);
        setLoaded(true);
      })
      .catch(() => {
        setHistory([]);
        setLoaded(true);
      });
  }, []);

  return { history, loaded };
}
