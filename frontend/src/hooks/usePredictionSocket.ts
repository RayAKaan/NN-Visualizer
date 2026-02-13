import { useEffect, useState } from "react";

export type PredictionPayload = {
  prediction: number;
  probabilities: number[];
  layers: { hidden1: number[]; hidden2: number[] };
  explanation?: unknown;
};

export const usePredictionSocket = (apiBase: string, pixels: number[], enabled: boolean) => {
  const [data, setData] = useState<PredictionPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!enabled) {
      return;
    }

    const timeout = setTimeout(async () => {
      try {
        const response = await fetch(`${apiBase}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ pixels }),
        });
        if (!response.ok) {
          throw new Error(await response.text());
        }
        const payload = (await response.json()) as PredictionPayload;
        setData(payload);
        setError(null);
      } catch {
        setError("Prediction request failed.");
      }
    }, 30);

    return () => clearTimeout(timeout);
  }, [apiBase, pixels, enabled]);

  return { data, error };
};
