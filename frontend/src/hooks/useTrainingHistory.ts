import { useState } from "react";

export function useTrainingHistory() {
  const [history, setHistory] = useState<unknown[]>([]);
  const [loading, setLoading] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const fetchHistory = async () => {
    setLoading(true);
    const data = await fetch("http://localhost:8000/training/history").then(r => r.json());
    setHistory(data); setLoaded(true); setLoading(false);
  };
  return { history, loading, loaded, fetchHistory };
}
