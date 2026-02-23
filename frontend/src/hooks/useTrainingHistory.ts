import { useState, useCallback } from "react";
import { BatchUpdate } from "../types";

const API = "http://localhost:8000";

export function useTrainingHistory() {
  const [history, setHistory] = useState<BatchUpdate[]>([]);
  const [loading, setLoading] = useState(false);
  const [loaded, setLoaded] = useState(false);

  const fetchHistory = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API}/training/history`);
      if (res.ok) {
        const data = await res.json();
        setHistory(data);
        setLoaded(true);
      }
    } catch (e) {
      console.error("Failed to fetch training history", e);
    } finally {
      setLoading(false);
    }
  }, []);

  return { history, loading, loaded, fetchHistory };
}
