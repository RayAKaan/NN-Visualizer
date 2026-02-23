import { useEffect, useState } from "react";

export function useHealthCheck() {
  const [backendOnline, setBackendOnline] = useState(false);
  const [retryCount, setRetryCount] = useState(0);
  const check = async () => {
    try {
      const res = await fetch("http://localhost:8000/health");
      setBackendOnline(res.ok);
      if (res.ok) setRetryCount(0); else setRetryCount((x) => x + 1);
    } catch {
      setBackendOnline(false);
      setRetryCount((x) => x + 1);
    }
  };
  useEffect(() => { check(); const t = setInterval(check, 5000); return () => clearInterval(t); }, []);
  return { backendOnline, retryCount, check };
}
