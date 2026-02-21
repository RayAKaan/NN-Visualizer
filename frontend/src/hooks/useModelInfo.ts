import { useEffect, useState } from "react";
import { ModelInfo } from "../types";

export function useModelInfo(apiBase: string) {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);

  useEffect(() => {
    fetch(`${apiBase}/model/info`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data) {
          setModelInfo(data as ModelInfo);
        }
      })
      .catch(() => setModelInfo(null));
  }, [apiBase]);

  return modelInfo;
}
