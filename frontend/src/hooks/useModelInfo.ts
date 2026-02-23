import { useEffect, useState } from "react";
import { ModelInfo, ModelsAvailable, ModelType } from "../types";

export function useModelInfo() {
  const [info, setInfo] = useState<ModelInfo>({ model_type: "ann", loaded: false });
  const [available, setAvailable] = useState<ModelsAvailable>({ available: [], active: "ann" });

  const fetchInfo = async (type?: ModelType) => {
    const a = await fetch("http://localhost:8000/models/available").then(r => r.json());
    setAvailable(a);
    const i = await fetch(`http://localhost:8000/model/info${type ? `?type=${type}` : ""}`).then(r => r.json());
    setInfo(i);
  };
  const switchModel = async (model: ModelType) => {
    await fetch("http://localhost:8000/model/switch", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model_type: model }) });
    await fetchInfo(model);
  };
  useEffect(() => { fetchInfo().catch(() => null); }, []);
  return { info, available, switchModel, fetchInfo };
}
