import { useCallback, useEffect, useState } from "react";
import { ModelType, ModelsAvailable } from "../types";

export function useModelSwitcher(apiBase: string) {
  const [activeModel, setActiveModel] = useState<ModelType>("ann");
  const [availableModels, setAvailableModels] = useState<ModelType[]>(["ann"]);
  const [switching, setSwitching] = useState(false);

  useEffect(() => {
    fetch(`${apiBase}/models/available`)
      .then((r) => r.json())
      .then((data: ModelsAvailable) => {
        setAvailableModels(data.available);
        setActiveModel(data.active);
      })
      .catch(() => {
        setAvailableModels(["ann"]);
        setActiveModel("ann");
      });
  }, [apiBase]);

  const switchModel = useCallback(
    async (type: ModelType) => {
      if (type === activeModel) return;
      setSwitching(true);
      try {
        const res = await fetch(`${apiBase}/model/switch`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model_type: type }),
        });
        const data = await res.json();
        setActiveModel(data.active_model);
        setAvailableModels(data.available_models ?? availableModels);
      } catch (e) {
        console.error("Failed to switch model", e);
      } finally {
        setSwitching(false);
      }
    },
    [activeModel, apiBase, availableModels]
  );

  return { activeModel, availableModels, switching, switchModel };
}
