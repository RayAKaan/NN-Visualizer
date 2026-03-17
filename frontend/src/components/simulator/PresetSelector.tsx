import React, { useEffect, useState } from "react";
import { useArchitectureStore } from "../../store/architectureStore";
import { simulatorApi } from "../../hooks/useSimulatorApi";
import { NeuralButton } from "@/design-system/components/NeuralButton";

const presets = ["Simple 2-4-1", "Deep 2-8-8-4-1", "Wide 2-16-1", "Simple CNN", "LSTM Classifier"];

interface TemplateItem {
  id: string;
  name: string;
  category: string;
  layers: any[];
}

export function PresetSelector() {
  const loadPreset = useArchitectureStore((s) => s.loadPreset);
  const setLayers = useArchitectureStore((s) => s.setLayers);
  const [templates, setTemplates] = useState<TemplateItem[]>([]);

  useEffect(() => {
    let active = true;
    simulatorApi.templatesList().then((data) => {
      if (active && data?.templates) setTemplates(data.templates);
    });
    return () => {
      active = false;
    };
  }, []);

  return (
    <div className="preset-section">
      <div>
        <div className="preset-title">Presets</div>
        <div className="preset-list">
          {presets.map((name) => (
            <NeuralButton
              key={name}
              onClick={() => loadPreset(name)}
              variant="secondary"
              className="preset-chip"
            >
              {name}
            </NeuralButton>
          ))}
        </div>
      </div>
      {templates.length ? (
        <div>
          <div className="preset-title">Templates</div>
          <div className="preset-list">
            {templates.map((t) => (
              <NeuralButton
                key={t.id}
                onClick={() => setLayers(t.layers as any, t.name)}
                variant="primary"
                className="preset-chip"
              >
                {t.name}
              </NeuralButton>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}
