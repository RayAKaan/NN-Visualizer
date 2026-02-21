import React from "react";
import { ModelType } from "../../types";

interface Props {
  active: ModelType;
  available: ModelType[];
  switching: boolean;
  onSwitch: (type: ModelType) => void;
}

const ModelSwitcher: React.FC<Props> = ({ active, available, switching, onSwitch }) => {
  const onlyOne = available.length <= 1;
  const hasCNN = available.includes("cnn");

  return (
    <div className="model-switcher" title={!hasCNN ? "Train CNN model first" : undefined}>
      <button
        className={`model-switcher-option ${active === "ann" ? "active" : ""} ${switching ? "switching" : ""}`}
        onClick={() => onSwitch("ann")}
        disabled={onlyOne && active !== "ann"}
      >
        ANN (Dense)
      </button>
      <button
        className={`model-switcher-option ${active === "cnn" ? "active" : ""} ${switching ? "switching" : ""}`}
        onClick={() => onSwitch("cnn")}
        disabled={!hasCNN}
      >
        CNN (Conv)
      </button>
    </div>
  );
};

export default ModelSwitcher;
