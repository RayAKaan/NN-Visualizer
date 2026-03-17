import type { Architecture, Dataset, StageDefinition } from "../../../../types/pipeline";
import { getExplanation } from "../../../../data/explanations";

type Level = "simple" | "technical" | "mathematical";

interface Props {
  stage: StageDefinition;
  architecture: Architecture;
  dataset: Dataset;
  level: Level;
  onLevelChange: (level: Level) => void;
}

export function WhatIsThis({ stage, architecture, dataset, level, onLevelChange }: Props) {
  const explanation = getExplanation(stage.type, architecture, dataset, level);

  return (
    <section className="overflow-hidden rounded-xl border" style={{ borderColor: "var(--glass-border)", background: "var(--bg-panel)" }}>
      <div className="flex border-b" style={{ borderColor: "var(--glass-border)" }}>
        {([
          { key: "simple" as Level, label: "Simple" },
          { key: "technical" as Level, label: "Technical" },
          { key: "mathematical" as Level, label: "Math" },
        ]).map(({ key, label }) => (
          <button
            key={key}
            type="button"
            onClick={() => onLevelChange(key)}
            className="relative flex-1 py-2 text-xs font-medium"
            style={{ color: level === key ? "var(--fwd)" : "var(--text-4)", background: level === key ? "var(--fwd-bg)" : "transparent" }}
          >
            {label}
            {level === key ? <div className="absolute bottom-0 left-0 right-0 h-0.5" style={{ background: "var(--fwd)" }} /> : null}
          </button>
        ))}
      </div>
      <div className="space-y-2 p-4">
        <div className="flex items-start gap-2">
          <span className="text-xl">{explanation.icon}</span>
          <div>
            <h4 className="text-sm font-semibold" style={{ color: "var(--text-1)" }}>{explanation.title}</h4>
            {explanation.analogy ? <p className="text-xs italic" style={{ color: "var(--fwd)" }}>Analogy: {explanation.analogy}</p> : null}
          </div>
        </div>
        {explanation.paragraphs.map((p, i) => (
          <p key={i} className="text-sm leading-relaxed" style={{ color: "var(--text-2)" }}>{p}</p>
        ))}
      </div>
    </section>
  );
}
