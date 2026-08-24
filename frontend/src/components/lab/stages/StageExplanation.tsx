import type { LayerType } from "../../../types/pipeline";

const ICONS: Record<LayerType, string> = {
  input: "??",
  preprocessing: "??",
  dense: "??",
  conv2d: "??",
  activation_relu: "?",
  max_pool: "??",
  flatten: "??",
  softmax: "??",
  lstm_cell: "??",
  output: "??",
};

interface Props {
  explanation: string;
  layerType: LayerType;
}

export function StageExplanation({ explanation, layerType }: Props) {
  return (
    <div className="flex gap-3 rounded-xl border border-barley-linestrong bg-white p-3">
      <span className="text-xl">{ICONS[layerType]}</span>
      <div>
        <div className="text-[12px] font-semibold uppercase tracking-wide text-ember-700">What is happening</div>
        <p className="mt-1 text-sm leading-relaxed text-ink-soft">{explanation}</p>
      </div>
    </div>
  );
}
