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
    <div className="flex gap-3 rounded-xl border border-slate-700/70 bg-slate-900/55 p-3">
      <span className="text-xl">{ICONS[layerType]}</span>
      <div>
        <div className="text-[11px] font-semibold uppercase tracking-wide text-cyan-300/70">What is happening</div>
        <p className="mt-1 text-sm leading-relaxed text-slate-300">{explanation}</p>
      </div>
    </div>
  );
}
