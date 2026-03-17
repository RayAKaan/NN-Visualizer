import { useMemo } from "react";
import type { Dataset } from "../../types/pipeline";

interface Props {
  rawPixels: Float32Array;
  imageUrl: string | null;
  dataset: Dataset;
}

function summarize(arr: Float32Array) {
  if (arr.length === 0) return { mean: 0, min: 0, max: 0, nnz: 0 };
  let sum = 0;
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let nnz = 0;
  for (let i = 0; i < arr.length; i += 1) {
    const v = arr[i];
    sum += v;
    if (v < min) min = v;
    if (v > max) max = v;
    if (v > 0.05) nnz += 1;
  }
  return { mean: sum / arr.length, min, max, nnz };
}

export function PreprocessingView({ rawPixels, imageUrl, dataset }: Props) {
  const stats = useMemo(() => summarize(rawPixels), [rawPixels]);

  return (
    <div className="rounded-xl border border-slate-700/70 bg-slate-950/40 p-4">
      <h3 className="text-sm font-semibold text-slate-100">Preprocessing Preview</h3>
      <p className="mt-1 text-xs text-slate-400">
        {dataset === "mnist"
          ? "Raw drawing is downscaled to 28x28 and normalized."
          : "Image is resized and converted into channel tensors."}
      </p>
      <div className="mt-3 grid gap-2 text-xs text-slate-300 sm:grid-cols-2">
        <div className="rounded-lg border border-slate-700 bg-slate-900/55 p-2">
          <div className="text-slate-500">Mean</div>
          <div className="font-mono">{stats.mean.toFixed(4)}</div>
        </div>
        <div className="rounded-lg border border-slate-700 bg-slate-900/55 p-2">
          <div className="text-slate-500">Range</div>
          <div className="font-mono">{stats.min.toFixed(3)} - {stats.max.toFixed(3)}</div>
        </div>
        <div className="rounded-lg border border-slate-700 bg-slate-900/55 p-2">
          <div className="text-slate-500">Non-zero</div>
          <div className="font-mono">{stats.nnz}</div>
        </div>
        <div className="rounded-lg border border-slate-700 bg-slate-900/55 p-2">
          <div className="text-slate-500">Vector size</div>
          <div className="font-mono">{rawPixels.length}</div>
        </div>
      </div>

      {imageUrl && (
        <div className="mt-3 rounded-lg border border-slate-700 bg-slate-900/45 p-2">
          <img src={imageUrl} alt="Input preview" className="max-h-[160px] rounded-md object-contain" />
        </div>
      )}
    </div>
  );
}
