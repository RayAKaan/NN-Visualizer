import { useMemo, useState } from "react";
import type { Dataset, StageActivation, StageDefinition } from "../../../../types/pipeline";

interface Props {
  activation: StageActivation;
  stage: StageDefinition;
  dataset: Dataset;
}

function toHeat(data: Float32Array, width: number, height: number): string {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";
  const img = ctx.createImageData(width, height);
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < data.length; i += 1) {
    if (data[i] < min) min = data[i];
    if (data[i] > max) max = data[i];
  }
  const range = max - min || 1;
  for (let i = 0; i < data.length; i += 1) {
    const v = (data[i] - min) / range;
    const o = i * 4;
    img.data[o] = Math.round(20 + 220 * v);
    img.data[o + 1] = Math.round(70 + 180 * v);
    img.data[o + 2] = Math.round(130 + 120 * v);
    img.data[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL();
}

export function ConvolutionViz({ activation, stage, dataset }: Props) {
  const [selected, setSelected] = useState(0);
  const [animate, setAnimate] = useState(false);
  const outShape = activation.metadata.outputShape;
  const numMaps = outShape[1] && outShape[2] ? outShape[0] : 1;
  const mapSize = outShape[1] && outShape[2] ? outShape[1] * outShape[2] : activation.outputData.length;

  const maps = useMemo(() => {
    const list: string[] = [];
    for (let i = 0; i < Math.min(numMaps, 32); i += 1) {
      const start = i * mapSize;
      const end = start + mapSize;
      list.push(toHeat(activation.outputData.slice(start, end), outShape[2] ?? 28, outShape[1] ?? 28));
    }
    return list;
  }, [activation.outputData, mapSize, numMaps, outShape]);

  return (
    <div className="space-y-3">
      <div className="text-xs text-ink-mute">{dataset === "catdog" ? "Cat/Dog" : "MNIST"} convolution response maps.</div>
      <button
        type="button"
        onClick={() => {
          setAnimate(false);
          requestAnimationFrame(() => setAnimate(true));
        }}
        className="rounded-lg border border-ember-600/40 bg-ember-600/15 px-2 py-1 text-xs text-ember-700"
      >
        Animate signal sweep
      </button>
      <div className="grid grid-cols-4 gap-1 sm:grid-cols-6 md:grid-cols-8">
        {maps.map((url, i) => (
          <button
            key={i}
            type="button"
            onClick={() => setSelected(i)}
            className="relative aspect-square overflow-hidden rounded-md border transition"
            style={{
              borderColor: selected === i ? "rgba(194,65,12,0.8)" : "rgba(28,25,23,0.15)",
              transform: selected === i ? "scale(1.06)" : "scale(1)",
            }}
          >
            <img src={url} alt={`Feature map ${i}`} className="h-full w-full object-cover" />
          </button>
        ))}
      </div>
      {maps[selected] && (
        <div className="relative h-44 overflow-hidden rounded-lg border border-arch-cnn/40 bg-barley-page p-2">
          <img src={maps[selected]} alt="Selected feature map" className="h-full w-full rounded object-contain" />
          {animate && <div className="lab-sweep" />}
          <div className="absolute right-2 top-2 rounded bg-ink/45 px-2 py-0.5 text-[12px] text-arch-cnn">Map {selected + 1}</div>
        </div>
      )}
      {activation.kernels && activation.kernels[selected] && (
        <div>
          <div className="mb-1 text-[12px] text-ink-faint">Kernel preview</div>
          <div className="grid w-fit grid-cols-3 gap-1 rounded border border-barley-linestrong bg-white p-2">
            {Array.from(activation.kernels[selected].slice(0, 9)).map((v, idx) => (
              <div
                key={idx}
                className="grid h-7 w-7 place-items-center rounded text-[12px] font-mono"
                style={{
                  background: v >= 0 ? `rgba(194,65,12,${Math.min(Math.abs(v), 1)})` : `rgba(0,114,178,${Math.min(Math.abs(v), 1)})`,
                }}
              >
                {v.toFixed(1)}
              </div>
            ))}
          </div>
        </div>
      )}
      <div className="text-[12px] text-ink-faint">Layer: {stage.name}</div>
    </div>
  );
}
