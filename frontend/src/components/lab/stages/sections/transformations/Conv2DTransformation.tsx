import { useMemo, useState } from "react";
import type { Dataset, StageActivation, StageDefinition } from "../../../../../types/pipeline";
import { ACTIVATION_RAMP_LIGHT, renderHeatmap } from "../../../../../utils/colorRamps";

interface Props {
  stage: StageDefinition;
  activation: StageActivation;
  dataset: Dataset;
}

export function Conv2DTransformation({ stage, activation }: Props) {
  const [selected, setSelected] = useState(0);
  const shapeIn = activation.metadata.inputShape;
  const shapeOut = activation.metadata.outputShape;
  const inH = shapeIn[shapeIn.length - 2] ?? 28;
  const inW = shapeIn[shapeIn.length - 1] ?? 28;
  const outF = shapeOut[0] ?? Number(stage.params?.filters ?? 1);
  const outH = shapeOut[shapeOut.length - 2] ?? 26;
  const outW = shapeOut[shapeOut.length - 1] ?? 26;

  const inputUrl = useMemo(() => renderHeatmap(activation.inputData.slice(0, inH * inW), inW, inH, ACTIVATION_RAMP_LIGHT), [activation.inputData, inH, inW]);
  const mapUrls = useMemo(() => {
    const size = outH * outW;
    const urls: string[] = [];
    const count = Math.min(outF, Math.floor(activation.outputData.length / size));
    for (let f = 0; f < count; f += 1) {
      urls.push(renderHeatmap(activation.outputData.slice(f * size, (f + 1) * size), outW, outH, ACTIVATION_RAMP_LIGHT));
    }
    return urls;
  }, [activation.outputData, outF, outH, outW]);

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-start justify-center gap-4">
        <div>
          <div className="mb-1 text-[12px] text-ink-faint">
            Input [{inH}
            {"\u00d7"}
            {inW}]
          </div>
          <img src={inputUrl} alt="input" className="h-28 w-28 rounded border border-barley-linestrong object-cover" style={{ imageRendering: "pixelated" }} />
        </div>
        <div className="pt-10 text-xl text-arch-cnn">&rarr;</div>
        <div>
          <div className="mb-1 text-[12px] text-ink-faint">Feature Map #{selected + 1}</div>
          {mapUrls[selected] ? <img src={mapUrls[selected]} alt="map" className="h-28 w-28 rounded object-cover" style={{ border: "1px solid rgba(0,128,106,0.4)", imageRendering: "pixelated" }} /> : null}
        </div>
      </div>
      <div className="grid gap-1" style={{ gridTemplateColumns: "repeat(auto-fill,minmax(36px,1fr))" }}>
        {mapUrls.map((u, i) => (
          <button
            key={i}
            type="button"
            onClick={() => setSelected(i)}
            className={`overflow-hidden rounded border ${i === selected ? "border-arch-cnn" : "border-barley-linestrong"}`}
            aria-label={`Select feature map ${i + 1}`}
          >
            <img src={u} alt={`fm-${i}`} className="h-9 w-full object-cover" style={{ imageRendering: "pixelated" }} />
          </button>
        ))}
      </div>
    </div>
  );
}
