import { useLabStore } from "../../store/labStore";
import { DrawingCanvas } from "./DrawingCanvas";
import { ImageSelector } from "./ImageSelector";
import { PreprocessingView } from "./PreprocessingView";
import { NeuralPanel } from "@/design-system/components/NeuralPanel";

export function InputStage() {
  const dataset = useLabStore((s) => s.dataset);
  const inputPixels = useLabStore((s) => s.inputPixels);
  const inputImageUrl = useLabStore((s) => s.inputImageUrl);

  return (
    <NeuralPanel className="my-5 p-4 mx-auto max-w-[1000px]">
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="grid h-9 w-9 place-items-center rounded-lg bg-[rgba(234,88,12,0.10)] shadow-[0_1px_6px_rgba(194,65,12,0.25)] text-sm font-bold text-[#9A3412] border border-[rgba(194,65,12,0.35)]">IN</div>
          <div>
            <h2 className="text-lg font-semibold tracking-tight text-ink">Input Stage</h2>
            <div className="text-xs uppercase tracking-wider text-ink-faint">{dataset === "mnist" ? "Draw a digit" : "Select an image"}</div>
          </div>
        </div>
      </div>
      <div className="grid gap-6 lg:grid-cols-[minmax(240px,320px)_minmax(0,1fr)]">
        <div>{dataset === "mnist" ? <DrawingCanvas /> : <ImageSelector />}</div>
        <div className="neural-panel-sunken rounded-xl p-4 border border-barley-line bg-barley-sunken">
          <PreprocessingView rawPixels={inputPixels} imageUrl={inputImageUrl} dataset={dataset} />
        </div>
      </div>
    </NeuralPanel>
  );
}
