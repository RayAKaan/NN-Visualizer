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
          <div className="grid h-9 w-9 place-items-center rounded-lg bg-[rgba(34,211,238,0.15)] shadow-[0_0_12px_rgba(34,211,238,0.3)] text-sm font-bold text-[#67e8f9] border border-[rgba(34,211,238,0.3)]">IN</div>
          <div>
            <h2 className="text-lg font-semibold tracking-tight text-[#e8ecf8]">Input Stage</h2>
            <div className="text-xs uppercase tracking-wider text-[#9ba3c2]">{dataset === "mnist" ? "Draw a digit" : "Select an image"}</div>
          </div>
        </div>
      </div>
      <div className="grid gap-6 lg:grid-cols-[300px_minmax(0,1fr)]">
        <div>{dataset === "mnist" ? <DrawingCanvas /> : <ImageSelector />}</div>
        <div className="neural-panel-sunken rounded-xl p-4 border border-[rgba(36,40,54,0.3)] bg-[rgba(8,9,13,0.5)]">
          <PreprocessingView rawPixels={inputPixels} imageUrl={inputImageUrl} dataset={dataset} />
        </div>
      </div>
    </NeuralPanel>
  );
}
