import { useLabStore } from "../../store/labStore";
import type { Architecture, Dataset } from "../../types/pipeline";
import { NeuralPanel } from "@/design-system/components/NeuralPanel";

const ARCHES: Architecture[] = ["ANN", "CNN", "RNN"];
const DATASETS: Dataset[] = ["mnist", "catdog"];

export function LabHeader() {
  const architecture = useLabStore((s) => s.architecture);
  const dataset = useLabStore((s) => s.dataset);
  const isRunning = useLabStore((s) => s.isRunning);
  const setArchitecture = useLabStore((s) => s.setArchitecture);
  const setDataset = useLabStore((s) => s.setDataset);

  return (
    <NeuralPanel variant="base" className="sticky top-0 z-30 flex h-14 items-center !rounded-none !border-l-0 !border-r-0 !border-t-0 px-4 shadow-sm">
      <div className="mx-auto flex h-full w-full max-w-6xl items-center justify-between gap-3">
        <div className="flex items-center gap-6">
          <div className="sim-title text-lg tracking-tight">Neurofluxion Lab</div>
          <div className="flex items-center gap-2">
            {ARCHES.map((arch) => {
              const active = arch === architecture;
              return (
                <button
                  key={arch}
                  type="button"
                  disabled={isRunning}
                  onClick={() => setArchitecture(arch)}
                  className={`neural-button !h-8 ${active ? "neural-button-primary" : "neural-button-secondary"}`}
                  style={{ opacity: isRunning ? 0.6 : 1 }}
                >
                  {arch}
                </button>
              );
            })}
          </div>
        </div>

        <div className="flex items-center gap-2 text-sm text-slate-300">
          <span className="opacity-75">Dataset</span>
          <div className="flex items-center gap-1">
            {DATASETS.map((item) => {
              const active = item === dataset;
              return (
                <button
                  key={item}
                  type="button"
                  disabled={isRunning}
                  onClick={() => setDataset(item)}
                  className={`neural-button !h-8 ${active ? "neural-button-primary" : "neural-button-secondary"}`}
                >
                  {item === "mnist" ? "MNIST" : "Cat/Dog"}
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </NeuralPanel>
  );
}
