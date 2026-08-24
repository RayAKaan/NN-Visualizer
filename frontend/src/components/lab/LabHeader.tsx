import type { Architecture, Dataset } from "../../types/pipeline";
import { PageHeader } from "@/design-system/components/PageHeader";
import { NeuralButton } from "@/design-system/components/NeuralButton";
import { useLabStore } from "../../store/labStore";

const ARCHES: Architecture[] = ["ANN", "CNN", "RNN"];
const DATASETS: Dataset[] = ["mnist", "catdog"];

export function LabHeader() {
  const architecture = useLabStore((s) => s.architecture);
  const dataset = useLabStore((s) => s.dataset);
  const isRunning = useLabStore((s) => s.isRunning);
  const setArchitecture = useLabStore((s) => s.setArchitecture);
  const setDataset = useLabStore((s) => s.setDataset);

  return (
    <header className="border-b border-barley-line bg-barley-page/85 backdrop-blur-md">
      <div className="page-shell [--shell-max:72rem] py-3">
        <PageHeader
          title="Neurofluxion Lab"
          subtitle="Follow one signal through every layer — forward, then back."
          actions={
            <>
              <div className="flex items-center gap-2" role="group" aria-label="Architecture">
                <span className="text-xs text-ink-faint">Arch</span>
                {ARCHES.map((arch) => (
                  <NeuralButton
                    key={arch}
                    size="sm"
                    variant={arch === architecture ? "primary" : "secondary"}
                    disabled={isRunning}
                    onClick={() => setArchitecture(arch)}
                  >
                    {arch}
                  </NeuralButton>
                ))}
              </div>
              <div className="flex items-center gap-2" role="group" aria-label="Dataset">
                <span className="text-xs text-ink-faint">Data</span>
                {DATASETS.map((item) => (
                  <NeuralButton
                    key={item}
                    size="sm"
                    variant={item === dataset ? "primary" : "secondary"}
                    disabled={isRunning}
                    onClick={() => setDataset(item)}
                  >
                    {item === "mnist" ? "MNIST" : "Cat/Dog"}
                  </NeuralButton>
                ))}
              </div>
            </>
          }
        />
      </div>
    </header>
  );
}
