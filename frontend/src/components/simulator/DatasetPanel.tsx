import React, { useEffect, useState } from "react";
import { useDatasetStore } from "../../store/datasetStore";
import { DatasetCanvas } from "./DatasetCanvas";
import { NeuralButton } from "@/design-system/components/NeuralButton";
import { NeuralInput } from "@/design-system/components/NeuralInput";
import { NeuralSelect } from "@/design-system/components/NeuralSelect";
import { NeuralPanel } from "@/design-system/components/NeuralPanel";

const datasetOptions = ["circle", "xor", "spiral", "moons", "blobs", "linear", "rings"];
const standardOptions = ["mnist", "fashion_mnist", "cifar10"];
const sequenceOptions = ["sine_wave", "square_wave", "noise", "text_tokens"];

export function DatasetPanel() {
  const datasetType = useDatasetStore((s) => s.datasetType);
  const datasetCategory = useDatasetStore((s) => s.datasetCategory);
  const standardName = useDatasetStore((s) => s.standardName);
  const sequenceType = useDatasetStore((s) => s.sequenceType);
  const seqLength = useDatasetStore((s) => s.seqLength);
  const nFeatures = useDatasetStore((s) => s.nFeatures);
  const nClasses = useDatasetStore((s) => s.nClasses);
  const vocabSize = useDatasetStore((s) => s.vocabSize);
  const nSamples = useDatasetStore((s) => s.nSamples);
  const noise = useDatasetStore((s) => s.noise);
  const trainSplit = useDatasetStore((s) => s.trainSplit);
  const trainData = useDatasetStore((s) => s.trainData);
  const stats = useDatasetStore((s) => s.stats);
  const standardData = useDatasetStore((s) => s.standardData);
  const sequenceData = useDatasetStore((s) => s.sequenceData);
  const generate = useDatasetStore((s) => s.generate);
  const loadStandard = useDatasetStore((s) => s.loadStandard);
  const generateSequence = useDatasetStore((s) => s.generateSequence);
  const set = useDatasetStore.setState;

  const [status, setStatus] = useState<string | null>(null);

  useEffect(() => {
    if (datasetCategory === "synthetic") {
      void generate();
    }
  }, [datasetCategory, generate]);

  const handleStandard = async () => {
    setStatus("Loading dataset...");
    try {
      await loadStandard();
      setStatus(null);
    } catch {
      setStatus("Failed to load dataset. Check models_store files.");
    }
  };

  const handleSequence = async () => {
    setStatus("Generating sequence...");
    try {
      await generateSequence();
      setStatus(null);
    } catch {
      setStatus("Failed to generate sequence.");
    }
  };

  return (
    <NeuralPanel className="dataset-panel" variant="base">
      <div className="dataset-title">Dataset</div>
      <div className="dataset-fields">
        <div>
          <label className="dataset-label">Category</label>
          <NeuralSelect
            value={datasetCategory}
            onChange={(e) => set({ datasetCategory: e.target.value as any })}
            className="dataset-control"
          >
            <option value="synthetic">2D Synthetic</option>
            <option value="image">Image Classification</option>
            <option value="sequence">Sequence</option>
          </NeuralSelect>
        </div>

        {datasetCategory === "synthetic" && (
          <>
            <div>
              <label className="dataset-label">Type</label>
              <NeuralSelect
                value={datasetType}
                onChange={(e) => set({ datasetType: e.target.value })}
                className="dataset-control"
              >
                {datasetOptions.map((opt) => (
                  <option key={opt} value={opt}>{opt}</option>
                ))}
              </NeuralSelect>
            </div>
            <div className="dataset-grid">
              <label className="dataset-label">
                Samples
                <NeuralInput
                  type="number"
                  value={nSamples}
                  onChange={(e) => set({ nSamples: parseInt(e.target.value) || 200 })}
                  className="dataset-control"
                />
              </label>
              <label className="dataset-label">
                Noise
                <NeuralInput
                  type="number"
                  step="0.01"
                  value={noise}
                  onChange={(e) => set({ noise: parseFloat(e.target.value) || 0 })}
                  className="dataset-control"
                />
              </label>
            </div>
            <label className="dataset-label">
              Train Split
              <NeuralInput
                type="number"
                step="0.05"
                value={trainSplit}
                onChange={(e) => set({ trainSplit: parseFloat(e.target.value) || 0.8 })}
                className="dataset-control"
              />
            </label>
            <NeuralButton
              onClick={() => void generate()}
              className="dataset-button"
            >
              Regenerate
            </NeuralButton>
          </>
        )}

        {datasetCategory === "image" && (
          <>
            <div>
              <label className="dataset-label">Dataset</label>
              <NeuralSelect
                value={standardName}
                onChange={(e) => set({ standardName: e.target.value })}
                className="dataset-control"
              >
                {standardOptions.map((opt) => (
                  <option key={opt} value={opt}>{opt}</option>
                ))}
              </NeuralSelect>
            </div>
            <div className="dataset-grid">
              <label className="dataset-label">
                Samples
                <NeuralInput
                  type="number"
                  value={nSamples}
                  onChange={(e) => set({ nSamples: parseInt(e.target.value) || 500 })}
                  className="dataset-control"
                />
              </label>
              <label className="dataset-label">
                Train Split
                <NeuralInput
                  type="number"
                  step="0.05"
                  value={trainSplit}
                  onChange={(e) => set({ trainSplit: parseFloat(e.target.value) || 0.8 })}
                  className="dataset-control"
                />
              </label>
            </div>
            <NeuralButton onClick={handleStandard} className="dataset-button">
              Load Dataset
            </NeuralButton>
          </>
        )}

        {datasetCategory === "sequence" && (
          <>
            <div>
              <label className="dataset-label">Type</label>
              <NeuralSelect
                value={sequenceType}
                onChange={(e) => set({ sequenceType: e.target.value })}
                className="dataset-control"
              >
                {sequenceOptions.map((opt) => (
                  <option key={opt} value={opt}>{opt}</option>
                ))}
              </NeuralSelect>
            </div>
            <div className="dataset-grid">
              <label className="dataset-label">
                Seq Length
                <NeuralInput
                  type="number"
                  value={seqLength}
                  onChange={(e) => set({ seqLength: parseInt(e.target.value) || 20 })}
                  className="dataset-control"
                />
              </label>
              <label className="dataset-label">
                Features
                <NeuralInput
                  type="number"
                  value={nFeatures}
                  onChange={(e) => set({ nFeatures: parseInt(e.target.value) || 1 })}
                  className="dataset-control"
                />
              </label>
              <label className="dataset-label">
                Classes
                <NeuralInput
                  type="number"
                  value={nClasses}
                  onChange={(e) => set({ nClasses: parseInt(e.target.value) || 3 })}
                  className="dataset-control"
                />
              </label>
              <label className="dataset-label">
                Vocab Size
                <NeuralInput
                  type="number"
                  value={vocabSize}
                  onChange={(e) => set({ vocabSize: parseInt(e.target.value) || 50 })}
                  className="dataset-control"
                />
              </label>
            </div>
            <NeuralButton onClick={handleSequence} className="dataset-button">
              Generate Sequence
            </NeuralButton>
          </>
        )}
      </div>

      {status && <div className="dataset-status">{status}</div>}

      {datasetCategory === "synthetic" && <DatasetCanvas points={trainData} />}

      {datasetCategory === "image" && standardData?.train?.samples_preview && (
        <div className="dataset-preview-grid">
          {standardData.train.samples_preview.slice(0, 12).map((s: any) => (
            <div key={s.index} className="dataset-preview-card">
              <div className="dataset-preview-label">#{s.label}</div>
            </div>
          ))}
        </div>
      )}

      {datasetCategory === "sequence" && sequenceData && (
        <div className="dataset-status">
          Sequence dataset ready: {sequenceData.dataset_id}
        </div>
      )}

      {stats ? (
        <div className="dataset-status">
          Train: {stats.n_train} • Test: {stats.n_test}
        </div>
      ) : null}
    </NeuralPanel>
  );
}
