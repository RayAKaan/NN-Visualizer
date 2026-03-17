import React, { useState } from "react";
import { simulatorApi } from "../../hooks/useSimulatorApi";
import { useSimulatorStore } from "../../store/simulatorStore";
import { NeuralPanel } from "@/design-system/components/NeuralPanel";
import { NeuralButton } from "@/design-system/components/NeuralButton";

export function ImportExportPanel() {
  const { graphId, setGraphId } = useSimulatorStore();
  const [file, setFile] = useState<File | null>(null);
  const [importStatus, setImportStatus] = useState<string | null>(null);
  const [importId, setImportId] = useState<string | null>(null);
  const [exportCode, setExportCode] = useState<string | null>(null);
  const [exportImage, setExportImage] = useState<string | null>(null);

  const handleUpload = async () => {
    if (!file) return;
    setImportStatus("Uploading...");
    try {
      const res = await simulatorApi.importUpload(file);
      setImportId(res.import_id);
      setImportStatus("Upload complete. Build to create graph.");
    } catch (err) {
      setImportStatus("Import failed.");
    }
  };

  const handleBuild = async () => {
    if (!importId) return;
    setImportStatus("Building graph...");
    try {
      const res = await simulatorApi.importBuild(importId);
      setGraphId(res.graph_id);
      setImportStatus("Graph ready.");
    } catch {
      setImportStatus("Build failed.");
    }
  };

  const handleExport = async (format: "json" | "pytorch" | "keras") => {
    if (!graphId) return;
    const res = await simulatorApi.exportCode(graphId, format);
    setExportCode(res.code);
    setExportImage(null);
  };

  const handleExportImage = async () => {
    if (!graphId) return;
    const res = await simulatorApi.exportImage(graphId, "svg");
    setExportImage(res.image_base64);
    setExportCode(null);
  };

  return (
    <NeuralPanel className="import-panel" variant="base">
      <div className="import-title">Import / Export</div>
      <div className="import-section">
        <div className="import-label">Import (JSON architecture)</div>
        <input className="import-file" type="file" accept=".json" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        <div className="import-actions">
          <NeuralButton variant="secondary" onClick={handleUpload}>
            Upload
          </NeuralButton>
          <NeuralButton variant="primary" onClick={handleBuild} disabled={!importId}>
            Build
          </NeuralButton>
        </div>
        {importStatus ? <div className="import-status">{importStatus}</div> : null}
      </div>
      <div className="import-section">
        <div className="import-label">Export</div>
        <div className="import-actions">
          <NeuralButton variant="secondary" onClick={() => handleExport("json")}>JSON</NeuralButton>
          <NeuralButton variant="secondary" onClick={() => handleExport("pytorch")}>PyTorch</NeuralButton>
          <NeuralButton variant="secondary" onClick={() => handleExport("keras")}>Keras</NeuralButton>
          <NeuralButton variant="secondary" onClick={handleExportImage}>SVG</NeuralButton>
        </div>
        {exportCode ? (
          <textarea className="import-code" value={exportCode} readOnly />
        ) : null}
        {exportImage ? (
          <img src={exportImage} className="import-image" />
        ) : null}
      </div>
    </NeuralPanel>
  );
}
