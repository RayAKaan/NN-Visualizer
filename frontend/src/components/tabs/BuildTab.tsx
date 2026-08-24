import React from "react";
import { useSessionStore } from "../../store/sessionStore";
import { ArchitectureBuilder } from "../simulator/ArchitectureBuilder";
import { DatasetPanel } from "../simulator/DatasetPanel";
import { HyperparameterPanel } from "../simulator/HyperparameterPanel";
import { ImportExportPanel } from "../simulator/ImportExportPanel";
import { NeuralPanel } from "@/design-system/components/NeuralPanel";

export function BuildTab() {
  const userMode = useSessionStore((s) => s.userMode);
  
  return (
    <div className="tab-content build-tab">
      <div className="build-main">
        <div className="build-section">
          <NeuralPanel className="build-panel" variant="base">
            <h3 className="section-title">Architecture</h3>
            <ArchitectureBuilder />
          </NeuralPanel>
        </div>
        
        <div className="build-sidebar">
          <NeuralPanel className="dataset-panel" variant="base">
            <h3 className="section-title">Dataset</h3>
            <DatasetPanel />
          </NeuralPanel>
          
          <NeuralPanel className="hyperparams-panel" variant="base">
            <h3 className="section-title">Hyperparameters</h3>
            <HyperparameterPanel />
          </NeuralPanel>
          
          {userMode !== 'beginner' && (
            <NeuralPanel className="import-export-panel" variant="base">
              <h3 className="section-title">Import/Export</h3>
              <ImportExportPanel />
            </NeuralPanel>
          )}
        </div>
      </div>
    </div>
  );
}