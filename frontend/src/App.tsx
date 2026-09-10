import React, { useCallback, useEffect, useMemo, useState } from "react";
import TrainingMode from "./components/training/TrainingMode";
import ModelsMode from "./components/models/ModelsMode";
import PredictionMode from "./components/prediction/PredictionMode";
import LabPage from "./pages/LabPage";
import SimulatorPage from "./pages/SimulatorPage";
import { apiClient } from "./api/client";
import { NeuralAmbient } from "./design-system/ambient/NeuralAmbient";
import { CommandPalette, type AppCommand } from "./components/layout/CommandPalette";
import {
  MobileWorkspaceHeader,
  MobileWorkspaceNav,
  WorkspaceNav,
  type BackendStatus,
  type WorkspaceId,
} from "./components/layout/WorkspaceNav";
import { WORKSPACE_EVENTS, emitWorkspaceEvent } from "./utils/workspaceEvents";
import type { CatalogModel } from "./types";
import { useSessionStore } from "./store/sessionStore";

export default function App() {
  const [mode, setMode] = useState<WorkspaceId>("predict");
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("checking");
  const [activeModel, setActiveModel] = useState<string | null>(null);
  const [pendingModelSelection, setPendingModelSelection] = useState<CatalogModel | null>(null);
  const [startupError, setStartupError] = useState<string | null>(null);
  const [isPaletteOpen, setIsPaletteOpen] = useState(false);
  const [paletteQuery, setPaletteQuery] = useState("");
  const [paletteIndex, setPaletteIndex] = useState(0);
  const [bootstrapAttempt, setBootstrapAttempt] = useState(0);

  const refreshApplicationState = useCallback(async () => {
    setBackendStatus("checking");
    setStartupError(null);
    try {
      const response = await apiClient.get("/models/available");
      const active = typeof response.data?.active === "string" ? response.data.active : null;
      setActiveModel(active);
      setBackendStatus("online");
    } catch {
      setBackendStatus("offline");
      setStartupError("The local backend is not responding. Workspaces remain available, but inference and execution are paused.");
    }
  }, []);

  useEffect(() => {
    void refreshApplicationState();
  }, [refreshApplicationState, bootstrapAttempt]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target?.closest("input, textarea, select, [contenteditable=\"true\"]")) return;
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setIsPaletteOpen((open) => !open);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    setPaletteIndex(0);
  }, [paletteQuery]);

  useEffect(() => {
    const workspaceTitles: Record<WorkspaceId, string> = {
      predict: "Prediction",
      lab: "Lab",
      simulator: "Simulator",
      train: "Training",
      models: "Models",
    };
    document.title = `${workspaceTitles[mode]} · Neurofluxion`;
  }, [mode]);

  useEffect(() => {
    if (mode !== "predict" || !pendingModelSelection) return;
    const timer = window.setTimeout(() => {
      window.dispatchEvent(new CustomEvent(WORKSPACE_EVENTS.selectModel, { detail: pendingModelSelection }));
      setPendingModelSelection(null);
    }, 0);
    return () => window.clearTimeout(timer);
  }, [mode, pendingModelSelection]);

  const onModelsChanged = useCallback((_: string[], nextActive: string | null) => {
    setBackendStatus("online");
    setActiveModel(nextActive);
    setStartupError(null);
  }, []);

  const navigate = useCallback((workspace: WorkspaceId) => {
    setMode(workspace);
    setIsPaletteOpen(false);
    setPaletteQuery("");
  }, []);

  const commands = useMemo<AppCommand[]>(() => {
    const navigationCommands: AppCommand[] = [
      { id: "nav-predict", label: "Go to Prediction", description: "Use a trained model", group: "Navigation", shortcut: "1", run: () => navigate("predict") },
      { id: "nav-lab", label: "Go to Lab", description: "Understand the computation", group: "Navigation", shortcut: "2", run: () => navigate("lab") },
      { id: "nav-simulator", label: "Go to Simulator", description: "Build and execute a network", group: "Navigation", shortcut: "3", run: () => navigate("simulator") },
      { id: "nav-training", label: "Go to Training", description: "Train and save a model", group: "Navigation", shortcut: "4", run: () => navigate("train") },
      { id: "nav-models", label: "Go to Models", description: "Discover and manage models", group: "Navigation", shortcut: "5", run: () => navigate("models") },
    ];

    const contextualCommands: AppCommand[] = [];
    if (mode === "predict") {
      contextualCommands.push({
        id: "predict-run",
        label: "Run prediction",
        description: "Use the current input and selected model",
        group: "Actions",
        shortcut: "Space",
        run: () => emitWorkspaceEvent(WORKSPACE_EVENTS.predict),
      });
    }
    if (mode === "lab") {
      contextualCommands.push({
        id: "lab-reset",
        label: "Reset Lab pipeline",
        description: "Return the forward pass to its starting point",
        group: "Actions",
        run: () => emitWorkspaceEvent(WORKSPACE_EVENTS.resetLab),
      });
    }
    if (mode === "simulator") {
      contextualCommands.push({
        id: "simulator-forward",
        label: "Run forward pass",
        description: "Execute the current simulator network",
        group: "Actions",
        run: () => {
          useSessionStore.getState().setActiveTab("run");
          window.setTimeout(() => emitWorkspaceEvent(WORKSPACE_EVENTS.runForward), 0);
        },
      });
    }
    if (mode === "train") {
      contextualCommands.push({
        id: "training-start",
        label: "Start training",
        description: "Start a run with the current configuration",
        group: "Actions",
        run: () => emitWorkspaceEvent(WORKSPACE_EVENTS.startTraining),
      });
    }
    if (mode === "models") {
      contextualCommands.push({
        id: "models-refresh",
        label: "Refresh model registry",
        description: "Fetch current availability and load state",
        group: "Models",
        shortcut: "R",
        run: () => emitWorkspaceEvent(WORKSPACE_EVENTS.refreshModels),
      });
    }

    return [
      ...navigationCommands,
      ...contextualCommands,
      {
        id: "refresh-application",
        label: "Reload application",
        description: "Restart the local interface",
        group: "Utility",
        run: () => window.location.reload(),
      },
    ];
  }, [mode, navigate]);

  const paletteResults = useMemo(() => {
    const query = paletteQuery.trim().toLowerCase();
    if (!query) return commands;
    return commands.filter((command) =>
      [command.label, command.description, command.group].filter(Boolean).some((value) => value!.toLowerCase().includes(query)),
    );
  }, [commands, paletteQuery]);

  const runCommand = useCallback((command: AppCommand) => {
    command.run();
    setIsPaletteOpen(false);
    setPaletteQuery("");
  }, []);

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">Skip to workspace</a>
      {mode === "simulator" ? <NeuralAmbient /> : null}
      <WorkspaceNav
        active={mode}
        onNavigate={navigate}
        backendStatus={backendStatus}
        activeModel={activeModel}
        onOpenCommand={() => setIsPaletteOpen(true)}
        onReload={() => window.location.reload()}
      />
      <MobileWorkspaceHeader
        active={mode}
        backendStatus={backendStatus}
        onOpenCommand={() => setIsPaletteOpen(true)}
      />

      <main id="main-content" className="workspace-main" aria-busy={backendStatus === "checking"}>
        {startupError ? (
          <div className="workspace-alert" role="alert">
            <div>
              <strong>Backend unavailable</strong>
              <span>{startupError}</span>
            </div>
            <button type="button" className="workspace-alert-action" onClick={() => setBootstrapAttempt((attempt) => attempt + 1)}>
              Try again
            </button>
          </div>
        ) : null}
        <div key={mode} className="workspace-route routing-fade">
          {mode === "predict"
            ? <PredictionMode />
            : mode === "train"
              ? <TrainingMode />
              : mode === "models"
                ? (
                  <ModelsMode
                    onModelsChanged={onModelsChanged}
                    onUseModel={(model: CatalogModel) => {
                      setActiveModel(model.id);
                      setPendingModelSelection(model);
                      navigate("predict");
                    }}
                  />
                )
                : mode === "simulator"
                  ? <SimulatorPage />
                  : <LabPage />}
        </div>
      </main>

      <MobileWorkspaceNav active={mode} onNavigate={navigate} />

      <CommandPalette
        open={isPaletteOpen}
        query={paletteQuery}
        commands={paletteResults}
        activeIndex={paletteIndex}
        onQueryChange={setPaletteQuery}
        onActiveIndexChange={setPaletteIndex}
        onRun={runCommand}
        onClose={() => setIsPaletteOpen(false)}
      />

      {/* Context actions are intentionally not global chrome; commands invoke the same real handlers. */}
      <span className="sr-only" aria-live="polite">
        {backendStatus === "online" ? "Backend ready" : backendStatus === "offline" ? "Backend unavailable" : "Connecting to backend"}
      </span>
    </div>
  );
}
