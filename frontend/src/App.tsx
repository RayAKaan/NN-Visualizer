import React, { useCallback, useEffect, useMemo, useState } from "react";
import TrainingMode from "./components/training/TrainingMode";
import ModelsMode from "./components/models/ModelsMode";
import PredictionMode from "./components/prediction/PredictionMode";
import LabPage from "./pages/LabPage";
import SimulatorPage from "./pages/SimulatorPage";
import { apiClient } from "./api/client";
import { Activity, Brain, Command, Database, FlaskConical, LineChart, Search, ScanEye } from "lucide-react";
import { NeuralAmbient } from "./design-system/ambient/NeuralAmbient";
import { useLabStore } from "./store/labStore";
import { useSimulatorStore } from "./store/simulatorStore";
import { useComputationStore } from "./store/computationStore";

type AppMode = "predict" | "train" | "models" | "lab" | "simulator";

interface CommandItem {
  id: string;
  label: string;
  group: "Actions" | "Navigation" | "Models";
  run: () => void;
}

export default function App() {
  const [mode, setMode] = useState<AppMode>("lab");
  const [isBootstrapping, setIsBootstrapping] = useState(true);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [activeModel, setActiveModel] = useState<string | null>(null);
  const [startupError, setStartupError] = useState<string | null>(null);
  const [isPaletteOpen, setIsPaletteOpen] = useState(false);
  const [paletteQuery, setPaletteQuery] = useState("");
  const [paletteIndex, setPaletteIndex] = useState(0);
  const [bootstrapAttempt, setBootstrapAttempt] = useState(0);
  const simGraphId = useSimulatorStore((s) => s.graphId);

  useEffect(() => {
    let mounted = true;
    const bootstrap = async () => {
      setIsBootstrapping(true);
      setStartupError(null);
      // Tolerate transient backend downtime (e.g. server restarts) before
      // surfacing an error banner.
      const backoffs = [0, 1500, 3000, 5000];
      for (let attempt = 0; attempt < backoffs.length; attempt += 1) {
        if (backoffs[attempt] > 0) {
          await new Promise((resolve) => setTimeout(resolve, backoffs[attempt]));
        }
        if (!mounted) return;
        try {
          const res = await apiClient.get("/models/available");
          const models: string[] = Array.isArray(res.data?.available) ? res.data.available : [];
          const active: string | null = typeof res.data?.active === "string" ? res.data.active : null;
          if (!mounted) return;
          setAvailableModels(models);
          setActiveModel(active);
          setStartupError(null);
          return;
        } catch {
          /* retry */
        }
      }
      if (!mounted) return;
      setStartupError("Could not fetch model availability.");
    };
    void bootstrap().finally(() => {
      if (mounted) setIsBootstrapping(false);
    });
    return () => {
      mounted = false;
    };
  }, [bootstrapAttempt]);

  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === "k") {
        ev.preventDefault();
        setIsPaletteOpen((v) => !v);
      }
      if (ev.key === "Escape") setIsPaletteOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    setPaletteIndex(0);
  }, [paletteQuery]);

  const onModelsChanged = useCallback((available: string[], active: string | null) => {
    setAvailableModels((prev) => (prev.join("|") === available.join("|") ? prev : available));
    setActiveModel((prev) => (prev === active ? prev : active));
  }, []);

  const commands = useMemo<CommandItem[]>(
    () => [
      { id: "nav-predict", label: "Go to Prediction", group: "Navigation", run: () => setMode("predict") },
      { id: "nav-lab", label: "Go to Lab", group: "Navigation", run: () => setMode("lab") },
      { id: "nav-simulator", label: "Go to Simulator", group: "Navigation", run: () => setMode("simulator") },
      { id: "nav-train", label: "Go to Training", group: "Navigation", run: () => setMode("train") },
      { id: "nav-models", label: "Go to Models", group: "Navigation", run: () => setMode("models") },
      { id: "open-palette", label: "Open Command Palette", group: "Actions", run: () => setIsPaletteOpen(true) },
      { id: "refresh-models", label: "Refresh Model Registry", group: "Models", run: () => window.location.reload() },
    ],
    [],
  );

  const paletteResults = useMemo(() => {
    const q = paletteQuery.trim().toLowerCase();
    if (!q) return commands;
    return commands.filter((c) => c.label.toLowerCase().includes(q) || c.group.toLowerCase().includes(q));
  }, [commands, paletteQuery]);

  const runCommand = useCallback((item: CommandItem) => {
    item.run();
    setIsPaletteOpen(false);
    setPaletteQuery("");
  }, []);

  const handleLabReset = useCallback(() => {
    useLabStore.getState().resetPipeline();
  }, []);

  const handleSimForward = useCallback(async () => {
    const { graphId, currentInput } = useSimulatorStore.getState();
    if (!graphId) return;
    const input =
      currentInput && currentInput.length > 0
        ? currentInput
        : Array.from({ length: 16 }, () => Math.random() * 2 - 1);
    try {
      await useComputationStore.getState().runFullForward(input);
    } catch {
      /* forward failures surface in the simulator's own status UI */
    }
  }, []);

  const statusText = isBootstrapping
    ? "Bootstrapping"
    : availableModels.length === 0
      ? "No model"
      : mode === "train"
        ? "Training"
        : "Ready";

  const hasQuickAction = mode === "lab" || mode === "simulator";

  return (
    <div className="ncc-shell">
      {mode === "simulator" ? <NeuralAmbient /> : null}
      <aside className="ncc-rail" aria-label="Primary">
        <div className="ncc-rail-inner">
          <button className="ncc-logo" onClick={() => setMode("lab")} title="Neurofluxion" aria-label="Neurofluxion home">
            <Brain size={20} />
          </button>
          <div className="ncc-rail-links">
            <button className={`ncc-link ${mode === "predict" ? "active" : ""}`} onClick={() => setMode("predict")} aria-current={mode === "predict" ? "page" : undefined}><ScanEye size={18} /><span>Prediction</span></button>
            <button className={`ncc-link ${mode === "lab" ? "active" : ""}`} onClick={() => setMode("lab")} aria-current={mode === "lab" ? "page" : undefined}><FlaskConical size={18} /><span>Lab</span></button>
            <button className={`ncc-link ${mode === "simulator" ? "active" : ""}`} onClick={() => setMode("simulator")} aria-current={mode === "simulator" ? "page" : undefined}><Brain size={18} /><span>Simulator</span></button>
            <button className={`ncc-link ${mode === "train" ? "active" : ""}`} onClick={() => setMode("train")} aria-current={mode === "train" ? "page" : undefined}><LineChart size={18} /><span>Training</span></button>
            <button className={`ncc-link ${mode === "models" ? "active" : ""}`} onClick={() => setMode("models")} aria-current={mode === "models" ? "page" : undefined}><Database size={18} /><span>Models</span></button>
          </div>
          <div className="ncc-rail-footer">
            <button className="ncc-link ncc-link-danger" onClick={() => window.location.reload()} title="Reload application">
              <Activity size={18} />
              <span>Reload</span>
            </button>
          </div>
        </div>
      </aside>

      <main className={`ncc-content ${mode === "simulator" ? "ncc-content-sim" : ""}`}>
        {startupError && (
          <div className="ncc-banner danger" role="alert">
            <span>{startupError}</span>
            <button className="ncc-chip" onClick={() => setBootstrapAttempt((n) => n + 1)}>
              Retry
            </button>
          </div>
        )}
        <div key={mode} className="routing-fade">
          {mode === "predict"
            ? <PredictionMode />
            : mode === "train"
              ? <TrainingMode />
              : mode === "models"
                ? <ModelsMode onModelsChanged={onModelsChanged} />
                : mode === "simulator"
                  ? <SimulatorPage />
                  : <LabPage />}
        </div>
      </main>

      <section className="ncc-command-strip" aria-label="Status">
        <div className={`ncc-status-dot ${statusText === "Training" ? "training" : statusText === "Ready" ? "ready" : statusText === "No model" ? "off" : ""}`} aria-hidden="true" />
        <div className="ncc-model-info">
          <div className="ncc-model-title">{activeModel ?? "Dense_v3"}</div>
          <div className="ncc-model-meta">{statusText}</div>
        </div>
        <div className="ncc-sep" />
        {hasQuickAction ? (
          <>
            <div className="ncc-actions">
              {mode === "lab" && (
                <button className="ncc-chip" onClick={handleLabReset}>Reset</button>
              )}
              {mode === "simulator" && (
                <button
                  className="ncc-chip"
                  onClick={() => void handleSimForward()}
                  disabled={!simGraphId}
                  title={simGraphId ? "Run a full forward pass" : "Build a model first"}
                >
                  Forward
                </button>
              )}
            </div>
            <div className="ncc-sep" />
          </>
        ) : null}
        <button className="ncc-k-button" onClick={() => setIsPaletteOpen(true)}>
          <Search size={14} />
          <span>Command</span>
          <kbd>Ctrl+K</kbd>
        </button>
      </section>

      {isPaletteOpen && (
        <div className="ncc-palette-backdrop" onClick={() => setIsPaletteOpen(false)}>
          <div
            className="ncc-palette"
            role="dialog"
            aria-modal="true"
            aria-label="Command palette"
            onClick={(e) => e.stopPropagation()}
            onKeyDown={(e) => {
              if (e.key === "ArrowDown") {
                e.preventDefault();
                setPaletteIndex((i) => Math.min(i + 1, Math.max(0, paletteResults.length - 1)));
              } else if (e.key === "ArrowUp") {
                e.preventDefault();
                setPaletteIndex((i) => Math.max(i - 1, 0));
              } else if (e.key === "Enter") {
                const item = paletteResults[paletteIndex];
                if (item) runCommand(item);
              } else if (e.key === "Tab") {
                const nodes = Array.from(
                  e.currentTarget.querySelectorAll<HTMLElement>("input, .ncc-palette-item"),
                );
                if (nodes.length === 0) return;
                const first = nodes[0];
                const last = nodes[nodes.length - 1];
                if (e.shiftKey && document.activeElement === first) {
                  e.preventDefault();
                  last.focus();
                } else if (!e.shiftKey && document.activeElement === last) {
                  e.preventDefault();
                  first.focus();
                }
              }
            }}
          >
            <div className="ncc-palette-input-wrap">
              <Command size={16} />
              <input
                value={paletteQuery}
                onChange={(e) => setPaletteQuery(e.target.value)}
                placeholder="Search actions, navigation, models..."
                autoFocus
                role="combobox"
                aria-expanded="true"
                aria-controls="ncc-palette-listbox"
                aria-activedescendant={paletteResults[paletteIndex]?.id}
              />
            </div>
            <div className="ncc-palette-list" role="listbox" id="ncc-palette-listbox">
              {paletteResults.map((item, idx) => (
                <button
                  key={item.id}
                  id={item.id}
                  role="option"
                  aria-selected={idx === paletteIndex}
                  data-active={idx === paletteIndex}
                  className="ncc-palette-item"
                  onMouseEnter={() => setPaletteIndex(idx)}
                  onClick={() => runCommand(item)}
                  ref={(node) => {
                    if (node && idx === paletteIndex) {
                      node.scrollIntoView({ block: "nearest" });
                    }
                  }}
                >
                  <span>{item.label}</span>
                  <small>{item.group}</small>
                </button>
              ))}
              {paletteResults.length === 0 && <div className="ncc-empty">No matches</div>}
            </div>
          </div>
        </div>
      )}
      <nav className="ncc-mobile-tabs" aria-label="Primary mobile">
        <button className={mode === "predict" ? "active" : ""} onClick={() => setMode("predict")} aria-current={mode === "predict" ? "page" : undefined}><ScanEye size={16} />Predict</button>
        <button className={mode === "lab" ? "active" : ""} onClick={() => setMode("lab")} aria-current={mode === "lab" ? "page" : undefined}><FlaskConical size={16} />Lab</button>
        <button className={mode === "simulator" ? "active" : ""} onClick={() => setMode("simulator")} aria-current={mode === "simulator" ? "page" : undefined}><Brain size={16} />Sim</button>
        <button className={mode === "train" ? "active" : ""} onClick={() => setMode("train")} aria-current={mode === "train" ? "page" : undefined}><Activity size={16} />Train</button>
        <button className={mode === "models" ? "active" : ""} onClick={() => setMode("models")} aria-current={mode === "models" ? "page" : undefined}><Database size={16} />Models</button>
      </nav>
    </div>
  );
}
