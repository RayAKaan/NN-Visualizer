import React, { useCallback, useEffect, useMemo, useState } from "react";
import TrainingMode from "./components/training/TrainingMode";
import ModelsMode from "./components/models/ModelsMode";
import NeurofluxionLayout from "./components/neurofluxion/NeurofluxionLayout";
import PredictionMode from "./components/prediction/PredictionMode";
import { apiClient } from "./api/client";
import { Activity, Brain, Command, Database, FlaskConical, LineChart, Moon, Search, Settings, Sun, ScanEye } from "lucide-react";

type AppMode = "predict" | "train" | "models" | "lab";
type ThemeMode = "dark" | "light";

interface CommandItem {
  id: string;
  label: string;
  group: "Actions" | "Navigation" | "Models" | "Settings";
  run: () => void;
}

export default function App() {
  const [mode, setMode] = useState<AppMode>("lab");
  const [theme, setTheme] = useState<ThemeMode>("dark");
  const [isBootstrapping, setIsBootstrapping] = useState(true);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [activeModel, setActiveModel] = useState<string | null>(null);
  const [startupError, setStartupError] = useState<string | null>(null);
  const [isPaletteOpen, setIsPaletteOpen] = useState(false);
  const [paletteQuery, setPaletteQuery] = useState("");

  useEffect(() => {
    const stored = window.localStorage.getItem("nv-theme");
    if (stored === "light" || stored === "dark") setTheme(stored);
  }, []);

  useEffect(() => {
    window.localStorage.setItem("nv-theme", theme);
    document.documentElement.classList.toggle("light", theme === "light");
    document.documentElement.classList.toggle("dark", theme === "dark");
  }, [theme]);

  useEffect(() => {
    let mounted = true;
    const bootstrap = async () => {
      try {
        const res = await apiClient.get("/models/available");
        const models: string[] = Array.isArray(res.data?.available) ? res.data.available : [];
        const active: string | null = typeof res.data?.active === "string" ? res.data.active : null;
        if (!mounted) return;
        setAvailableModels(models);
        setActiveModel(active);
      } catch {
        if (!mounted) return;
        setStartupError("Could not fetch model availability.");
      } finally {
        if (mounted) setIsBootstrapping(false);
      }
    };
    void bootstrap();
    return () => {
      mounted = false;
    };
  }, []);

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

  const onModelsChanged = useCallback((available: string[], active: string | null) => {
    setAvailableModels((prev) => (prev.join("|") === available.join("|") ? prev : available));
    setActiveModel((prev) => (prev === active ? prev : active));
  }, []);

  const commands = useMemo<CommandItem[]>(
    () => [
      { id: "nav-predict", label: "Go to Prediction", group: "Navigation", run: () => setMode("predict") },
      { id: "nav-lab", label: "Go to Lab", group: "Navigation", run: () => setMode("lab") },
      { id: "nav-train", label: "Go to Training", group: "Navigation", run: () => setMode("train") },
      { id: "nav-models", label: "Go to Models", group: "Navigation", run: () => setMode("models") },
      { id: "theme", label: "Toggle Theme", group: "Settings", run: () => setTheme((t) => (t === "dark" ? "light" : "dark")) },
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

  const statusText = isBootstrapping
    ? "Bootstrapping"
    : availableModels.length === 0
      ? "No model"
      : mode === "train"
        ? "Training"
        : "Ready";

  const quickActions = useMemo(() => {
    if (mode === "predict") return ["Undo", "Clear", "Compare"];
    if (mode === "lab") return ["Play", "Step", "Reset"];
    if (mode === "train") return ["Start", "Stop", "Save"];
    return ["Import", "New"];
  }, [mode]);

  return (
    <div className={`ncc-shell ${theme === "light" ? "theme-light" : "theme-dark"}`}>
      <aside className="ncc-rail" aria-label="Neural Spine">
        <div className="ncc-rail-inner">
          <button className="ncc-logo" onClick={() => setMode("lab")} title="Neurofluxion">
            <Brain size={20} />
          </button>
          <div className="ncc-rail-links">
            <button className={`ncc-link ${mode === "predict" ? "active" : ""}`} onClick={() => setMode("predict")}><ScanEye size={18} /><span>Prediction</span></button>
            <button className={`ncc-link ${mode === "lab" ? "active" : ""}`} onClick={() => setMode("lab")}><FlaskConical size={18} /><span>Lab</span></button>
            <button className={`ncc-link ${mode === "train" ? "active" : ""}`} onClick={() => setMode("train")}><LineChart size={18} /><span>Training</span></button>
            <button className={`ncc-link ${mode === "models" ? "active" : ""}`} onClick={() => setMode("models")}><Database size={18} /><span>Models</span></button>
          </div>
          <div className="ncc-rail-footer">
            <button className="ncc-link"><Settings size={18} /><span>Settings</span></button>
            <button className="ncc-link" onClick={() => setTheme((prev) => (prev === "dark" ? "light" : "dark"))}>
              {theme === "dark" ? <Sun size={18} /> : <Moon size={18} />}
              <span>{theme === "dark" ? "Light" : "Dark"}</span>
            </button>
          </div>
        </div>
      </aside>

      <main className="ncc-content">
        {startupError && <div className="ncc-banner danger">{startupError}</div>}
        {mode === "predict"
          ? <PredictionMode />
          : mode === "train"
            ? <TrainingMode />
            : mode === "models"
              ? <ModelsMode onModelsChanged={onModelsChanged} />
              : <NeurofluxionLayout />}
      </main>

      <section className="ncc-command-strip" aria-label="Status Conduit">
        <div className={`ncc-status-dot ${statusText === "Training" ? "training" : statusText === "Ready" ? "ready" : statusText === "No model" ? "off" : ""}`} />
        <div className="ncc-model-info">
          <div className="ncc-model-title">{activeModel ?? "Dense_v3"}</div>
          <div className="ncc-model-meta">{statusText}</div>
        </div>
        <div className="ncc-sep" />
        <div className="ncc-actions">
          {quickActions.map((action) => (
            <button key={action} className="ncc-chip">{action}</button>
          ))}
        </div>
        <div className="ncc-sep" />
        <button className="ncc-k-button" onClick={() => setIsPaletteOpen(true)}>
          <Search size={14} />
          <span>Command</span>
          <kbd>Ctrl+K</kbd>
        </button>
      </section>

      {isPaletteOpen && (
        <div className="ncc-palette-backdrop" onClick={() => setIsPaletteOpen(false)}>
          <div className="ncc-palette" onClick={(e) => e.stopPropagation()}>
            <div className="ncc-palette-input-wrap">
              <Command size={16} />
              <input
                value={paletteQuery}
                onChange={(e) => setPaletteQuery(e.target.value)}
                placeholder="Search actions, navigation, models..."
                autoFocus
              />
            </div>
            <div className="ncc-palette-list">
              {paletteResults.map((item) => (
                <button
                  key={item.id}
                  className="ncc-palette-item"
                  onClick={() => {
                    item.run();
                    setIsPaletteOpen(false);
                    setPaletteQuery("");
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
      <div className="ncc-mobile-tabs">
        <button className={mode === "predict" ? "active" : ""} onClick={() => setMode("predict")}><ScanEye size={16} />Predict</button>
        <button className={mode === "lab" ? "active" : ""} onClick={() => setMode("lab")}><FlaskConical size={16} />Lab</button>
        <button className={mode === "train" ? "active" : ""} onClick={() => setMode("train")}><Activity size={16} />Train</button>
        <button className={mode === "models" ? "active" : ""} onClick={() => setMode("models")}><Database size={16} />Models</button>
      </div>
    </div>
  );
}
