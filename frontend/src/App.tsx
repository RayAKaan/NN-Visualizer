import React, { useCallback, useEffect, useMemo, useState } from "react";
import TrainingMode from "./components/training/TrainingMode";
import ModelsMode from "./components/models/ModelsMode";
import NeurofluxionLayout from "./components/neurofluxion/NeurofluxionLayout";
import { Brain, LineChart, AlertTriangle, CheckCircle2, Database, FlaskConical, Moon, Sun } from "lucide-react";
import { apiClient } from "./api/client";

export default function App() {
  const [mode, setMode] = useState<"train" | "models" | "lab">("lab");
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const [isBootstrapping, setIsBootstrapping] = useState(true);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [activeModel, setActiveModel] = useState<string | null>(null);
  const [startupError, setStartupError] = useState<string | null>(null);

  useEffect(() => {
    const stored = window.localStorage.getItem("nv-theme");
    if (stored === "light" || stored === "dark") {
      setTheme(stored);
    }
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

  const hasSavedModels = useMemo(() => availableModels.length > 0, [availableModels]);

  const onModelsChanged = useCallback(
    (available: string[], active: string | null) => {
      setAvailableModels((prev) => (prev.join("|") === available.join("|") ? prev : available));
      setActiveModel((prev) => (prev === active ? prev : active));
    },
    [],
  );

  return (
    <div className={`app-shell ${theme === "light" ? "theme-light" : "theme-dark"} flex flex-col h-screen bg-slate-950 text-slate-100 font-sans overflow-hidden`}>
      <header className="h-16 bg-slate-900 border-b border-slate-800 shrink-0">
        <div className="h-full px-4 md:px-6 max-w-[1800px] mx-auto flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 min-w-0">
            <div className="w-9 h-9 bg-cyan-600 rounded-lg flex items-center justify-center shrink-0">
              <Brain className="text-white" size={20} />
            </div>
            <h1 className="font-bold text-lg tracking-tight truncate">Neurofluxion</h1>
          </div>

          <div className="flex items-center gap-2 md:gap-3">
            <nav className="flex gap-1.5 md:gap-2 rounded-xl bg-slate-950/50 border border-slate-700 px-1.5 py-1">
              <button
                onClick={() => setMode("lab")}
                className={`px-3 md:px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2
                 ${mode === "lab" ? "bg-cyan-600/20 text-cyan-400 border border-cyan-600/50" : "text-slate-400 hover:text-white"}`}
                title="Open Neurofluxion lab"
              >
                <FlaskConical size={16} /> <span className="hidden sm:inline">Lab</span>
              </button>
              <button
                onClick={() => setMode("train")}
                className={`px-3 md:px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2
                 ${mode === "train" ? "bg-cyan-600/20 text-cyan-400 border border-cyan-600/50" : "text-slate-400 hover:text-white"}`}
              >
                <LineChart size={16} /> <span className="hidden sm:inline">Training</span>
              </button>
              <button
                onClick={() => setMode("models")}
                className={`px-3 md:px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2
                 ${mode === "models" ? "bg-cyan-600/20 text-cyan-400 border border-cyan-600/50" : "text-slate-400 hover:text-white"}`}
              >
                <Database size={16} /> <span className="hidden sm:inline">Models</span>
              </button>
            </nav>

            <button
              onClick={() => setTheme((prev) => (prev === "dark" ? "light" : "dark"))}
              className="h-10 px-3 rounded-xl border border-slate-700 bg-slate-950/60 text-slate-200 hover:text-cyan-300 hover:border-cyan-700 transition-colors flex items-center gap-2"
              title={`Switch to ${theme === "dark" ? "Light" : "Dark"} mode`}
              aria-label={`Switch to ${theme === "dark" ? "Light" : "Dark"} mode`}
            >
              {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
              <span className="hidden md:inline text-sm">{theme === "dark" ? "Light" : "Dark"}</span>
            </button>
          </div>
        </div>
      </header>

      <div
        className={`border-b text-sm ${hasSavedModels ? "bg-emerald-950/40 border-emerald-900 text-emerald-300" : "bg-amber-950/40 border-amber-900 text-amber-300"}`}
      >
        <div className="px-4 md:px-6 max-w-[1800px] mx-auto py-2 flex items-center gap-2">
          {hasSavedModels ? <CheckCircle2 size={16} /> : <AlertTriangle size={16} />}
          {isBootstrapping
            ? "Checking saved models..."
            : hasSavedModels
              ? `Saved models detected (${availableModels.join(", ")}). Active: ${activeModel ?? "n/a"}. Lab is set as your primary real-time workspace.`
              : "No saved models found yet. Use Lab or Training to train models in real time."}
        </div>
      </div>
      {startupError && (
        <div className="text-xs bg-rose-950/40 border-b border-rose-900 text-rose-300">
          <div className="px-4 md:px-6 max-w-[1800px] mx-auto py-2">{startupError}</div>
        </div>
      )}

      <main className="flex-1 overflow-auto">
        <div className={`h-full ${mode === "lab" ? "w-full" : "max-w-[1800px] mx-auto"}`}>
          {mode === "train" ? (
            <TrainingMode />
          ) : (
            mode === "models" ? <ModelsMode onModelsChanged={onModelsChanged} /> : <NeurofluxionLayout />
          )}
        </div>
      </main>
    </div>
  );
}
