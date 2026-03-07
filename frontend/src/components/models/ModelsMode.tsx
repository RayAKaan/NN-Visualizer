import React, { useCallback, useEffect, useState } from "react";
import { apiClient } from "../../api/client";
import { RefreshCcw, Trash2, CheckCircle2, Circle, RotateCw } from "lucide-react";

interface ModelEntry {
  model_type: string;
  path: string;
  exists_on_disk: boolean;
  loaded: boolean;
  active: boolean;
}

interface Props {
  onModelsChanged: (available: string[], active: string | null) => void;
}

export default function ModelsMode({ onModelsChanged }: Props) {
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [active, setActive] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiClient.get("/models/registry");
      const list: ModelEntry[] = Array.isArray(res.data?.models) ? res.data.models : [];
      const activeModel: string | null = typeof res.data?.active === "string" ? res.data.active : null;
      const available: string[] = Array.isArray(res.data?.available) ? res.data.available : [];
      setModels(list);
      setActive(activeModel);
      onModelsChanged(available, activeModel);
    } catch {
      setError("Failed to load model registry.");
    } finally {
      setLoading(false);
    }
  }, [onModelsChanged]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const switchModel = async (modelType: string) => {
    try {
      await apiClient.post("/model/switch", { model_type: modelType });
      await refresh();
    } catch {
      setError(`Could not switch active model to ${modelType}.`);
    }
  };

  const reloadModel = async (modelType: string) => {
    try {
      await apiClient.post(`/models/${modelType}/reload`);
      await refresh();
    } catch {
      setError(`Could not reload model ${modelType}.`);
    }
  };

  const deleteModel = async (modelType: string) => {
    const ok = window.confirm(`Delete saved model "${modelType}" from disk?`);
    if (!ok) return;
    try {
      await apiClient.delete(`/models/${modelType}`);
      await refresh();
    } catch {
      setError(`Could not delete model ${modelType}.`);
    }
  };

  return (
    <div className="h-full overflow-auto p-4 text-slate-100">
      <div className="max-w-6xl mx-auto space-y-4">
        <div className="flex items-center justify-between bg-slate-900 border border-slate-700 rounded-xl px-4 py-3">
          <div>
            <h2 className="text-xl font-bold text-cyan-300">Saved Models</h2>
            <p className="text-xs text-slate-400">Manage model files (switch, reload, delete).</p>
          </div>
          <button
            onClick={refresh}
            className="flex items-center gap-2 px-3 py-2 rounded-md bg-slate-800 border border-slate-600 hover:bg-slate-700"
          >
            <RefreshCcw size={14} />
            Refresh
          </button>
        </div>

        {error && (
          <div className="bg-rose-950/50 border border-rose-800 text-rose-300 text-sm rounded-lg p-3">{error}</div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
          {models.map((m) => (
            <div key={m.model_type} className="bg-slate-900 border border-slate-700 rounded-xl p-4 space-y-3">
              <div className="flex items-center justify-between">
                <div className="text-lg font-bold uppercase text-cyan-300">{m.model_type}</div>
                <div className="text-xs">
                  {m.active ? (
                    <span className="inline-flex items-center gap-1 text-emerald-300"><CheckCircle2 size={12} /> Active</span>
                  ) : (
                    <span className="inline-flex items-center gap-1 text-slate-500"><Circle size={12} /> Inactive</span>
                  )}
                </div>
              </div>

              <div className="text-xs text-slate-400 break-all">{m.path}</div>
              <div className="text-xs text-slate-300">
                Disk: <span className={m.exists_on_disk ? "text-emerald-300" : "text-rose-300"}>{m.exists_on_disk ? "Present" : "Missing"}</span>
                {" | "}
                Memory: <span className={m.loaded ? "text-emerald-300" : "text-slate-500"}>{m.loaded ? "Loaded" : "Not loaded"}</span>
              </div>

              <div className="flex gap-2">
                <button
                  disabled={!m.loaded || active === m.model_type}
                  onClick={() => void switchModel(m.model_type)}
                  className="flex-1 text-xs px-2 py-2 rounded bg-cyan-700 hover:bg-cyan-600 disabled:bg-slate-700 disabled:text-slate-400"
                >
                  Use
                </button>
                <button
                  disabled={!m.exists_on_disk}
                  onClick={() => void reloadModel(m.model_type)}
                  className="px-2 py-2 rounded bg-slate-800 hover:bg-slate-700 border border-slate-600 disabled:opacity-50"
                  title="Reload"
                >
                  <RotateCw size={14} />
                </button>
                <button
                  disabled={!m.exists_on_disk}
                  onClick={() => void deleteModel(m.model_type)}
                  className="px-2 py-2 rounded bg-rose-700 hover:bg-rose-600 disabled:bg-slate-700 disabled:text-slate-400"
                  title="Delete"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            </div>
          ))}
        </div>

        {loading && <div className="text-xs text-slate-400">Loading model registry...</div>}
      </div>
    </div>
  );
}
