import React, { useCallback, useEffect, useState } from "react";
import axios from "axios";
import { apiClient } from "../../api/client";
import { RefreshCcw, Trash2, CheckCircle2, Circle, RotateCw, Database } from "lucide-react";
import { PageHeader } from "@/design-system/components/PageHeader";
import { NeuralButton } from "@/design-system/components/NeuralButton";

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

const ARCH_ACCENT: Record<string, { dot: string; bar: string }> = {
  ann: { dot: "bg-arch-ann", bar: "border-arch-ann" },
  cnn: { dot: "bg-arch-cnn", bar: "border-arch-cnn" },
  rnn: { dot: "bg-arch-rnn", bar: "border-arch-rnn" },
};

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
    } catch (e) {
      if (axios.isAxiosError(e)) {
        setError(
          e.response
            ? `Registry request failed (${e.response.status}).`
            : "Backend unreachable — is the server running on http://localhost:8000?"
        );
      } else {
        setError("Failed to load model registry.");
      }
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
    <div className="min-h-full text-ink">
      <div className="page-shell [--shell-max:75rem] py-6 space-y-5">
        <PageHeader
          title="Models"
          subtitle="Manage saved weights — switch, reload, or remove."
          actions={
            <NeuralButton size="sm" onClick={refresh} disabled={loading}>
              <RefreshCcw size={14} className={loading ? "animate-spin" : undefined} />
              Refresh
            </NeuralButton>
          }
        />

        {error && (
          <div role="alert" className="flex items-center justify-between gap-3 rounded-lg border border-status-danger/35 bg-status-danger/10 px-3 py-2 text-sm text-status-dangerhover">
            <span>{error}</span>
            <NeuralButton size="sm" onClick={() => void refresh()}>
              <RotateCw size={13} /> Retry
            </NeuralButton>
          </div>
        )}

        {loading && models.length === 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {[0, 1, 2].map((i) => (
              <div key={i} className="bg-white border border-barley-linestrong rounded-xl p-4 space-y-3 shadow-card">
                <div className="neural-skeleton h-5 w-24" />
                <div className="neural-skeleton h-3 w-full" />
                <div className="neural-skeleton h-8 w-full" />
              </div>
            ))}
          </div>
        ) : models.length === 0 && !loading && !error ? (
          <div className="empty-state rounded-xl border border-dashed border-barley-linestrong bg-white py-12">
            <Database size={28} className="text-ink-faint" aria-hidden="true" />
            <p className="mt-2 text-sm font-medium text-ink-soft">No models in the registry</p>
            <p className="text-xs text-ink-faint">Train a model to see it appear here.</p>
            <NeuralButton size="sm" variant="primary" className="mt-4" onClick={() => void refresh()}>
              <RotateCw size={14} /> Refresh registry
            </NeuralButton>
          </div>
        ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {models.map((m) => {
            const accent = ARCH_ACCENT[m.model_type] ?? { dot: "bg-ink-faint", bar: "border-ink-faint" };
            return (
              <article
                key={m.model_type}
                className={`relative bg-white border border-barley-linestrong border-l-4 ${accent.bar} rounded-xl p-4 space-y-3 shadow-card transition-shadow hover:shadow-md`}
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2 min-w-0">
                    <span className={`h-2.5 w-2.5 rounded-full flex-none ${accent.dot}`} aria-hidden="true" />
                    <h2 className="text-base font-bold uppercase tracking-wide text-ink truncate">{m.model_type}</h2>
                  </div>
                  {m.active ? (
                    <span className="inline-flex flex-none items-center gap-1 rounded-full bg-status-success/10 px-2 py-0.5 text-[11px] font-semibold text-status-success">
                      <CheckCircle2 size={12} /> Active
                    </span>
                  ) : (
                    <span className="inline-flex flex-none items-center gap-1 rounded-full bg-barley-sunken px-2 py-0.5 text-[11px] font-medium text-ink-faint">
                      <Circle size={12} /> Inactive
                    </span>
                  )}
                </div>

                <div className="rounded-md bg-barley-sunken px-2 py-1.5 font-mono text-[11px] leading-snug text-ink-mute break-all" title={m.path}>
                  {m.path}
                </div>

                <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-ink-soft">
                  <span className="inline-flex items-center gap-1.5">
                    <span className={`h-1.5 w-1.5 rounded-full ${m.exists_on_disk ? "bg-status-success" : "bg-status-danger"}`} aria-hidden="true" />
                    Disk: <span className={m.exists_on_disk ? "font-medium text-status-success" : "font-medium text-status-danger"}>{m.exists_on_disk ? "Present" : "Missing"}</span>
                  </span>
                  <span className="inline-flex items-center gap-1.5">
                    <span className={`h-1.5 w-1.5 rounded-full ${m.loaded ? "bg-status-success" : "bg-barley-linestrong"}`} aria-hidden="true" />
                    Memory: <span className={m.loaded ? "font-medium text-status-success" : "font-medium text-ink-faint"}>{m.loaded ? "Loaded" : "Not loaded"}</span>
                  </span>
                </div>

                <div className="flex gap-2 pt-1">
                  <button
                    disabled={!m.loaded || active === m.model_type}
                    onClick={() => void switchModel(m.model_type)}
                    className="flex-1 h-9 text-xs font-semibold px-2 rounded-md bg-ember-700 hover:bg-ember-800 active:scale-[0.98] text-white transition-colors disabled:bg-barley-sunken disabled:text-ink-faint disabled:cursor-not-allowed"
                  >
                    {active === m.model_type ? "In use" : "Use"}
                  </button>
                  <button
                    disabled={!m.exists_on_disk}
                    onClick={() => void reloadModel(m.model_type)}
                    className="inline-flex h-9 w-10 items-center justify-center rounded-md bg-barley-wash hover:bg-barley-sunken border border-barley-linestrong transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-ink-soft"
                    title="Reload from disk"
                    aria-label={`Reload ${m.model_type} from disk`}
                  >
                    <RotateCw size={14} />
                  </button>
                  <button
                    disabled={!m.exists_on_disk}
                    onClick={() => void deleteModel(m.model_type)}
                    className="inline-flex h-9 w-10 items-center justify-center rounded-md border border-status-danger/35 bg-status-danger/5 hover:bg-status-danger/15 active:scale-[0.98] text-status-danger transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    title="Delete from disk"
                    aria-label={`Delete ${m.model_type} from disk`}
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              </article>
            );
          })}
        </div>
        )}
      </div>
    </div>
  );
}
