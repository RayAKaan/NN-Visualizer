import React, { useCallback, useEffect, useMemo, useState } from "react";
import axios from "axios";
import { apiClient } from "../../api/client";
import {
  RefreshCcw,
  Trash2,
  CheckCircle2,
  Circle,
  RotateCw,
  Database,
  Download,
  Upload,
  Box,
  FileWarning,
  Cpu,
  Layers3,
  Hash,
  Braces,
  PenLine,
  Image as ImageIcon,
  Type,
  Search,
  ChevronDown,
  Check,
} from "lucide-react";
import { PageHeader } from "@/design-system/components/PageHeader";
import { NeuralButton } from "@/design-system/components/NeuralButton";
import type { CatalogModel, ModelFamily } from "../../types";
import { WORKSPACE_EVENTS } from "../../utils/workspaceEvents";

interface ModelEntry {
  model_type: string;
  path: string;
  exists_on_disk: boolean;
  loaded: boolean;
  active: boolean;
}

interface Props {
  onModelsChanged: (available: string[], active: string | null) => void;
  onUseModel?: (model: CatalogModel) => void;
}

const FAMILY_COLOR: Record<ModelFamily, string> = {
  ANN: "#0072B2",
  CNN: "#00806A",
  RNN: "#A64D85",
};

const FAMILY_INPUT_HINT: Record<string, { icon: React.ReactNode; label: string }> = {
  mnist_pixels: { icon: <PenLine size={13} />, label: "Draw a digit (28×28)" },
  image: { icon: <ImageIcon size={13} />, label: "Upload an image (224×224)" },
  text: { icon: <Type size={13} />, label: "Type a review / text" },
};

const STATUS_META: Record<
  string,
  { label: string; cls: string; dot: string }
> = {
  available: { label: "Available", cls: "bg-ember-700/10 text-ember-800", dot: "bg-amber-500" },
  loading: { label: "Loading…", cls: "bg-sky-700/10 text-sky-800", dot: "bg-sky-500 animate-pulse" },
  loaded: { label: "Loaded in memory", cls: "bg-status-success/10 text-status-success", dot: "bg-status-success" },
  unavailable: { label: "Unavailable", cls: "bg-status-danger/10 text-status-danger", dot: "bg-status-danger" },
  error: { label: "Load error", cls: "bg-status-danger/10 text-status-danger", dot: "bg-status-danger" },
};

const fmtParams = (n: number | null | undefined) =>
  n == null ? "—" : n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(0)}k` : `${n}`;

export default function ModelsMode({ onModelsChanged, onUseModel }: Props) {
  // --- locally-trained (legacy) models -----------------------------------
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [active, setActive] = useState<string | null>(null);
  // --- pretrained registry catalog ----------------------------------------
  const [catalog, setCatalog] = useState<CatalogModel[]>([]);
  const [busyIds, setBusyIds] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [familyFilter, setFamilyFilter] = useState<"all" | ModelFamily>("all");
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());

  const fetchCatalog = useCallback(async () => {
    const res = await apiClient.get("/models/catalog");
    const list: CatalogModel[] = Array.isArray(res.data?.models) ? res.data.models : [];
    const loadedIds = new Set<string>(Array.isArray(res.data?.loaded) ? res.data.loaded : []);
    const withLoaded = list.map((m) =>
      loadedIds.has(m.id) && m.status !== "loaded" ? { ...m, status: "loaded" as const } : m,
    );
    setCatalog(withLoaded);
    return withLoaded;
  }, []);

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
      await fetchCatalog();
    } catch (e) {
      if (axios.isAxiosError(e)) {
        setError(
          e.response
            ? `Registry request failed (${e.response.status}).`
            : "Backend unreachable — start the local Neurofluxion server and try again.",
        );
      } else {
        setError("Failed to load model registry.");
      }
    } finally {
      setLoading(false);
    }
  }, [onModelsChanged, fetchCatalog]);

  useEffect(() => {
    void refresh();
    const onRefreshCommand = () => void refresh();
    window.addEventListener(WORKSPACE_EVENTS.refreshModels, onRefreshCommand);
    return () => window.removeEventListener(WORKSPACE_EVENTS.refreshModels, onRefreshCommand);
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

  const toggleLoad = async (m: CatalogModel) => {
    if (busyIds.has(m.id)) return;
    setBusyIds((prev) => new Set(prev).add(m.id));
    setError(null);
    try {
      if (m.status === "loaded") {
        await apiClient.post(`/models/${m.id}/unload`);
      } else if (m.status === "available" || m.status === "error") {
        await apiClient.post(`/models/${m.id}/load`);
      }
      await fetchCatalog();
    } catch (e) {
      const detail = axios.isAxiosError(e) ? (e.response?.data as any)?.detail : undefined;
      setError(typeof detail === "string" ? detail : `Could not ${m.status === "loaded" ? "unload" : "load"} ${m.name}.`);
    } finally {
      setBusyIds((prev) => {
        const next = new Set(prev);
        next.delete(m.id);
        return next;
      });
    }
  };

  const handleUseModel = async (model: CatalogModel) => {
    if (model.status === "unavailable") return;
    if (model.status === "available" || model.status === "error") {
      await toggleLoad(model);
      const refreshed = await fetchCatalog();
      const next = refreshed.find((item) => item.id === model.id);
      if (next?.status !== "loaded") return;
      onUseModel?.(next);
      return;
    }
    onUseModel?.(model);
  };

  const visibleCatalog = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    return catalog.filter((model) => {
      const familyMatches = familyFilter === "all" || model.family === familyFilter;
      const queryMatches = !normalized || [model.name, model.id, model.family, model.dataset, model.architecture]
        .filter(Boolean)
        .some((value) => String(value).toLowerCase().includes(normalized));
      return familyMatches && queryMatches;
    });
  }, [catalog, familyFilter, query]);

  const groups = useMemo(() => {
    const order: ModelFamily[] = ["ANN", "CNN", "RNN"];
    return order.map((fam) => ({
      family: fam,
      models: visibleCatalog.filter((m) => m.family === fam),
    }));
  }, [visibleCatalog]);

  return (
    <div className="min-h-full text-ink">
      <div className="page-shell [--shell-max:80rem] py-6 space-y-6">
        <PageHeader
          eyebrow="Discover · Inspect · Use · Manage"
          title="Models"
          subtitle="Manage locally trained weights and browse the pretrained model registry."
          actions={
            <NeuralButton size="sm" onClick={refresh} disabled={loading}>
              <RefreshCcw size={14} className={loading ? "animate-spin" : undefined} />
              Refresh
            </NeuralButton>
          }
        />

        {error && (
          <div role="alert" className="flex items-center justify-between gap-3 rounded-lg border border-status-danger/35 bg-status-danger/10 px-3 py-2 text-sm text-status-danger">
            <span className="break-all">{error}</span>
            <NeuralButton size="sm" onClick={() => void refresh()}>
              <RotateCw size={13} /> Retry
            </NeuralButton>
          </div>
        )}

        <section className="model-registry-toolbar" aria-label="Registry filters">
          <label className="model-registry-search">
            <Search size={15} aria-hidden="true" />
            <span className="sr-only">Search models</span>
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search models, datasets, or architectures" />
          </label>
          <label className="model-registry-filter">
            <span>Family</span>
            <select value={familyFilter} onChange={(event) => setFamilyFilter(event.target.value as "all" | ModelFamily)}>
              <option value="all">All families</option>
              <option value="ANN">ANN</option>
              <option value="CNN">CNN</option>
              <option value="RNN">RNN</option>
            </select>
          </label>
          <span className="model-registry-count">{visibleCatalog.length} registry models</span>
        </section>

        {/* ================= Pretrained registry ================= */}
        <section className="space-y-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Box size={18} className="text-ink-soft" aria-hidden="true" />
              <h2 className="text-base font-bold uppercase tracking-wide text-ink">Pretrained model registry</h2>
            </div>
            <span className="text-[12px] text-ink-mute">
              Real pre-trained checkpoints · lazy-loaded · CPU-first
            </span>
          </div>

          {loading && catalog.length === 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
              {[0, 1, 2].map((i) => (
                <div key={i} className="bg-white border border-barley-linestrong rounded-xl p-4 space-y-3 shadow-card">
                  <div className="neural-skeleton h-5 w-24" />
                  <div className="neural-skeleton h-3 w-full" />
                  <div className="neural-skeleton h-8 w-full" />
                </div>
              ))}
            </div>
          ) : catalog.length === 0 && !loading && !error ? (
            <div className="empty-state rounded-xl border border-dashed border-barley-linestrong bg-white py-12">
              <Database size={28} className="text-ink-faint" aria-hidden="true" />
              <p className="mt-2 text-sm font-medium text-ink-soft">No pretrained models in the catalog</p>
              <p className="text-xs text-ink-faint">Is the backend running? Check /models/catalog.</p>
            </div>
          ) : (
            groups.map(({ family, models: famModels }) => {
              const color = FAMILY_COLOR[family];
              if (famModels.length === 0) return null;
              return (
                <div key={family} className="space-y-3">
                  <div className="flex items-center gap-2">
                    <span className="h-3 w-3 rounded-full" style={{ background: color }} aria-hidden="true" />
                    <h3 className="text-sm font-bold uppercase tracking-wider" style={{ color }}>
                      {family}
                      <span className="ml-2 font-mono text-[11px] font-normal text-ink-mute">
                        {famModels.filter((m) => m.status === "available" || m.status === "loaded").length} runnable ·{" "}
                        {famModels.filter((m) => m.status === "unavailable").length} listed as unavailable
                      </span>
                    </h3>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                    {famModels.map((m) => {
                      const sm = STATUS_META[m.status] ?? STATUS_META.available;
                      const input = FAMILY_INPUT_HINT[m.input_type ?? ""];
                      const busy = busyIds.has(m.id);
                      const canLoad = m.status === "available" || m.status === "error";
                      const expanded = expandedIds.has(m.id);
                      const canUse = m.status === "available" || m.status === "loaded" || m.status === "error";
                      return (
                        <article
                          key={m.id}
                          className="relative bg-white border border-barley-linestrong rounded-xl p-4 space-y-3 shadow-card transition-shadow hover:shadow-md"
                          style={{ borderLeft: `4px solid ${color}` }}
                        >
                          <div className="flex items-start justify-between gap-2">
                            <div className="min-w-0">
                              <h4 className="text-sm font-bold text-ink leading-tight truncate" title={m.name}>
                                {m.name}
                              </h4>
                              <code className="text-[11px] font-mono text-ink-mute break-all">{m.id}</code>
                            </div>
                            <span
                              className={`inline-flex flex-none items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] font-semibold ${sm.cls}`}
                            >
                              <span className={`h-1.5 w-1.5 rounded-full ${sm.dot}`} aria-hidden="true" />
                              {sm.label}
                            </span>
                          </div>

                          {m.description && <p className="text-xs text-ink-soft leading-relaxed">{m.description}</p>}

                          {input && (
                            <div className="inline-flex items-center gap-1.5 rounded-md bg-barley-sunken px-2 py-1 text-[11px] text-ink-soft">
                              {input.icon} {input.label}
                              {m.input_shape ? ` · [${(m.input_shape as number[]).join("×")}]` : ""}
                            </div>
                          )}

                          <button
                            type="button"
                            className="model-details-toggle"
                            aria-expanded={expanded}
                            onClick={() => setExpandedIds((current) => {
                              const next = new Set(current);
                              if (next.has(m.id)) next.delete(m.id); else next.add(m.id);
                              return next;
                            })}
                          >
                            <span>{expanded ? "Hide technical details" : "View technical details"}</span>
                            <ChevronDown size={14} className={expanded ? "rotate-180" : ""} aria-hidden="true" />
                          </button>
                          {expanded ? <dl className="model-technical-details space-y-1 text-[11px]">
                            {m.framework && (
                              <div className="flex gap-2">
                                <dt className="w-20 shrink-0 text-ink-faint uppercase tracking-wide">Framework</dt>
                                <dd className="text-ink-soft break-words">{m.framework}</dd>
                              </div>
                            )}
                            {m.architecture && (
                              <div className="flex gap-2">
                                <dt className="w-20 shrink-0 text-ink-faint uppercase tracking-wide">Architecture</dt>
                                <dd className="text-ink-soft break-words">{m.architecture}</dd>
                              </div>
                            )}
                            <div className="flex gap-2">
                              <dt className="w-20 shrink-0 text-ink-faint uppercase tracking-wide inline-flex items-center gap-1">
                                <Hash size={11} /> Params
                              </dt>
                              <dd className="font-mono text-ink-soft">{fmtParams(m.parameter_count)}</dd>
                            </div>
                            <div className="flex gap-2">
                              <dt className="w-20 shrink-0 text-ink-faint uppercase tracking-wide inline-flex items-center gap-1">
                                <Braces size={11} /> Classes
                              </dt>
                              <dd className="font-mono text-ink-soft">{m.num_classes ?? "—"}</dd>
                            </div>
                            {m.dataset && (
                              <div className="flex gap-2">
                                <dt className="w-20 shrink-0 text-ink-faint uppercase tracking-wide inline-flex items-center gap-1">
                                  <Layers3 size={11} /> Dataset
                                </dt>
                                <dd className="text-ink-soft">{m.dataset}</dd>
                              </div>
                            )}
                            {m.preprocessing && (
                              <div className="flex gap-2">
                                <dt className="w-20 shrink-0 text-ink-faint uppercase tracking-wide inline-flex items-center gap-1">
                                  <Cpu size={11} /> Pre-process
                                </dt>
                                <dd className="text-ink-soft break-words">{m.preprocessing}</dd>
                              </div>
                            )}
                            {m.source && (
                              <div className="flex gap-2">
                                <dt className="w-20 shrink-0 text-ink-faint uppercase tracking-wide">Source</dt>
                                <dd className="text-ink-soft break-words break-all">{m.source}</dd>
                              </div>
                            )}
                            {m.weights && (
                              <div className="flex gap-2">
                                <dt className="w-20 shrink-0 text-ink-faint uppercase tracking-wide">Weights</dt>
                                <dd className="text-ink-soft break-all font-mono">{m.weights}</dd>
                              </div>
                            )}
                            {m.license && (
                              <div className="flex gap-2">
                                <dt className="w-20 shrink-0 text-ink-faint uppercase tracking-wide">License</dt>
                                <dd className="text-ink-soft">{m.license}</dd>
                              </div>
                            )}
                          </dl> : null}

                          {m.status === "unavailable" && m.unavailable_reason && (
                            <div className="flex gap-2 rounded-md border border-status-danger/25 bg-status-danger/5 px-2 py-1.5 text-[11px] text-status-danger leading-snug">
                              <FileWarning size={13} className="mt-0.5 shrink-0" />
                              <span>{m.unavailable_reason}</span>
                            </div>
                          )}
                          {m.status === "error" && m.error && (
                            <div className="flex gap-2 rounded-md border border-status-danger/25 bg-status-danger/5 px-2 py-1.5 text-[11px] text-status-danger leading-snug">
                              <FileWarning size={13} className="mt-0.5 shrink-0" />
                              <span>{m.error}</span>
                            </div>
                          )}

                          {canUse && (
                            <div className="model-card-actions">
                              <button
                                type="button"
                                disabled={busy}
                                onClick={() => void handleUseModel(m)}
                                className="model-use-button"
                                style={{ background: color }}
                              >
                                {busy ? <RotateCw size={13} className="animate-spin" aria-hidden="true" /> : <Check size={13} aria-hidden="true" />}
                                {m.status === "loaded" ? "Use model" : "Load and use"}
                              </button>
                              {canLoad || m.status === "loaded" ? (
                                <button
                                  type="button"
                                  disabled={busy}
                                  onClick={() => void toggleLoad(m)}
                                  className="model-secondary-action"
                                >
                                  {busy ? <RotateCw size={13} className="animate-spin" aria-hidden="true" /> : m.status === "loaded" ? <Upload size={13} aria-hidden="true" /> : <Download size={13} aria-hidden="true" />}
                                  {m.status === "loaded" ? "Unload" : "Load"}
                                </button>
                              ) : null}
                            </div>
                          )}
                        </article>
                      );
                    })}
                  </div>
                </div>
              );
            })
          )}
        </section>

        {/* ================= Locally trained (legacy) models ================= */}
        <section className="space-y-3">
          <div className="flex items-center gap-2">
            <Cpu size={16} className="text-ink-soft" aria-hidden="true" />
            <h2 className="text-base font-bold uppercase tracking-wide text-ink">Saved weights (trained in-app)</h2>
          </div>

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
            <div className="empty-state rounded-xl border border-dashed border-barley-linestrong bg-white py-10">
              <Cpu size={24} className="text-ink-faint" aria-hidden="true" />
              <p className="mt-2 text-sm font-medium text-ink-soft">No locally trained weights on disk</p>
              <p className="text-xs text-ink-faint">Train a model in Training mode to manage it here.</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
              {models.map((m) => {
                const fam = (m.model_type.toUpperCase() === "CNN" ? "CNN" : m.model_type.toUpperCase() === "RNN" ? "RNN" : "ANN") as ModelFamily;
                const color = FAMILY_COLOR[fam];
                return (
                  <article
                    key={m.model_type}
                    className={`relative bg-white border border-barley-linestrong rounded-xl p-4 space-y-3 shadow-card transition-shadow hover:shadow-md`}
                    style={{ borderLeft: `4px solid ${color}` }}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2 min-w-0">
                        <span className="h-2.5 w-2.5 rounded-full flex-none" style={{ background: color }} aria-hidden="true" />
                        <h3 className="text-base font-bold uppercase tracking-wide text-ink truncate">{m.model_type}</h3>
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
                        type="button"
                        disabled={!m.loaded || active === m.model_type}
                        onClick={() => void switchModel(m.model_type)}
                        className="flex-1 h-9 text-xs font-semibold px-2 rounded-md bg-ember-700 hover:bg-ember-800 active:scale-[0.98] text-white transition-colors disabled:bg-barley-sunken disabled:text-ink-faint disabled:cursor-not-allowed"
                      >
                        {active === m.model_type ? "In use" : "Use"}
                      </button>
                      <button
                        type="button"
                        disabled={!m.exists_on_disk}
                        onClick={() => void reloadModel(m.model_type)}
                        className="inline-flex h-9 w-10 items-center justify-center rounded-md bg-barley-wash hover:bg-barley-sunken border border-barley-linestrong transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-ink-soft"
                        title="Reload from disk"
                        aria-label={`Reload ${m.model_type} from disk`}
                      >
                        <RotateCw size={14} />
                      </button>
                      <button
                        type="button"
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
        </section>
      </div>
    </div>
  );
}
