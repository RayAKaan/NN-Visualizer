import React, { useEffect, useMemo, useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { Pause, Play, RefreshCcw, Save, Square, Wifi, WifiOff } from "lucide-react";
import { useTrainingSocket } from "../../hooks/useTrainingSocket";
import { TrainingConfig, ModelType } from "../../types";
import { apiClient } from "../../api/client";
import { PageHeader } from "@/design-system/components/PageHeader";
import { NeuralBadge } from "@/design-system/components/NeuralBadge";
import { NeuralButton } from "@/design-system/components/NeuralButton";
import { NeuralInput } from "@/design-system/components/NeuralInput";
import { NeuralProgress } from "@/design-system/components/NeuralProgress";
import { NeuralSelect } from "@/design-system/components/NeuralSelect";
import { NeuralPanel } from "@/design-system/components/NeuralPanel";
import { WORKSPACE_EVENTS } from "../../utils/workspaceEvents";

function Metric({ label, value, tone = "default" }: { label: string; value: string; tone?: "default" | "accent" | "success" }) {
  return (
    <div className={`training-metric-card is-${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function EmptyChart({ message }: { message: string }) {
  return <div className="training-chart-empty"><span>{message}</span></div>;
}

export default function TrainingMode() {
  const { status, history, batchHistory, liveBatch, logs, sendCommand, isConnected, connect } = useTrainingSocket();
  const [isSaving, setIsSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [showSecondary, setShowSecondary] = useState(false);
  const [showDiagnostics, setShowDiagnostics] = useState(false);
  const [config, setConfig] = useState<TrainingConfig>({
    model_type: "ann",
    epochs: 10,
    batch_size: 32,
    learning_rate: 0.001,
    optimizer: "adam",
  });

  const latestEpoch = history.length > 0 ? history[history.length - 1] : null;
  const average = (values?: number[]) => values && values.length > 0 ? values.reduce((total: number, value: number) => total + value, 0) / values.length : null;
  const isRunning = status.status === "training";
  const isPaused = status.status === "paused";
  const canStart = status.status === "idle" || status.status === "completed" || status.status === "stopped";
  const epochTotal = status.total_epochs || config.epochs;
  const epochProgress = epochTotal > 0 ? Math.min(100, (status.current_epoch / epochTotal) * 100) : 0;
  const batchProgress = liveBatch ? `${liveBatch.batch} / ${liveBatch.total_batches}` : "—";
  const validationError = useMemo(() => {
    if (!Number.isInteger(config.epochs) || config.epochs < 1) return "Epochs must be at least 1.";
    if (!Number.isInteger(config.batch_size) || config.batch_size < 1) return "Batch size must be at least 1.";
    if (!Number.isFinite(config.learning_rate) || config.learning_rate <= 0) return "Learning rate must be greater than 0.";
    return null;
  }, [config]);

  const handleStart = () => {
    if (validationError || !isConnected) return;
    setSaveMessage(null);
    sendCommand("start", config);
  };
  const handleStop = () => sendCommand("stop");
  const handlePause = () => sendCommand("pause");
  const handleResume = () => sendCommand("resume");

  const handleSaveModel = async () => {
    setSaveMessage(null);
    setIsSaving(true);
    try {
      const res = await apiClient.post(`/models/${config.model_type}/save`);
      const path = typeof res.data?.path === "string" ? res.data.path : "model file";
      setSaveMessage(`Saved ${config.model_type.toUpperCase()} model to ${path}`);
    } catch (err: any) {
      const detail = err?.response?.data?.detail;
      setSaveMessage(typeof detail === "string" ? detail : "Could not save model. Complete a run first.");
    } finally {
      setIsSaving(false);
    }
  };

  useEffect(() => {
    const onStartCommand = () => handleStart();
    window.addEventListener(WORKSPACE_EVENTS.startTraining, onStartCommand);
    return () => window.removeEventListener(WORKSPACE_EVENTS.startTraining, onStartCommand);
  }, [handleStart]);

  const stateCopy: Record<string, { label: string; body: string; tone: "neutral" | "info" | "success" | "warning" | "danger" }> = {
    idle: { label: "Ready to configure", body: "Choose a model and run settings, then start the experiment.", tone: "neutral" },
    training: { label: "Training in progress", body: "Metrics update as batches and epochs complete.", tone: "info" },
    paused: { label: "Training paused", body: "Review the latest metrics or resume this run.", tone: "warning" },
    stopping: { label: "Stopping run", body: "The current batch is finishing and the run will stop safely.", tone: "warning" },
    stopped: { label: "Run stopped", body: "The latest completed metrics remain available for evaluation.", tone: "warning" },
    completed: { label: "Training complete", body: "Review validation performance, then save the model.", tone: "success" },
  };
  const currentState = stateCopy[status.status] ?? stateCopy.idle;

  return (
    <div className="training-page text-ink">
      <div className="page-shell [--shell-max:110rem] py-6">
        <PageHeader
          eyebrow="Train and save a model"
          title="Training"
          subtitle="Configure → run → monitor → evaluate → save. Keep the training run—not the raw log—at the center."
          status={
            <span className={`training-connection ${isConnected ? "is-online" : "is-offline"}`} role="status">
              {isConnected ? <Wifi size={14} aria-hidden="true" /> : <WifiOff size={14} aria-hidden="true" />}
              {isConnected ? "Connected" : "Disconnected"}
            </span>
          }
          actions={!isConnected ? <NeuralButton size="sm" onClick={connect}><RefreshCcw size={14} /> Reconnect</NeuralButton> : null}
        />

        <form onSubmit={(event) => { event.preventDefault(); handleStart(); }}>
          <section className="training-run-banner" aria-labelledby="training-run-heading">
            <div className="training-run-copy">
              <span className="training-kicker">Training run</span>
              <h2 id="training-run-heading">{currentState.label}</h2>
              <p>{currentState.body}</p>
            </div>
            <div className="training-run-actions">
              {canStart ? (
                <NeuralButton size="lg" variant="primary" type="submit" disabled={!isConnected || Boolean(validationError)}>
                  <Play size={16} /> Start training
                </NeuralButton>
              ) : isRunning ? (
                <>
                  <NeuralButton size="lg" variant="secondary" onClick={handlePause}><Pause size={16} /> Pause</NeuralButton>
                  <NeuralButton size="lg" variant="danger" onClick={handleStop}><Square size={16} /> Stop</NeuralButton>
                </>
              ) : isPaused ? (
                <>
                  <NeuralButton size="lg" variant="primary" onClick={handleResume}><Play size={16} /> Resume</NeuralButton>
                  <NeuralButton size="lg" variant="danger" onClick={handleStop}><Square size={16} /> Stop</NeuralButton>
                </>
              ) : null}
              {(status.status === "completed" || status.status === "stopped") ? (
                <NeuralButton size="lg" variant="secondary" onClick={() => void handleSaveModel()} disabled={isSaving}>
                  <Save size={16} /> {isSaving ? "Saving…" : "Save model"}
                </NeuralButton>
              ) : null}
            </div>
            <div className="training-run-progress">
              <NeuralProgress value={epochProgress} label={`Epoch ${status.current_epoch} of ${epochTotal}`} showValue />
              {liveBatch ? <span>Batch {batchProgress} · live loss {liveBatch.loss.toFixed(4)}</span> : <span>Epoch progress appears when a run starts.</span>}
            </div>
          </section>

          <div className="training-layout">
            <NeuralPanel className="training-config-panel" variant="base">
              <div className="training-panel-heading">
                <div><span className="training-kicker">Configure</span><h2>Run settings</h2></div>
                <NeuralBadge tone={validationError ? "danger" : "neutral"}>{validationError ? "Check inputs" : "Ready"}</NeuralBadge>
              </div>
              <div className="training-form-grid">
                <label>
                  <span>Model architecture</span>
                  <NeuralSelect value={config.model_type} onChange={(event) => setConfig({ ...config, model_type: event.target.value as ModelType })} disabled={isRunning}>
                    <option value="ann">ANN · Dense</option>
                    <option value="cnn">CNN · Convolutional</option>
                    <option value="rnn">RNN · LSTM</option>
                  </NeuralSelect>
                </label>
                <label>
                  <span>Epochs</span>
                  <NeuralInput type="number" min={1} step={1} value={config.epochs} onChange={(event) => setConfig({ ...config, epochs: Number(event.target.value) })} disabled={isRunning} />
                </label>
                <label>
                  <span>Batch size</span>
                  <NeuralInput type="number" min={1} step={1} value={config.batch_size} onChange={(event) => setConfig({ ...config, batch_size: Number(event.target.value) })} disabled={isRunning} />
                </label>
                <label>
                  <span>Learning rate</span>
                  <NeuralInput type="number" min={0.000001} step={0.0001} value={config.learning_rate} onChange={(event) => setConfig({ ...config, learning_rate: Number(event.target.value) })} disabled={isRunning} />
                </label>
                <label>
                  <span>Optimizer</span>
                  <NeuralSelect value={config.optimizer} onChange={(event) => setConfig({ ...config, optimizer: event.target.value })} disabled={isRunning}>
                    <option value="adam">Adam</option>
                    <option value="sgd">SGD</option>
                    <option value="rmsprop">RMSprop</option>
                  </NeuralSelect>
                </label>
              </div>
              {validationError ? <p className="training-field-error" role="alert">{validationError}</p> : <p className="training-form-help">These settings apply to the next run. You can change them after stopping.</p>}
              {saveMessage ? <div className="training-save-message" role="status">{saveMessage}</div> : null}
            </NeuralPanel>

            <main className="training-monitor" aria-label="Training monitor">
              <section className="training-section-heading">
                <div><span className="training-kicker">Monitor</span><h2>Run health</h2></div>
                <span className="training-section-note">Primary metrics</span>
              </section>
              <div className="training-primary-metrics">
                <Metric label="Epoch" value={`${status.current_epoch} / ${epochTotal}`} tone="accent" />
                <Metric label="Training loss" value={liveBatch ? liveBatch.loss.toFixed(4) : latestEpoch ? latestEpoch.loss.toFixed(4) : "—"} />
                <Metric label="Validation loss" value={latestEpoch ? latestEpoch.val_loss.toFixed(4) : "—"} />
                <Metric label="Training accuracy" value={liveBatch ? `${(liveBatch.accuracy * 100).toFixed(1)}%` : latestEpoch ? `${(latestEpoch.accuracy * 100).toFixed(1)}%` : "—"} tone="success" />
                <Metric label="Validation accuracy" value={latestEpoch ? `${(latestEpoch.val_accuracy * 100).toFixed(1)}%` : "—"} tone="success" />
                <Metric label="Batch" value={batchProgress} />
              </div>

              <section className="training-evaluation-heading" aria-labelledby="training-evaluation-heading">
                <div><span className="training-kicker">Evaluate</span><h2 id="training-evaluation-heading">Validation behavior</h2></div>
                <span>Compare what the model learns with what it generalizes.</span>
              </section>
              <section className="training-chart-grid" aria-label="Training charts">
                <NeuralPanel className="training-chart-panel" variant="base">
                  <div className="training-chart-heading"><div><h3>Accuracy</h3><span>Train vs validation · proportion</span></div></div>
                  {history.length === 0 ? <EmptyChart message="Accuracy will appear after the first epoch." /> : (
                    <div className="training-chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={history}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#E7E0D4" />
                      <XAxis dataKey="epoch" stroke="#79716B" fontSize={11} />
                      <YAxis domain={[0, 1]} stroke="#79716B" fontSize={11} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                      <Tooltip contentStyle={{ backgroundColor: "#FFFFFF", border: "1px solid #E7E0D4", borderRadius: "8px" }} formatter={(value: number) => `${(value * 100).toFixed(1)}%`} />
                      <Legend />
                      <Line type="monotone" dataKey="accuracy" stroke="#009E73" name="Train" strokeWidth={2} dot={false} />
                      <Line type="monotone" dataKey="val_accuracy" stroke="#0072B2" name="Validation" strokeWidth={2} dot={false} />
                    </LineChart></ResponsiveContainer></div>
                  )}
                </NeuralPanel>
                <NeuralPanel className="training-chart-panel" variant="base">
                  <div className="training-chart-heading"><div><h3>Loss</h3><span>Train vs validation · lower is better</span></div></div>
                  {history.length === 0 ? <EmptyChart message="Loss will appear after the first epoch." /> : (
                    <div className="training-chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={history}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#E7E0D4" />
                      <XAxis dataKey="epoch" stroke="#79716B" fontSize={11} />
                      <YAxis stroke="#79716B" fontSize={11} />
                      <Tooltip contentStyle={{ backgroundColor: "#FFFFFF", border: "1px solid #E7E0D4", borderRadius: "8px" }} />
                      <Legend />
                      <Line type="monotone" dataKey="loss" stroke="#B91C1C" name="Train" strokeWidth={2} dot={false} />
                      <Line type="monotone" dataKey="val_loss" stroke="#D55E00" name="Validation" strokeWidth={2} dot={false} />
                    </LineChart></ResponsiveContainer></div>
                  )}
                </NeuralPanel>
              </section>

              <section className="training-disclosure-group">
                <button type="button" className="training-disclosure" onClick={() => setShowSecondary((open) => !open)} aria-expanded={showSecondary}>
                  <span><strong>More metrics</strong><small>Precision, recall, F1, and gradient norm</small></span><span aria-hidden="true">{showSecondary ? "−" : "+"}</span>
                </button>
                {showSecondary ? <div className="training-secondary-metrics">
                  <Metric label="Avg precision" value={average(latestEpoch?.precision_per_class) != null ? `${(average(latestEpoch?.precision_per_class)! * 100).toFixed(1)}%` : "—"} />
                  <Metric label="Avg recall" value={average(latestEpoch?.recall_per_class) != null ? `${(average(latestEpoch?.recall_per_class)! * 100).toFixed(1)}%` : "—"} />
                  <Metric label="Avg F1" value={average(latestEpoch?.f1_per_class) != null ? `${(average(latestEpoch?.f1_per_class)! * 100).toFixed(1)}%` : "—"} />
                  <Metric label="Gradient norm" value={liveBatch ? liveBatch.gradient_norm.toFixed(4) : "—"} />
                </div> : null}
              </section>

              <section className="training-disclosure-group">
                <button type="button" className="training-disclosure" onClick={() => setShowDiagnostics((open) => !open)} aria-expanded={showDiagnostics}>
                  <span><strong>Diagnostics</strong><small>Batch chart and raw training logs</small></span><span aria-hidden="true">{showDiagnostics ? "−" : "+"}</span>
                </button>
                {showDiagnostics ? <div className="training-diagnostics">
                  <NeuralPanel className="training-chart-panel" variant="sunken">
                    <div className="training-chart-heading"><div><h3>Live batch metrics</h3><span>Useful while debugging a run</span></div></div>
                    {batchHistory.length === 0 ? <EmptyChart message="Batch metrics appear while training." /> : <div className="training-chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={batchHistory}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#E7E0D4" /><XAxis dataKey="batch" stroke="#79716B" fontSize={11} /><YAxis stroke="#79716B" fontSize={11} /><Tooltip contentStyle={{ backgroundColor: "#FFFFFF", border: "1px solid #E7E0D4", borderRadius: "8px" }} /><Legend />
                      <Line type="monotone" dataKey="loss" stroke="#B91C1C" name="Loss" strokeWidth={2} dot={false} /><Line type="monotone" dataKey="accuracy" stroke="#009E73" name="Accuracy" strokeWidth={2} dot={false} />
                    </LineChart></ResponsiveContainer></div>}
                  </NeuralPanel>
                  <div className="training-log-panel"><div className="training-chart-heading"><div><h3>Training log</h3><span>Latest events from the socket</span></div></div><pre>{logs.length ? logs.join("\n") : "Waiting for training events…"}</pre></div>
                </div> : null}
              </section>
            </main>
          </div>
        </form>
      </div>
    </div>
  );
}
