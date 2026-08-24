import React, { useState } from "react";
import { useTrainingSocket } from "../../hooks/useTrainingSocket";
import { Play, Square, Pause, Wifi, WifiOff, RefreshCcw, Save } from "lucide-react";
import { TrainingConfig, ModelType } from "../../types";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { apiClient } from "../../api/client";
import { PageHeader } from "@/design-system/components/PageHeader";
import { NeuralButton } from "@/design-system/components/NeuralButton";

function metricBox(label: string, value: string) {
  return (
    <div className="bg-white border border-barley-linestrong rounded p-3">
      <div className="text-[12px] uppercase tracking-wide text-ink-mute">{label}</div>
      <div className="text-base font-semibold mt-1 text-ink">{value}</div>
    </div>
  );
}

export default function TrainingMode() {
  const { status, history, batchHistory, liveBatch, logs, sendCommand, isConnected, connect } = useTrainingSocket();
  const [isSaving, setIsSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [config, setConfig] = useState<TrainingConfig>({
    model_type: "ann",
    epochs: 10,
    batch_size: 32,
    learning_rate: 0.001,
    optimizer: "adam"
  });

  const handleStart = () => sendCommand("start", config);
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
      setSaveMessage(typeof detail === "string" ? detail : "Could not save model. Ensure training produced an in-memory model.");
    } finally {
      setIsSaving(false);
    }
  };

  const latestEpoch = history.length > 0 ? history[history.length - 1] : null;
  const avg = (arr?: number[]) => {
    if (!arr || arr.length === 0) return 0;
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  };
  const batchProgress = liveBatch ? `${liveBatch.batch}/${liveBatch.total_batches}` : "0/0";
  const epochProgress = `${status.current_epoch}/${status.total_epochs || config.epochs}`;

  return (
    <div className="h-full overflow-auto text-ink">
      <div className="page-shell [--shell-max:100rem] flex flex-col gap-4 py-4">
        <PageHeader
          title="Training"
          subtitle={`Configure a run, stream live metrics, save the result · Epoch ${epochProgress}`}
          actions={
            <div className="flex items-center gap-3 text-sm">
              {isConnected ? (
                <span className="flex items-center gap-1.5 text-status-success">
                  <Wifi size={16} /> Socket connected
                </span>
              ) : (
                <>
                  <span className="flex items-center gap-1.5 text-arch-rnn">
                    <WifiOff size={16} /> Socket disconnected
                  </span>
                  <NeuralButton size="sm" onClick={connect}>
                    <RefreshCcw size={14} /> Reconnect
                  </NeuralButton>
                </>
              )}
            </div>
          }
        />

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {metricBox("Model", config.model_type.toUpperCase())}
        {metricBox("Training State", status.status.toUpperCase())}
        {metricBox("Batch Progress", batchProgress)}
        {metricBox("Learning Rate", liveBatch ? liveBatch.learning_rate.toExponential(2) : config.learning_rate.toExponential(2))}
        {metricBox("Live Loss", liveBatch ? liveBatch.loss.toFixed(4) : "-")}
        {metricBox("Live Accuracy", liveBatch ? `${(liveBatch.accuracy * 100).toFixed(2)}%` : "-")}
        {metricBox("Gradient Norm", liveBatch ? liveBatch.gradient_norm.toFixed(4) : "-")}
        {metricBox("Val Accuracy", latestEpoch ? `${(latestEpoch.val_accuracy * 100).toFixed(2)}%` : "-")}
      </div>

      {latestEpoch && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {metricBox("Avg Precision", `${(avg(latestEpoch.precision_per_class) * 100).toFixed(2)}%`)}
          {metricBox("Avg Recall", `${(avg(latestEpoch.recall_per_class) * 100).toFixed(2)}%`)}
          {metricBox("Avg F1", `${(avg(latestEpoch.f1_per_class) * 100).toFixed(2)}%`)}
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-[minmax(300px,380px)_minmax(0,1fr)] gap-4 min-h-0">
        <div className="bg-barley-wash p-4 rounded-lg border border-barley-linestrong overflow-y-auto xl:max-h-[calc(100vh-290px)]">
          <h2 className="text-xl font-bold mb-4 text-ember-700">Training Configuration</h2>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-ink-mute">Model Architecture</label>
              <select 
                className="w-full bg-white border border-barley-linestrong rounded p-2 mt-1"
                value={config.model_type}
                onChange={(e) => setConfig({...config, model_type: e.target.value as ModelType})}
                disabled={status.status === "training"}
              >
                <option value="ann">ANN (Dense)</option>
                <option value="cnn">CNN (Convolutional)</option>
                <option value="rnn">RNN (LSTM)</option>
              </select>
            </div>

            <div className="grid grid-cols-2 gap-2">
                <div>
                    <label className="block text-sm text-ink-mute">Epochs</label>
                    <input 
                        type="number" 
                        value={config.epochs}
                        onChange={(e) => setConfig({...config, epochs: parseInt(e.target.value)})}
                        className="w-full bg-white border border-barley-linestrong rounded p-2"
                    />
                </div>
                <div>
                    <label className="block text-sm text-ink-mute">Batch Size</label>
                    <input 
                        type="number" 
                        value={config.batch_size}
                        onChange={(e) => setConfig({...config, batch_size: parseInt(e.target.value)})}
                        className="w-full bg-white border border-barley-linestrong rounded p-2"
                    />
                </div>
            </div>

            <div className="flex gap-2 mt-6">
                {status.status === "idle" || status.status === "completed" || status.status === "stopped" ? (
                    <button
                        onClick={handleStart}
                        disabled={!isConnected}
                        className="flex-1 bg-status-success hover:bg-status-successhover active:scale-[0.98] disabled:bg-barley-sunken disabled:text-ink-faint text-white p-2 rounded-md flex items-center justify-center gap-2 transition-colors"
                    >
                        <Play size={16} /> Start Training
                    </button>
                ) : (
                    <button
                        onClick={handleStop}
                        className="flex-1 bg-status-danger hover:bg-status-dangerhover active:scale-[0.98] text-white p-2 rounded-md flex items-center justify-center gap-2 transition-colors"
                    >
                        <Square size={16} /> Stop
                    </button>
                )}

                {status.status === "training" && (
                    <button onClick={handlePause} aria-label="Pause training" title="Pause" className="neural-button neural-button-icon"><Pause size={16}/></button>
                )}
                {status.status === "paused" && (
                     <button onClick={handleResume} aria-label="Resume training" title="Resume" className="neural-button neural-button-icon"><Play size={16}/></button>
                )}
            </div>
            <button
              onClick={handleSaveModel}
              disabled={isSaving || status.status === "training"}
              className="w-full mt-2 bg-ember-700 hover:bg-ember-800 active:scale-[0.99] disabled:bg-barley-sunken disabled:text-ink-faint text-white p-2 rounded-md flex items-center justify-center gap-2 transition-colors"
            >
              <Save size={16} /> {isSaving ? "Saving..." : `Save ${config.model_type.toUpperCase()} Model`}
            </button>
            {saveMessage && (
              <div className="text-xs mt-2 p-2 rounded border border-barley-linestrong bg-white text-ink">{saveMessage}</div>
            )}
          </div>

          <div className="mt-6">
            <h3 className="text-sm font-semibold mb-2">System Logs</h3>
            <div className="bg-ink h-48 overflow-y-auto p-2 text-xs font-mono text-status-successbright rounded border border-ink">
                {logs.map((log, i) => <div key={i}>{log}</div>)}
                {logs.length === 0 && <div className="text-barley-page/40">Waiting for training events...</div>}
            </div>
          </div>
        </div>

        <div className="flex flex-col gap-4 min-w-0">
            <div className="bg-barley-wash p-4 rounded-lg border border-barley-linestrong">
                <h3 className="text-lg font-bold mb-2">Real-time Batch Metrics</h3>
                <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={batchHistory}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#E7E0D4" />
                            <XAxis dataKey="batch" stroke="#79716B" />
                            <YAxis stroke="#79716B" />
                            <Tooltip contentStyle={{ backgroundColor: '#FFFFFF', border: '1px solid #E7E0D4', borderRadius: '8px' }} />
                            <Legend />
                            <Line type="monotone" dataKey="loss" stroke="#B91C1C" name="Batch Loss" strokeWidth={2} dot={false} />
                            <Line type="monotone" dataKey="accuracy" stroke="#15803D" name="Batch Accuracy" strokeWidth={2} dot={false} />
                            <Line type="monotone" dataKey="gradient_norm" stroke="#A64D85" name="Gradient Norm" strokeWidth={2} dot={false} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>
            <div className="bg-barley-wash p-4 rounded-lg border border-barley-linestrong">
                <h3 className="text-lg font-bold mb-2">Accuracy Metrics</h3>
                <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={history}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#E7E0D4" />
                            <XAxis dataKey="epoch" stroke="#79716B" />
                            <YAxis domain={[0, 1]} stroke="#79716B" />
                            <Tooltip contentStyle={{ backgroundColor: '#FFFFFF', border: '1px solid #E7E0D4', borderRadius: '8px' }} />
                            <Legend />
                            <Line type="monotone" dataKey="accuracy" stroke="#009E73" name="Train Acc" strokeWidth={2} dot={false} />
                            <Line type="monotone" dataKey="val_accuracy" stroke="#0072B2" name="Val Acc" strokeWidth={2} dot={false} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>

            <div className="bg-barley-wash p-4 rounded-lg border border-barley-linestrong">
                <h3 className="text-lg font-bold mb-2">Loss Metrics</h3>
                <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={history}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#E7E0D4" />
                            <XAxis dataKey="epoch" stroke="#79716B" />
                            <YAxis stroke="#79716B" />
                            <Tooltip contentStyle={{ backgroundColor: '#FFFFFF', border: '1px solid #E7E0D4', borderRadius: '8px' }} />
                            <Legend />
                            <Line type="monotone" dataKey="loss" stroke="#B91C1C" name="Train Loss" strokeWidth={2} dot={false} />
                            <Line type="monotone" dataKey="val_loss" stroke="#D55E00" name="Val Loss" strokeWidth={2} dot={false} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
        </div>
      </div>
    </div>
  );
}
