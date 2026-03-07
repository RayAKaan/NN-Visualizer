import React, { useState } from "react";
import { useTrainingSocket } from "../../hooks/useTrainingSocket";
import { Play, Square, Pause, Wifi, WifiOff, RefreshCcw, Save } from "lucide-react";
import { TrainingConfig, ModelType } from "../../types";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { apiClient } from "../../api/client";

function metricBox(label: string, value: string) {
  return (
    <div className="bg-slate-900/70 border border-slate-700 rounded p-3">
      <div className="text-[11px] uppercase tracking-wide text-slate-400">{label}</div>
      <div className="text-base font-semibold mt-1 text-slate-100">{value}</div>
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
    <div className="flex flex-col h-full min-h-0 gap-4 p-4 text-slate-100 overflow-auto">
      <div className="flex items-center justify-between bg-slate-900 border border-slate-700 rounded-lg px-4 py-2">
        <div className="flex items-center gap-2 text-sm">
          {isConnected ? <Wifi size={16} className="text-emerald-400" /> : <WifiOff size={16} className="text-rose-400" />}
          <span className={isConnected ? "text-emerald-300" : "text-rose-300"}>
            {isConnected ? "Training socket connected" : "Training socket disconnected"}
          </span>
          <span className="text-slate-400">Epoch {epochProgress}</span>
        </div>
        {!isConnected && (
          <button
            onClick={connect}
            className="flex items-center gap-2 px-3 py-1.5 text-xs rounded border border-slate-600 bg-slate-800 hover:bg-slate-700"
          >
            <RefreshCcw size={14} /> Reconnect
          </button>
        )}
      </div>

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

      <div className="grid grid-cols-1 xl:grid-cols-[360px_minmax(0,1fr)] gap-4 min-h-0">
        <div className="bg-slate-800 p-4 rounded-lg border border-slate-700 overflow-y-auto xl:max-h-[calc(100vh-290px)]">
          <h2 className="text-xl font-bold mb-4 text-cyan-400">Training Configuration</h2>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-slate-400">Model Architecture</label>
              <select 
                className="w-full bg-slate-900 border border-slate-600 rounded p-2 mt-1"
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
                    <label className="block text-sm text-slate-400">Epochs</label>
                    <input 
                        type="number" 
                        value={config.epochs}
                        onChange={(e) => setConfig({...config, epochs: parseInt(e.target.value)})}
                        className="w-full bg-slate-900 border border-slate-600 rounded p-2"
                    />
                </div>
                <div>
                    <label className="block text-sm text-slate-400">Batch Size</label>
                    <input 
                        type="number" 
                        value={config.batch_size}
                        onChange={(e) => setConfig({...config, batch_size: parseInt(e.target.value)})}
                        className="w-full bg-slate-900 border border-slate-600 rounded p-2"
                    />
                </div>
            </div>

            <div className="flex gap-2 mt-6">
                {status.status === "idle" || status.status === "completed" || status.status === "stopped" ? (
                    <button 
                        onClick={handleStart} 
                        disabled={!isConnected}
                        className="flex-1 bg-green-600 hover:bg-green-500 disabled:bg-slate-600 disabled:text-slate-300 text-white p-2 rounded flex items-center justify-center gap-2"
                    >
                        <Play size={16} /> Start Training
                    </button>
                ) : (
                    <button 
                        onClick={handleStop} 
                        className="flex-1 bg-red-600 hover:bg-red-500 text-white p-2 rounded flex items-center justify-center gap-2"
                    >
                        <Square size={16} /> Stop
                    </button>
                )}
                
                {status.status === "training" && (
                    <button onClick={handlePause} className="bg-yellow-600 p-2 rounded"><Pause size={16}/></button>
                )}
                {status.status === "paused" && (
                     <button onClick={handleResume} className="bg-green-600 p-2 rounded"><Play size={16}/></button>
                )}
            </div>
            <button
              onClick={handleSaveModel}
              disabled={isSaving || status.status === "training"}
              className="w-full mt-2 bg-cyan-700 hover:bg-cyan-600 disabled:bg-slate-600 disabled:text-slate-300 text-white p-2 rounded flex items-center justify-center gap-2"
            >
              <Save size={16} /> {isSaving ? "Saving..." : `Save ${config.model_type.toUpperCase()} Model`}
            </button>
            {saveMessage && (
              <div className="text-xs mt-2 p-2 rounded border border-slate-600 bg-slate-900/60 text-slate-200">{saveMessage}</div>
            )}
          </div>

          <div className="mt-6">
            <h3 className="text-sm font-semibold mb-2">System Logs</h3>
            <div className="bg-black/50 h-48 overflow-y-auto p-2 text-xs font-mono text-green-400 rounded">
                {logs.map((log, i) => <div key={i}>{log}</div>)}
                {logs.length === 0 && <div className="text-slate-500">Waiting for training events...</div>}
            </div>
          </div>
        </div>

        <div className="flex flex-col gap-4 min-w-0">
            <div className="bg-slate-800 p-4 rounded-lg border border-slate-700">
                <h3 className="text-lg font-bold mb-2">Real-time Batch Metrics</h3>
                <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={batchHistory}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                            <XAxis dataKey="batch" stroke="#888" />
                            <YAxis stroke="#888" />
                            <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: 'none' }} />
                            <Legend />
                            <Line type="monotone" dataKey="loss" stroke="#f43f5e" name="Batch Loss" strokeWidth={2} dot={false} />
                            <Line type="monotone" dataKey="accuracy" stroke="#22c55e" name="Batch Accuracy" strokeWidth={2} dot={false} />
                            <Line type="monotone" dataKey="gradient_norm" stroke="#a78bfa" name="Gradient Norm" strokeWidth={2} dot={false} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>
            <div className="bg-slate-800 p-4 rounded-lg border border-slate-700">
                <h3 className="text-lg font-bold mb-2">Accuracy Metrics</h3>
                <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={history}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                            <XAxis dataKey="epoch" stroke="#888" />
                            <YAxis domain={[0, 1]} stroke="#888" />
                            <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: 'none' }} />
                            <Legend />
                            <Line type="monotone" dataKey="accuracy" stroke="#00ff88" name="Train Acc" strokeWidth={2} dot={false} />
                            <Line type="monotone" dataKey="val_accuracy" stroke="#0088ff" name="Val Acc" strokeWidth={2} dot={false} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>

            <div className="bg-slate-800 p-4 rounded-lg border border-slate-700">
                <h3 className="text-lg font-bold mb-2">Loss Metrics</h3>
                <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={history}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                            <XAxis dataKey="epoch" stroke="#888" />
                            <YAxis stroke="#888" />
                            <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: 'none' }} />
                            <Legend />
                            <Line type="monotone" dataKey="loss" stroke="#ff4444" name="Train Loss" strokeWidth={2} dot={false} />
                            <Line type="monotone" dataKey="val_loss" stroke="#ff8844" name="Val Loss" strokeWidth={2} dot={false} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
      </div>
    </div>
  );
}
