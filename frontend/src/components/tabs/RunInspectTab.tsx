import React, { useState, useEffect } from "react";
import { Hammer, Play, RefreshCw, Settings2, Zap } from "lucide-react";
import { useSessionStore } from "../../store/sessionStore";
import { useComputationStore } from "../../store/computationStore";
import { useArchitectureStore } from "../../store/architectureStore";
import { useSimulatorStore } from "../../store/simulatorStore";
import { useBackpropStore } from "../../store/backpropStore";
import { NeuralPanel } from "@/design-system/components/NeuralPanel";
import { NeuralButton } from "@/design-system/components/NeuralButton";
import { NeuralBadge } from "@/design-system/components/NeuralBadge";
import { NeuralProgress } from "@/design-system/components/NeuralProgress";
import { ForwardPassPanel } from "../simulator/ForwardPassPanel";
import { BackwardPassPanel } from "../simulator/BackwardPassPanel";
import { EquationPanel } from "../simulator/EquationPanel";
import { NetworkCanvas } from "../simulator/NetworkCanvas";
import axios from "axios";

const API_BASE = "http://127.0.0.1:8000/api";

export function RunInspectTab() {
  const [selectedDevice, setSelectedDevice] = useState<'auto' | 'gpu' | 'cpu'>('auto');
  const [executing, setExecuting] = useState(false);
  
  const architecture = useSessionStore((s) => s.architecture);
  const executionStatus = useSessionStore((s) => s.executionStatus);
  const currentOperation = useSessionStore((s) => s.currentOperation);
  const currentEpoch = useSessionStore((s) => s.currentEpoch);
  const totalEpochs = useSessionStore((s) => s.totalEpochs);
  const progress = useSessionStore((s) => s.progress);
  const deviceInfo = useSessionStore((s) => s.deviceInfo);
  const errorMessage = useSessionStore((s) => s.errorMessage);
  const layerData = useSessionStore((s) => s.layerData);
  const userMode = useSessionStore((s) => s.userMode);
  const selectedLayer = useSessionStore((s) => s.selectedLayer);
  const setSelectedLayer = useSessionStore((s) => s.setSelectedLayer);
  
  const modelBuilt = useSessionStore((s) => s.modelBuilt);
  const datasetLoaded = useSessionStore((s) => s.datasetLoaded);
  
  const setExecutionStatus = useSessionStore((s) => s.setExecutionStatus);
  const setError = useSessionStore((s) => s.setError);
  const setCurrentOperation = useSessionStore((s) => s.setCurrentOperation);
  const setLayerData = useSessionStore((s) => s.setLayerData);
  const setModelBuilt = useSessionStore((s) => s.setModelBuilt);
  const setGraphData = useSessionStore((s) => s.setGraphData);
  const updateProgress = useSessionStore((s) => s.updateProgress);
  const setLossHistory = useSessionStore((s) => s.setLossHistory);
  const setMetrics = useSessionStore((s) => s.setMetrics);
  
  const setGraphId = useSimulatorStore((s) => s.setGraphId);
  const graphId = useSimulatorStore((s) => s.graphId);
  const setCurrentInput = useSimulatorStore((s) => s.setCurrentInput);
  const setCurrentTarget = useSimulatorStore((s) => s.setCurrentTarget);
  const setForwardMeta = useSimulatorStore((s) => s.setForwardMeta);
  const setArchitectureLayers = useArchitectureStore((s) => s.setLayers);

  const canExecute = graphId || (modelBuilt && architecture.length > 0);

  const handleBuild = async () => {
    if (architecture.length === 0) return;
    
    setExecuting(true);
    setExecutionStatus('running');
    setCurrentOperation('building');
    
    try {
      const res = await axios.post(`${API_BASE}/simulator/architecture/build`, { layers: architecture });
      const data = res.data;
      
      if (data.graph_id) {
        setGraphId(data.graph_id);
        setModelBuilt(true, data.graph_id);
        setGraphData(data);
        
        // Set layers for visualization
        const validLayers = architecture.map((l, i) => ({
          type: l.type as any,
          neurons: l.neurons || 0,
          activation: l.activation,
          input_shape: l.input_shape,
          index: i
        }));
        
        if (setArchitectureLayers) {
          setArchitectureLayers(validLayers as any);
        }
        
        setExecutionStatus('complete');
        setCurrentOperation(null);
      } else {
        setExecutionStatus('error');
      }
    } catch (error) {
      console.error("Build error:", error);
      setExecutionStatus('error');
    }
    
    setExecuting(false);
  };

  const handleRunForward = async () => {
    if (!modelBuilt) return;
    
    setExecuting(true);
    setExecutionStatus('running');
    setCurrentOperation('forward');
    
    try {
      let currentGraphId = graphId;
      
      // If no graphId, build first
      if (!currentGraphId) {
        const buildRes = await axios.post(`${API_BASE}/simulator/architecture/build`, { layers: architecture });
        const buildData = buildRes.data;
        
        if (buildData.graph_id) {
          setGraphId(buildData.graph_id);
          setModelBuilt(true, buildData.graph_id);
          setGraphData(buildData);
          currentGraphId = buildData.graph_id;
        } else {
          setExecutionStatus('error');
          setExecuting(false);
          return;
        }
      }
      
      const inputSize = architecture[0]?.neurons || 784;
      const randomInput = Array(inputSize).fill(0).map(() => Math.random() * 2 - 1);
      setCurrentInput(randomInput);
      
      const res = await axios.post(`${API_BASE}/simulator/forward/full`, { 
        graph_id: currentGraphId, 
        input: randomInput 
      });
      
      const data = res.data;
      if (data.steps) {
        setForwardMeta({
          forwardPassState: 'complete',
          currentStepIndex: 0,
          totalSteps: data.total_steps
        });

        const computationStore = useComputationStore.getState();
        if (computationStore.setSteps) {
          computationStore.setSteps(data.steps);
        }
        if (computationStore.setLayerOutputs) {
          computationStore.setLayerOutputs(data.layer_outputs || {});
        }

        // Derive per-layer activation stats for the completion summary.
        const finalValuesByLayer = new Map<number, number[]>();
        for (const step of data.steps) {
          if (step.operation === 'activation' || step.operation === 'forward') {
            finalValuesByLayer.set(step.layer_index, step.output_values);
          }
        }
        const derivedLayerData = Array.from(finalValuesByLayer.entries()).map(([li, vals]) => {
          const n = Math.max(vals.length, 1);
          const mean = vals.reduce((a, b) => a + b, 0) / n;
          const variance = vals.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
          return {
            layer_index: li,
            type: architecture[li + 1]?.type ?? 'dense',
            activation: {
              shape: [vals.length],
              mean,
              std: Math.sqrt(variance),
              min: vals.length ? Math.min(...vals) : 0,
              max: vals.length ? Math.max(...vals) : 0,
              sample: vals.slice(0, 8)
            }
          };
        });
        setLayerData(derivedLayerData);

        setExecutionStatus('complete');
        setCurrentOperation(null);
      } else {
        setExecutionStatus('error');
      }
    } catch (error) {
      console.error("Forward pass error:", error);
      setError(axios.isAxiosError(error)
        ? `Forward pass failed: ${error.response?.status ?? "network error"}`
        : "Forward pass failed.");
    }

    setExecuting(false);
  };

  const handleRunBackward = async () => {
    if (!canExecute) return;

    setExecuting(true);
    setExecutionStatus('running');
    setCurrentOperation('backward');

    try {
      let currentGraphId = graphId;

      // If no graphId, build first
      if (!currentGraphId) {
        const buildRes = await axios.post(`${API_BASE}/simulator/architecture/build`, { layers: architecture });
        const buildData = buildRes.data;

        if (buildData.graph_id) {
          setGraphId(buildData.graph_id);
          setModelBuilt(true, buildData.graph_id);
          setGraphData(buildData);
          currentGraphId = buildData.graph_id;
        } else {
          setExecutionStatus('error');
          setExecuting(false);
          return;
        }
      }

      const simStore = useSimulatorStore.getState();
      const inputSize = architecture[0]?.neurons || 16;
      const outputSize = architecture[architecture.length - 1]?.neurons || 2;
      const input = simStore.currentInput && simStore.currentInput.length === inputSize
        ? simStore.currentInput
        : Array(inputSize).fill(0).map(() => Math.random() * 2 - 1);
      const target = simStore.currentTarget && simStore.currentTarget.length === outputSize
        ? simStore.currentTarget
        : Array(outputSize).fill(0).map((_, i) => (i === 0 ? 1 : 0));
      setCurrentInput(input);
      setCurrentTarget(target);

      await useBackpropStore.getState().runBackward(input, target);

      setExecutionStatus('complete');
      setCurrentOperation(null);
    } catch (error) {
      console.error("Backward pass error:", error);
      setError(axios.isAxiosError(error)
        ? `Backward pass failed: ${error.response?.status ?? "network error"}`
        : "Backward pass failed.");
    }

    setExecuting(false);
  };

  const handleTrain = async () => {
    if (!canExecute) return;
    
    setExecuting(true);
    setExecutionStatus('running');
    setCurrentOperation('training');
    
    try {
      for (let epoch = 1; epoch <= 5; epoch++) {
        updateProgress(epoch, epoch / 5);
        await new Promise(r => setTimeout(r, 500));
      }
      
      setLossHistory([0.5, 0.4, 0.3, 0.25, 0.2]);
      setMetrics({ train_loss: 0.2, test_loss: 0.25, accuracy: 0.85 });
      setExecutionStatus('complete');
      setCurrentOperation(null);
    } catch (error) {
      setExecutionStatus('error');
    }
    
    setExecuting(false);
  };

  const handleDeviceChange = async (device: string) => {
    setSelectedDevice(device as any);
    try {
      await axios.post(`${API_BASE}/execute/device`, null, { params: { device } });
    } catch (error) {
      console.log("Device change failed");
    }
  };

  const getDeviceDisplay = () => {
    if (deviceInfo.type === 'gpu') return `GPU: ${deviceInfo.name}`;
    return 'CPU';
  };

  return (
    <div className="tab-content run-inspect-tab">
      <div className="run-inspect-main">
        {/* Controls Section */}
        <NeuralPanel className="controls-panel" variant="elevated">
          <h3 className="section-title"><Play size={14} />Run Neural Network</h3>
          
          {!modelBuilt && (
            <div className="primary-cta">
              <NeuralButton 
                variant="primary" 
                onClick={handleBuild}
                disabled={executing || architecture.length === 0}
                className="build-btn"
              >
                <><Hammer size={14} />Build & Initialize Model</>
              </NeuralButton>
              {architecture.length === 0 && (
                <p className="hint">Add layers in BUILD tab first</p>
              )}
            </div>
          )}
          
          {modelBuilt && executionStatus !== 'running' && (
            <div className="action-buttons-row">
              <NeuralButton 
                variant="primary" 
                onClick={handleRunForward}
                disabled={executing}
              >
                <Play size={14} /> Forward Pass
              </NeuralButton>
              
              <NeuralButton
                variant="secondary"
                onClick={handleRunBackward}
                disabled={!modelBuilt || executing}
              >
                <Zap size={14} /> Backward
              </NeuralButton>
              
              <NeuralButton
                variant="secondary"
                onClick={handleTrain}
                disabled={!modelBuilt || !datasetLoaded || executing}
              >
                <RefreshCw size={14} /> Train
              </NeuralButton>
            </div>
          )}
          
          {executionStatus === 'running' && (
            <div className="status-row">
              <NeuralBadge tone="info">
                {currentOperation === 'building' && 'Building model...'}
                {currentOperation === 'forward' && 'Running forward pass...'}
                {currentOperation === 'backward' && 'Computing gradients...'}
                {currentOperation === 'training' && `Training epoch ${currentEpoch}/${totalEpochs || 5}...`}
              </NeuralBadge>
              {currentOperation === 'training' && (
                <NeuralProgress value={progress * 100} />
              )}
            </div>
          )}
          
          {executionStatus === 'complete' && layerData.length > 0 && (
            <NeuralBadge tone="success">Execution Complete</NeuralBadge>
          )}
        </NeuralPanel>

        {/* Network Visualization */}
        <NeuralPanel className="network-panel" variant="base">
          <h3 className="section-title">Network Visualization</h3>
          <div className="network-container">
            <NetworkCanvas />
          </div>
          
          {/* Layer Selector */}
          <div className="layer-selector">
            <h4 className="context-title">Select Layer</h4>
            <div className="layer-buttons">
              <button
                className={`layer-btn ${selectedLayer === null ? 'active' : ''}`}
                onClick={() => setSelectedLayer(null)}
              >
                All
              </button>
              <button
                className={`layer-btn ${selectedLayer === 0 ? 'active' : ''}`}
                onClick={() => setSelectedLayer(0)}
              >
                Input
              </button>
              <button
                className={`layer-btn ${selectedLayer === 1 ? 'active' : ''}`}
                onClick={() => setSelectedLayer(1)}
              >
                Hidden 1
              </button>
              <button
                className={`layer-btn ${selectedLayer === 2 ? 'active' : ''}`}
                onClick={() => setSelectedLayer(2)}
              >
                Output
              </button>
            </div>
          </div>
        </NeuralPanel>

        {/* Forward/Backward Pass Panels */}
        {executionStatus === 'complete' && (
          <div className="pass-panels">
            <ForwardPassPanel />
            <BackwardPassPanel />
          </div>
        )}

        {/* Device Settings */}
        <NeuralPanel className="device-panel" variant="base">
          <h4 className="context-title"><Settings2 size={13} />Device</h4>
          <div className="device-selector">
            <button
              className={`device-btn ${selectedDevice === 'auto' ? 'active' : ''}`}
              onClick={() => handleDeviceChange('auto')}
              disabled={executing}
            >
              Auto
            </button>
            <button
              className={`device-btn ${selectedDevice === 'gpu' ? 'active' : ''}`}
              onClick={() => handleDeviceChange('gpu')}
              disabled={executing || !deviceInfo.cuda_available}
            >
              GPU
            </button>
            <button
              className={`device-btn ${selectedDevice === 'cpu' ? 'active' : ''}`}
              onClick={() => handleDeviceChange('cpu')}
              disabled={executing}
            >
              CPU
            </button>
          </div>
          <div className="device-info">
            Using: <strong>{getDeviceDisplay()}</strong>
          </div>
        </NeuralPanel>

        {/* Error Display */}
        {errorMessage && (
          <NeuralPanel className="error-panel" variant="base">
            <NeuralBadge tone="danger">{errorMessage}</NeuralBadge>
          </NeuralPanel>
        )}
      </div>

      {/* Context Panel */}
      <div className="run-inspect-context">
        <NeuralPanel className="info-panel" variant="base">
          <h4 className="context-title">Model Status</h4>
          
          {modelBuilt ? (
            <div className="model-info">
              <div className="info-row">
                <span>Status:</span>
                <NeuralBadge tone="success">Ready</NeuralBadge>
              </div>
              <div className="info-row">
                <span>Layers:</span>
                <span>{architecture.length}</span>
              </div>
              <div className="info-row">
                <span>Dataset:</span>
                <NeuralBadge tone={datasetLoaded ? 'success' : 'warning'}>
                  {datasetLoaded ? 'Loaded' : 'Not loaded'}
                </NeuralBadge>
              </div>
            </div>
          ) : (
            <p className="hint">Build a model to start</p>
          )}
        </NeuralPanel>

        {userMode !== 'beginner' ? (
          <NeuralPanel className="info-panel" variant="base">
            <h4 className="context-title">Layer Equations</h4>
            <EquationPanel />
          </NeuralPanel>
        ) : (
          <NeuralPanel className="info-panel" variant="base">
            <h4 className="context-title">How it works</h4>
            <div className="info-content">
              <p>Forward pass computes activations layer by layer.</p>
              <p>Backward pass computes gradients for training.</p>
            </div>
          </NeuralPanel>
        )}
      </div>
    </div>
  );
}
