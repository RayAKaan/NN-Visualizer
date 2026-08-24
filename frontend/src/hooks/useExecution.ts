import { useCallback } from 'react';
import { useSessionStore } from '../store/sessionStore';
import { useComputationStore } from '../store/computationStore';
import { useTrainingSimStore } from '../store/trainingSimStore';
import { useDatasetStore } from '../store/datasetStore';
import { useSimulatorStore } from '../store/simulatorStore';
import axios from 'axios';

const API_BASE = 'http://127.0.0.1:8000/api';

export function useExecution() {
  const setExecutionStatus = useSessionStore((s) => s.setExecutionStatus);
  const setCurrentOperation = useSessionStore((s) => s.setCurrentOperation);
  const updateProgress = useSessionStore((s) => s.updateProgress);
  const setError = useSessionStore((s) => s.setError);
  const setModelBuilt = useSessionStore((s) => s.setModelBuilt);
  const setMetrics = useSessionStore((s) => s.setMetrics);
  const hyperparameters = useSessionStore((s) => s.hyperparameters);
  const architecture = useSessionStore((s) => s.architecture);
  
  const datasetId = useDatasetStore((s) => s.datasetId);
  
  const graphId = useSimulatorStore((s) => s.graphId);
  const setGraphId = useSimulatorStore((s) => s.setGraphId);
  const currentInput = useSimulatorStore((s) => s.currentInput);
  const currentTarget = useSimulatorStore((s) => s.currentTarget);
  
  const pushMetrics = useTrainingSimStore((s) => s.pushMetrics);
  const setIsTraining = useTrainingSimStore((s) => s.setIsTraining);

  const selectDevice = useCallback(async (preference: string) => {
    try {
      const response = await axios.post(`${API_BASE}/device/select`, null, {
        params: { preference }
      });
      return response.data;
    } catch (error) {
      console.error('Device selection failed:', error);
      return { success: false, message: 'Device selection failed' };
    }
  }, []);

  const buildModel = useCallback(async () => {
    if (architecture.length === 0) {
      setError('No architecture defined');
      return false;
    }
    
    try {
      setExecutionStatus('running');
      setCurrentOperation('building');
      
      const response = await axios.post(`${API_BASE}/simulator/architecture/build`, {
        layers: architecture
      });
      
      if (response.data.graph_id) {
        setGraphId(response.data.graph_id);
        setModelBuilt(true, response.data.graph_id);
        setExecutionStatus('complete');
        setCurrentOperation(null);
        return true;
      } else {
        setError('Failed to build model');
        setExecutionStatus('error');
        return false;
      }
    } catch (error: any) {
      setError(error.response?.data?.detail || 'Build failed');
      setExecutionStatus('error');
      return false;
    }
  }, [architecture, setExecutionStatus, setCurrentOperation, setError, setGraphId, setModelBuilt]);

  const runForward = useCallback(async () => {
    // First ensure model is built
    if (!graphId) {
      const built = await buildModel();
      if (!built) return;
    }
    
    if (!currentInput) {
      // Use random input if none provided
      const input = Array(784).fill(0).map(() => Math.random() * 2 - 1);
      
      try {
        setExecutionStatus('running');
        setCurrentOperation('forward');
        
        const response = await axios.post(`${API_BASE}/simulator/forward/full`, {
          graph_id: graphId,
          input: input
        });
        
        if (response.data.steps) {
          setExecutionStatus('complete');
          setCurrentOperation(null);
        }
      } catch (error: any) {
        setError(error.response?.data?.detail || 'Forward pass failed');
        setExecutionStatus('error');
      }
    }
  }, [graphId, currentInput, buildModel, setExecutionStatus, setCurrentOperation, setError]);

  const runBackward = useCallback(async () => {
    if (!graphId) {
      setError('Model not built');
      return;
    }
    
    if (!currentInput || !currentTarget) {
      setError('No input/target available. Run forward pass first.');
      return;
    }
    
    try {
      setExecutionStatus('running');
      setCurrentOperation('backward');
      
      const response = await axios.post(`${API_BASE}/simulator/backward/full`, {
        graph_id: graphId,
        input: currentInput,
        target: currentTarget,
        loss_function: hyperparameters.loss_function || 'bce',
        l2_lambda: hyperparameters.l2_lambda || 0
      });
      
      if (response.data.gradients_W) {
        setExecutionStatus('complete');
        setCurrentOperation(null);
      }
    } catch (error: any) {
      setError(error.response?.data?.detail || 'Backward pass failed');
      setExecutionStatus('error');
    }
  }, [graphId, currentInput, currentTarget, hyperparameters, setExecutionStatus, setCurrentOperation, setError]);

  const train = useCallback(async () => {
    if (!graphId) {
      const built = await buildModel();
      if (!built) return;
    }
    
    if (!datasetId) {
      setError('No dataset loaded');
      return;
    }
    
    try {
      setExecutionStatus('running');
      setCurrentOperation('training');
      setIsTraining(true);
      
      const response = await axios.post(`${API_BASE}/simulator/train/start`, {
        graph_id: graphId,
        dataset_id: datasetId,
        config: {
          epochs: hyperparameters.epochs || 10,
          batch_size: hyperparameters.batch_size || 32,
          learning_rate: hyperparameters.learning_rate || 0.001,
          optimizer: hyperparameters.optimizer || 'adam',
          loss_function: hyperparameters.loss_function || 'bce'
        }
      });
      
      // Simulate training progress (in real implementation, use WebSocket)
      for (let epoch = 1; epoch <= (hyperparameters.epochs || 10); epoch++) {
        updateProgress(epoch, epoch / (hyperparameters.epochs || 10));
        
        // Add mock metrics for demo
        pushMetrics(epoch, {
          train_loss: Math.random() * 0.5,
          test_loss: Math.random() * 0.5,
          train_accuracy: Math.random() * 0.3 + 0.7,
          test_accuracy: Math.random() * 0.3 + 0.6,
          learning_rate: hyperparameters.learning_rate || 0.001,
          gradient_norms: [],
          weight_norms: [],
          weight_deltas: [],
          dead_neurons: [],
          epoch_duration_ms: 100
        });
        
        await new Promise(resolve => setTimeout(resolve, 500));
      }
      
      setMetrics({
        train_loss: 0.1,
        test_loss: 0.15,
        train_accuracy: 0.95,
        test_accuracy: 0.92
      });
      
      setExecutionStatus('complete');
      setCurrentOperation(null);
      setIsTraining(false);
    } catch (error: any) {
      setError(error.response?.data?.detail || 'Training failed');
      setExecutionStatus('error');
      setIsTraining(false);
    }
  }, [graphId, datasetId, hyperparameters, buildModel, setExecutionStatus, setCurrentOperation, setError, setIsTraining, updateProgress, pushMetrics, setMetrics]);

  const stop = useCallback(async () => {
    setExecutionStatus('idle');
    setCurrentOperation(null);
    setIsTraining(false);
  }, [setExecutionStatus, setCurrentOperation, setIsTraining]);

  return {
    buildModel,
    runForward,
    runBackward,
    train,
    stop,
    selectDevice
  };
}