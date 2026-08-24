import { create } from 'zustand';

export type TabId = 'build' | 'run' | 'inspect' | 'analyze' | 'advanced';
export type UserMode = 'beginner' | 'standard' | 'research';
export type ExecutionStatus = 'idle' | 'running' | 'complete' | 'error';
export type DeviceType = 'gpu' | 'cpu';

interface DeviceInfo {
  type: DeviceType | string;
  name: string;
  memory_total: number;
  memory_available: number;
  cuda_available: boolean;
  mps_available: boolean;
}

interface LayerConfig {
  type: string;
  neurons?: number;
  activation?: string;
  input_shape?: number[];
  [key: string]: any;
}

interface Hyperparameters {
  learning_rate: number;
  batch_size: number;
  epochs: number;
  optimizer: string;
  loss_function: string;
  l2_lambda: number;
}

interface DatasetInfo {
  name: string;
  dataset_id?: string;
  train_samples: number;
  test_samples: number;
  input_shape: number[];
  output_shape: number[];
}

interface LayerData {
  layer_index: number;
  type: string;
  activation?: {
    shape: number[];
    mean: number;
    std: number;
    min: number;
    max: number;
    sample: number[];
  };
  gradient?: {
    shape: number[];
    mean: number;
    std: number;
    norm: number;
  };
  weights?: {
    shape: number[];
    stats: Record<string, number>;
  };
}

interface GraphData {
  graph_id: string;
  total_params: number;
  layers: number;
  device: string;
  weights?: number[][][];
  biases?: number[][];
  weight_stats?: Array<{
    layer: number;
    mean: number;
    std: number;
    min: number;
    max: number;
  }>;
}

interface SimulationState {
  // Model
  modelBuilt: boolean;
  graphId: string | null;
  graphData: GraphData | null;
  architecture: LayerConfig[];
  hyperparameters: Hyperparameters;
  
  // Execution
  executionStatus: ExecutionStatus;
  currentOperation: string | null;
  currentEpoch: number;
  totalEpochs: number;
  progress: number;
  errorMessage: string | null;
  device: DeviceType;
  deviceInfo: DeviceInfo;
  
  // Results
  layerData: LayerData[];
  lossHistory: number[];
  metrics: Record<string, number>;
  
  // Dataset
  datasetLoaded: boolean;
  dataset: DatasetInfo | null;
  
  // UI
  activeTab: TabId;
  selectedLayer: number | null;
  userMode: UserMode;
  contextPanelCollapsed: boolean;
  
  // History
  history: Array<{ architecture: LayerConfig[]; hyperparameters: Hyperparameters }>;
  historyIndex: number;
  
  // Actions
  setModelBuilt: (built: boolean, graphId?: string) => void;
  setGraphData: (data: GraphData | null) => void;
  setArchitecture: (layers: LayerConfig[]) => void;
  setHyperparameters: (params: Partial<Hyperparameters>) => void;
  setDataset: (dataset: DatasetInfo | null) => void;
  setExecutionStatus: (status: ExecutionStatus) => void;
  setCurrentOperation: (operation: string | null) => void;
  updateProgress: (epoch: number, progress: number) => void;
  setError: (message: string | null) => void;
  setDevice: (device: DeviceType) => void;
  setDeviceInfo: (info: DeviceInfo) => void;
  setActiveTab: (tab: TabId) => void;
  setSelectedLayer: (layerIndex: number | null) => void;
  setUserMode: (mode: UserMode) => void;
  setLayerData: (data: LayerData[]) => void;
  setLossHistory: (history: number[]) => void;
  setMetrics: (metrics: Record<string, number>) => void;
  toggleContextPanel: () => void;
  saveToHistory: () => void;
  undo: () => void;
  redo: () => void;
  reset: () => void;
  
  // Computed
  getNextAction: () => string;
  isReady: () => boolean;
}

const initialState = {
  modelBuilt: false,
  graphId: null,
  graphData: null,
  architecture: [] as LayerConfig[],
  hyperparameters: {
    learning_rate: 0.001,
    batch_size: 32,
    epochs: 10,
    optimizer: 'adam',
    loss_function: 'bce',
    l2_lambda: 0.0,
  },
  
  executionStatus: 'idle' as ExecutionStatus,
  currentOperation: null,
  currentEpoch: 0,
  totalEpochs: 0,
  progress: 0,
  errorMessage: null,
  device: 'cpu' as DeviceType,
  deviceInfo: {
    type: 'cpu',
    name: 'CPU',
    memory_total: 0,
    memory_available: 0,
    cuda_available: false,
    mps_available: false,
  },
  
  layerData: [] as LayerData[],
  lossHistory: [] as number[],
  metrics: {},
  
  datasetLoaded: false,
  dataset: null as DatasetInfo | null,
  
  activeTab: 'build' as TabId,
  selectedLayer: null,
  userMode: 'standard' as UserMode,
  contextPanelCollapsed: false,
  
  history: [],
  historyIndex: -1,
};

export const useSessionStore = create<SimulationState>((set, get) => ({
  ...initialState,
  
  setModelBuilt: (built, graphId) => set({ 
    modelBuilt: built, 
    graphId: graphId || null 
  }),
  
  setGraphData: (data) => set({ graphData: data }),
  
  setArchitecture: (layers) => set({ architecture: layers }),
  
  setHyperparameters: (params) => set((state) => ({
    hyperparameters: { ...state.hyperparameters, ...params }
  })),
  
  setDataset: (dataset) => set({ 
    datasetLoaded: !!dataset, 
    dataset 
  }),
  
  setExecutionStatus: (status) => set({ executionStatus: status }),
  
  setCurrentOperation: (operation) => set({ currentOperation: operation }),
  
  updateProgress: (epoch, progress) => set({ 
    currentEpoch: epoch, 
    progress 
  }),
  
  setError: (message) => set({ 
    errorMessage: message,
    executionStatus: message ? 'error' : get().executionStatus
  }),
  
  setDevice: (device) => set({ device }),
  
  setDeviceInfo: (info) => {
    const deviceType: DeviceType = (info.type === 'gpu' || info.type === 'cpu') ? info.type : 'cpu';
    set({ deviceInfo: info, device: deviceType });
  },
  
  setActiveTab: (tab) => set({ activeTab: tab }),
  
  setSelectedLayer: (layerIndex) => set({ selectedLayer: layerIndex }),
  
  setUserMode: (mode) => set({ userMode: mode }),
  
  setLayerData: (data) => set({ layerData: data }),
  
  setLossHistory: (history) => set({ lossHistory: history }),
  
  setMetrics: (metrics) => set({ metrics }),
  
  toggleContextPanel: () => set((state) => ({ 
    contextPanelCollapsed: !state.contextPanelCollapsed 
  })),
  
  saveToHistory: () => set((state) => {
    const newHistory = state.history.slice(0, state.historyIndex + 1);
    newHistory.push({
      architecture: state.architecture,
      hyperparameters: state.hyperparameters,
    });
    if (newHistory.length > 10) newHistory.shift();
    return {
      history: newHistory as SimulationState[],
      historyIndex: newHistory.length - 1
    };
  }),
  
  undo: () => set((state) => {
    if (state.historyIndex > 0) {
      return {
        architecture: state.history[state.historyIndex - 1].architecture,
        hyperparameters: state.history[state.historyIndex - 1].hyperparameters,
        historyIndex: state.historyIndex - 1
      };
    }
    return {};
  }),
  
  redo: () => set((state) => {
    if (state.historyIndex < state.history.length - 1) {
      return {
        architecture: state.history[state.historyIndex + 1].architecture,
        hyperparameters: state.history[state.historyIndex + 1].hyperparameters,
        historyIndex: state.historyIndex + 1
      };
    }
    return {};
  }),
  
  reset: () => set({
    modelBuilt: false,
    graphId: null,
    graphData: null,
    architecture: [],
    hyperparameters: initialState.hyperparameters,
    executionStatus: 'idle',
    currentOperation: null,
    currentEpoch: 0,
    totalEpochs: 0,
    progress: 0,
    errorMessage: null,
    layerData: [],
    lossHistory: [],
    metrics: {},
    datasetLoaded: false,
    dataset: null,
    activeTab: 'build',
    selectedLayer: null,
    contextPanelCollapsed: false,
    history: [],
    historyIndex: -1,
  }),
  
  getNextAction: () => {
    const state = get();
    if (!state.modelBuilt) return 'Click "Build Model" to start';
    if (!state.datasetLoaded) return 'Select a dataset in BUILD tab';
    if (state.executionStatus === 'idle') return '▶️ Click "Run Forward" to execute';
    if (state.executionStatus === 'complete') return '🔍 View results in INSPECT tab';
    return 'Processing...';
  },
  
  isReady: () => {
    const state = get();
    return state.modelBuilt && state.datasetLoaded;
  },
}));

export type { LayerConfig, Hyperparameters, DatasetInfo, DeviceInfo };