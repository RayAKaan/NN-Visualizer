import { apiClient } from "../api/client";
import type {
  ActivationInspection,
  BuildResponse,
  DatasetResponse,
  BackwardFullResponse,
  EquationResponse,
  ForwardFullResponse,
  ForwardStepResponse,
  FeatureMapResponse,
  SaliencyResponse,
  FilterResponse,
  NeuronAtlasResponse,
  SequenceStepResponse,
  SequenceFullResponse,
  CompareSetupResponse,
  CompareResultsResponse,
  ProfileResponse,
  LandscapeComputeResponse,
  LandscapeStatusResponse,
  EmbeddingResponse,
  IntegratedGradientsResponse,
  InterpretStubResponse,
  AdversarialAttackResponse,
  RobustnessCurveResponse,
  PruneResponse,
  QuantizeResponse,
  PruneSweepResponse,
  GenerativeSampleResponse,
  AugmentationPreviewResponse,
  ExperimentsListResponse,
  ExperimentRecord,
  AssistantResponse,
  ShapResponse,
  LimeResponse,
  LrpResponse,
  LayerConfig,
  TrainingMetrics,
  ValidationResult,
  WeightInspection,
} from "../types/simulator";

export const simulatorApi = {
  validateArchitecture: async (layers: LayerConfig[]) => {
    const res = await apiClient.post<ValidationResult>("/api/simulator/architecture/validate", { layers });
    return res.data;
  },
  buildArchitecture: async (layers: LayerConfig[]) => {
    const res = await apiClient.post<BuildResponse>("/api/simulator/architecture/build", { layers });
    return res.data;
  },
  forwardFull: async (graphId: string, input: number[]) => {
    const res = await apiClient.post<ForwardFullResponse>("/api/simulator/forward/full", { graph_id: graphId, input });
    return res.data;
  },
  forwardStep: async (graphId: string, input: number[], stepIndex: number) => {
    const res = await apiClient.post<ForwardStepResponse>("/api/simulator/forward/step", {
      graph_id: graphId,
      input,
      step_index: stepIndex,
    });
    return res.data;
  },
  equationsLayer: async (graphId: string, layerIndex: number, includeNumeric: boolean) => {
    const res = await apiClient.post<EquationResponse>("/api/simulator/equations/layer", {
      graph_id: graphId,
      layer_index: layerIndex,
      include_numeric: includeNumeric,
    });
    return res.data;
  },
  datasetGenerate: async (payload: { type: string; n_samples: number; noise: number; train_split: number; seed?: number | null }) => {
    const res = await apiClient.post<DatasetResponse>("/api/simulator/dataset/generate", payload);
    return res.data;
  },
  datasetLoadStandard: async (payload: { name: string; n_samples: number; train_split: number; seed?: number | null }) => {
    const res = await apiClient.post("/api/simulator/dataset/load_standard", payload);
    return res.data;
  },
  datasetGenerateSequence: async (payload: {
    type: string;
    n_samples: number;
    seq_length: number;
    n_features: number;
    vocab_size: number;
    n_classes: number;
    noise: number;
    train_split: number;
  }) => {
    const res = await apiClient.post("/api/simulator/dataset/generate_sequence", payload);
    return res.data;
  },
  datasetCustom: async (payload: { points: Array<{ x: number[]; y: number[] }>; train_split: number }) => {
    const res = await apiClient.post<DatasetResponse>("/api/simulator/dataset/custom", payload);
    return res.data;
  },
  datasetUploadImage: async (file: File, targetSize = 28) => {
    const form = new FormData();
    form.append("file", file);
    const res = await apiClient.post("/api/simulator/dataset/upload_image", form, {
      params: { target_size: targetSize },
      headers: { "Content-Type": "multipart/form-data" },
    });
    return res.data;
  },
  inspectWeights: async (graphId: string, layerIndex: number) => {
    const res = await apiClient.post<WeightInspection>("/api/simulator/inspect/weights", { graph_id: graphId, layer_index: layerIndex });
    return res.data;
  },
  inspectActivations: async (graphId: string, layerIndex: number, input: number[]) => {
    const res = await apiClient.post<ActivationInspection>("/api/simulator/inspect/activations", {
      graph_id: graphId,
      layer_index: layerIndex,
      input,
    });
    return res.data;
  },
  backwardFull: async (graphId: string, input: number[], target: number[], lossFunction: string, l2: number) => {
    const res = await apiClient.post<BackwardFullResponse>("/api/simulator/backward/full", {
      graph_id: graphId,
      input,
      target,
      loss_function: lossFunction,
      l2_lambda: l2,
    });
    return res.data;
  },
  backwardStep: async (graphId: string, input: number[], target: number[], lossFunction: string, stepIndex: number) => {
    const res = await apiClient.post("/api/simulator/backward/step", {
      graph_id: graphId,
      input,
      target,
      loss_function: lossFunction,
      step_index: stepIndex,
    });
    return res.data;
  },
  inspectGradients: async (graphId: string, layerIndex: number) => {
    const res = await apiClient.post("/api/simulator/inspect/gradients", { graph_id: graphId, layer_index: layerIndex });
    return res.data;
  },
  gradientFlow: async (graphId: string) => {
    const res = await apiClient.post("/api/simulator/inspect/gradient_flow", { graph_id: graphId });
    return res.data;
  },
  weightHistory: async (graphId: string) => {
    const res = await apiClient.get(`/api/simulator/inspect/weight_history/${graphId}`);
    return res.data;
  },
  debugDiagnose: async (graphId: string) => {
    const res = await apiClient.post("/api/simulator/debug/diagnose", { graph_id: graphId });
    return res.data;
  },
  debugApplyFix: async (graphId: string, fixAction: any) => {
    const res = await apiClient.post("/api/simulator/debug/apply_fix", { graph_id: graphId, fix_action: fixAction });
    return res.data;
  },
  replaySnapshots: async (graphId: string) => {
    const res = await apiClient.get(`/api/simulator/replay/snapshots/${graphId}`);
    return res.data;
  },
  replayLoad: async (graphId: string, snapshotIndex: number) => {
    const res = await apiClient.post("/api/simulator/replay/load", { graph_id: graphId, snapshot_index: snapshotIndex });
    return res.data;
  },
  featureMaps: async (graphId: string, inputImage: number[], inputShape: number[]) => {
    const res = await apiClient.post<FeatureMapResponse>("/api/simulator/activations/feature_maps", {
      graph_id: graphId,
      input_image: inputImage,
      input_shape: inputShape,
    });
    return res.data;
  },
  saliency: async (graphId: string, input: number[], inputShape: number[], targetClass: number, method: string) => {
    const res = await apiClient.post<SaliencyResponse>("/api/simulator/activations/saliency", {
      graph_id: graphId,
      input,
      input_shape: inputShape,
      target_class: targetClass,
      method,
    });
    return res.data;
  },
  filterResponse: async (graphId: string, datasetId: string, layerIndex: number, filterIndex: number, nSamples: number) => {
    const res = await apiClient.post<FilterResponse>("/api/simulator/activations/filter_response", {
      graph_id: graphId,
      dataset_id: datasetId,
      layer_index: layerIndex,
      filter_index: filterIndex,
      n_samples: nSamples,
    });
    return res.data;
  },
  neuronAtlas: async (graphId: string, datasetId: string, layerIndex: number, nSamples: number) => {
    const res = await apiClient.post<NeuronAtlasResponse>("/api/simulator/activations/neuron_atlas", {
      graph_id: graphId,
      dataset_id: datasetId,
      layer_index: layerIndex,
      n_samples: nSamples,
    });
    return res.data;
  },
  sequenceStep: async (graphId: string, sequence: number[][], timestep: number) => {
    const res = await apiClient.post<SequenceStepResponse>("/api/simulator/sequence/step", {
      graph_id: graphId,
      sequence,
      timestep,
    });
    return res.data;
  },
  sequenceFull: async (graphId: string, sequence: number[][]) => {
    const res = await apiClient.post<SequenceFullResponse>("/api/simulator/sequence/full", {
      graph_id: graphId,
      sequence,
    });
    return res.data;
  },
  compareSetup: async (models: any[], datasetId: string, epochs: number) => {
    const res = await apiClient.post<CompareSetupResponse>("/api/simulator/compare/setup", {
      models,
      dataset_id: datasetId,
      epochs,
    });
    return res.data;
  },
  compareResults: async (comparisonId: string) => {
    const res = await apiClient.get<CompareResultsResponse>(`/api/simulator/compare/results/${comparisonId}`);
    return res.data;
  },
  profileFull: async (graphId: string, batchSizes: number[]) => {
    const res = await apiClient.post<ProfileResponse>("/api/simulator/profile/full", {
      graph_id: graphId,
      batch_sizes: batchSizes,
    });
    return res.data;
  },

  landscapeCompute: async (graphId: string, datasetId: string, resolution: number, range: number) => {
    const res = await apiClient.post<LandscapeComputeResponse>("/api/simulator/landscape/compute", {
      graph_id: graphId,
      dataset_id: datasetId,
      resolution,
      range,
    });
    return res.data;
  },
  landscapeStatus: async (taskId: string) => {
    const res = await apiClient.get<LandscapeStatusResponse>(`/api/simulator/landscape/status/${taskId}`);
    return res.data;
  },
  embeddingsCompute: async (graphId: string, datasetId: string, layerIndex: number, nSamples: number, method = "pca") => {
    const res = await apiClient.post<EmbeddingResponse>("/api/simulator/embeddings/compute", {
      graph_id: graphId,
      dataset_id: datasetId,
      layer_index: layerIndex,
      n_samples: nSamples,
      method,
    });
    return res.data;
  },
  integratedGradients: async (graphId: string, input: number[], targetClass: number, nSteps: number) => {
    const res = await apiClient.post<IntegratedGradientsResponse>("/api/simulator/interpret/integrated_gradients", {
      graph_id: graphId,
      input,
      target_class: targetClass,
      n_steps: nSteps,
    });
    return res.data;
  },

  interpretShap: async (graphId: string, input: number[], targetClass: number, baselineSamples: number) => {
    const res = await apiClient.post<ShapResponse>("/api/simulator/interpret/shap", {
      graph_id: graphId,
      input,
      target_class: targetClass,
      baseline_samples: baselineSamples,
    });
    return res.data;
  },
  interpretLime: async (graphId: string, input: number[], inputShape: number[] | null, targetClass: number, nSamples: number) => {
    const res = await apiClient.post<LimeResponse>("/api/simulator/interpret/lime", {
      graph_id: graphId,
      input,
      input_shape: inputShape,
      target_class: targetClass,
      n_samples: nSamples,
    });
    return res.data;
  },
  interpretLrp: async (graphId: string, input: number[], targetClass: number, rule: string, epsilon: number) => {
    const res = await apiClient.post<LrpResponse>("/api/simulator/interpret/lrp", {
      graph_id: graphId,
      input,
      target_class: targetClass,
      rule,
      epsilon,
    });
    return res.data;
  },
  interpretCompare: async (graphId: string, input: number[], targetClass: number, methods: string[]) => {
    const res = await apiClient.post<InterpretStubResponse>("/api/simulator/interpret/compare", {
      graph_id: graphId,
      input,
      target_class: targetClass,
      methods,
    });
    return res.data;
  },

  adversarialAttack: async (payload: { graph_id: string; input: number[]; true_label: number; method: string; epsilon: number; input_shape?: number[] | null; pgd_steps?: number; pgd_step_size?: number }) => {
    const res = await apiClient.post<AdversarialAttackResponse>("/api/simulator/adversarial/attack", payload);
    return res.data;
  },
  adversarialEvaluate: async (payload: { graph_id: string; dataset_id: string; epsilons: number[]; pgd_steps?: number }) => {
    const res = await apiClient.post<RobustnessCurveResponse>("/api/simulator/adversarial/evaluate", payload);
    return res.data;
  },
  compressPrune: async (graphId: string, sparsity: number) => {
    const res = await apiClient.post<PruneResponse>("/api/simulator/compress/prune", {
      graph_id: graphId,
      method: "magnitude",
      sparsity,
    });
    return res.data;
  },
  compressQuantize: async (graphId: string, targetDtype: string) => {
    const res = await apiClient.post<QuantizeResponse>("/api/simulator/compress/quantize", {
      graph_id: graphId,
      target_dtype: targetDtype,
    });
    return res.data;
  },
  compressSweep: async (graphId: string, sparsityRange: number[]) => {
    const res = await apiClient.post<PruneSweepResponse>("/api/simulator/compress/sweep", {
      graph_id: graphId,
      sparsity_range: sparsityRange,
    });
    return res.data;
  },

  generativeTrain: async (datasetId: string, mode: string, epochs: number) => {
    const res = await apiClient.post("/api/simulator/generative/train", {
      dataset_id: datasetId,
      mode,
      epochs,
    });
    return res.data;
  },
  generativeSample: async (mode: string, nSamples: number, size: number) => {
    const res = await apiClient.post<GenerativeSampleResponse>("/api/simulator/generative/sample", {
      mode,
      n_samples: nSamples,
      size,
    });
    return res.data;
  },
  augmentationPreview: async (input: number[], inputShape: number[] | null, nSamples: number, pipeline: Array<Record<string, any>> | null) => {
    const res = await apiClient.post<AugmentationPreviewResponse>("/api/simulator/augmentation/preview", {
      input,
      input_shape: inputShape,
      n_samples: nSamples,
      pipeline,
    });
    return res.data;
  },

  experimentsList: async () => {
    const res = await apiClient.get<ExperimentsListResponse>("/api/simulator/experiments/list");
    return res.data;
  },
  experimentsCreate: async (payload: { name: string; config: Record<string, any>; metrics?: Record<string, any> | null; notes?: string | null; tags?: string[] | null }) => {
    const res = await apiClient.post<ExperimentRecord>("/api/simulator/experiments/create", payload);
    return res.data;
  },
  experimentsDelete: async (experimentId: string) => {
    const res = await apiClient.delete(`/api/simulator/experiments/${experimentId}`);
    return res.data;
  },
  assistantQuery: async (message: string, context?: Record<string, any>) => {
    const res = await apiClient.post<AssistantResponse>("/api/simulator/assistant/query", { message, context });
    return res.data;
  },
  templatesList: async () => {
    const res = await apiClient.get("/api/simulator/templates/list");
    return res.data;
  },
  importUpload: async (file: File, format?: string) => {
    const form = new FormData();
    form.append("file", file);
    if (format) form.append("format", format);
    const res = await apiClient.post("/api/simulator/import/upload", form, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return res.data;
  },
  importBuild: async (importId: string) => {
    const res = await apiClient.post("/api/simulator/import/build", { import_id: importId });
    return res.data;
  },
  exportCode: async (graphId: string, format: string) => {
    const res = await apiClient.post("/api/simulator/export/code", { graph_id: graphId, format });
    return res.data;
  },
  exportImage: async (graphId: string, format: string, width = 1200, height = 600) => {
    const res = await apiClient.post("/api/simulator/export/image", { graph_id: graphId, format, width, height });
    return res.data;
  },
};
