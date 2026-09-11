import { apiClient } from "./client";
import type {
  ConvCellDetail,
  DenseNeuronDetail,
  ExecutionTrace,
  LstmDetail,
  NetworkMeta,
} from "../types/execution";

export async function fetchNetworks(): Promise<Record<string, NetworkMeta>> {
  const res = await apiClient.get("/api/lab/networks");
  return res.data.networks;
}

export interface TraceRequest {
  architecture: string;
  dataset: string;
  pixels: number[];
  useUntrainedWeights?: boolean;
}

export async function runTrace(payload: TraceRequest): Promise<ExecutionTrace> {
  const res = await apiClient.post("/api/lab/trace", payload);
  return res.data;
}

export interface TraceDetailRequest {
  architecture: string;
  dataset: string;
  pixels: number[];
  layerId: string;
  detail: "neuron" | "conv_cell" | "lstm_gates" | "lstm_gate_gradients";
  selection?: {
    neuronIndex?: number;
    filterIndex?: number;
    position?: { h: number; w: number };
    timestep?: number;
  };
  trueLabel?: number;
}

export async function fetchTraceDetail<T = DenseNeuronDetail | ConvCellDetail | LstmDetail>(
  payload: TraceDetailRequest,
): Promise<T> {
  const res = await apiClient.post("/api/lab/trace/detail", payload);
  return res.data;
}