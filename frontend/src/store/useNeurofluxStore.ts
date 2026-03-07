import { create } from "zustand";
import { ArchitectureType, EdgeState, IntrospectionMode, NeuronState } from "../components/neurofluxion/types";

export interface NeuroTopology {
  neurons: NeuronState[];
  edges: EdgeState[];
}

export interface MetricPoint {
  epoch: number;
  loss: number;
  accuracy: number;
}

type PlaybackState = "playing" | "paused";

interface NeurofluxStore {
  mode: IntrospectionMode;
  currentArchitecture: ArchitectureType;
  topology: NeuroTopology;
  selectedNeuronId: string | null;
  hoveredNeuronId: string | null;
  playbackState: PlaybackState;
  currentEpoch: number;
  history: NeuroTopology[];
  metricsHistory: MetricPoint[];
  showNeuronHealth: boolean;
  setMode: (mode: IntrospectionMode) => void;
  setCurrentArchitecture: (arch: ArchitectureType) => void;
  setSelectedNeuron: (id: string | null) => void;
  setHoveredNeuron: (id: string | null) => void;
  setTopology: (topology: NeuroTopology) => void;
  addTopologySnapshot: (topology: NeuroTopology) => void;
  addMetricsSnapshot: (metric: MetricPoint) => void;
  play: () => void;
  pause: () => void;
  stepForward: () => void;
  stepBackward: () => void;
  scrubToEpoch: (epochIndex: number) => void;
  toggleNeuronHealth: () => void;
  perturbWeight: (edgeId: string, newWeight: number) => void;
}

const MAX_HISTORY = 300;
const rand = (min: number, max: number) => min + Math.random() * (max - min);
const pick = <T,>(arr: T[]) => arr[Math.floor(Math.random() * arr.length)];

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const relu = (v: number) => (v > 0 ? v : 0);

const cloneTopology = (topology: NeuroTopology): NeuroTopology => {
  const edges = topology.edges.map((e) => ({ ...e }));
  const edgeById = new Map(edges.map((e) => [e.id, e]));
  const neurons = topology.neurons.map((n) => ({
    ...n,
    incomingEdges: n.incomingEdges.map((e) => edgeById.get(e.id) ?? { ...e }),
    outgoingEdges: n.outgoingEdges.map((e) => edgeById.get(e.id) ?? { ...e }),
  }));
  return { neurons, edges };
};

const recalculateForwardPass = (topology: NeuroTopology): NeuroTopology => {
  const cloned = cloneTopology(topology);
  const neuronById = new Map(cloned.neurons.map((n) => [n.id, n]));
  const indegree = new Map<string, number>();

  cloned.neurons.forEach((n) => indegree.set(n.id, n.incomingEdges.length));

  const queue: string[] = [];
  cloned.neurons.forEach((n) => {
    if ((indegree.get(n.id) ?? 0) === 0 || n.layerType === "input") {
      queue.push(n.id);
    }
  });

  while (queue.length > 0) {
    const id = queue.shift()!;
    const neuron = neuronById.get(id);
    if (!neuron) continue;

    if (neuron.layerType !== "input") {
      const z = neuron.incomingEdges.reduce((sum, edge) => {
        const fromNeuron = neuronById.get(edge.from);
        return sum + (fromNeuron?.activation ?? 0) * edge.weight;
      }, neuron.bias);
      neuron.activation = relu(z);
    }

    neuron.outgoingEdges.forEach((edge) => {
      const outEdge = cloned.edges.find((e) => e.id === edge.id);
      if (outEdge) {
        outEdge.contribution = outEdge.weight * neuron.activation;
      }
      const toId = edge.to;
      const nextIn = (indegree.get(toId) ?? 1) - 1;
      indegree.set(toId, nextIn);
      if (nextIn === 0) queue.push(toId);
    });
  }

  const edgeById = new Map(cloned.edges.map((e) => [e.id, e]));
  cloned.neurons.forEach((n) => {
    n.incomingEdges = n.incomingEdges.map((e) => edgeById.get(e.id) ?? e);
    n.outgoingEdges = n.outgoingEdges.map((e) => edgeById.get(e.id) ?? e);
  });

  return cloned;
};

export function generateMockANN(): NeuroTopology {
  const inputIds = ["I_1", "I_2", "I_3"];
  const hiddenIds = ["H1_1", "H1_2", "H1_3", "H1_4"];
  const outputIds = ["O_1", "O_2"];

  const edges: EdgeState[] = [];
  const neuronsMap = new Map<string, NeuronState>();

  const makeNeuron = (id: string, layerType: NeuronState["layerType"]): NeuronState => ({
    id,
    layerType,
    activation: rand(0.05, 0.95),
    bias: layerType === "input" ? 0 : rand(-0.35, 0.35),
    gradient: rand(-0.03, 0.03),
    incomingEdges: [],
    outgoingEdges: [],
  });

  [...inputIds.map((id) => makeNeuron(id, "input")), ...hiddenIds.map((id) => makeNeuron(id, "dense")), ...outputIds.map((id) => makeNeuron(id, "dense"))]
    .forEach((n) => neuronsMap.set(n.id, n));

  const connect = (fromIds: string[], toIds: string[]) => {
    fromIds.forEach((from) => {
      toIds.forEach((to) => {
        const fromNeuron = neuronsMap.get(from)!;
        const weight = rand(-1.2, 1.2);
        const gradient = rand(-0.04, 0.04);
        const contribution = weight * fromNeuron.activation;
        const edge: EdgeState = {
          id: `${from}_to_${to}`,
          from,
          to,
          weight,
          gradient,
          contribution,
        };
        edges.push(edge);
        neuronsMap.get(from)!.outgoingEdges.push(edge);
        neuronsMap.get(to)!.incomingEdges.push(edge);
      });
    });
  };

  connect(inputIds, hiddenIds);
  connect(hiddenIds, outputIds);
  outputIds.forEach((id) => {
    const n = neuronsMap.get(id)!;
    n.activation = pick([rand(0.05, 0.25), rand(0.65, 0.95)]);
  });

  return {
    neurons: Array.from(neuronsMap.values()),
    edges,
  };
}

export function generateMockCNN(): NeuroTopology {
  const neurons: NeuronState[] = [];
  const edges: EdgeState[] = [];
  const idMap = new Map<string, NeuronState>();

  const add = (id: string, layerType: NeuronState["layerType"], activation: number, bias = 0, gradient = 0) => {
    const n: NeuronState = { id, layerType, activation, bias, gradient, incomingEdges: [], outgoingEdges: [] };
    neurons.push(n);
    idMap.set(id, n);
  };

  const inSize = 6;
  const convSize = 4;
  for (let r = 0; r < inSize; r++) {
    for (let c = 0; c < inSize; c++) {
      add(`C0_${r}_${c}`, "input", rand(0.05, 0.95), 0, 0);
    }
  }
  for (let r = 0; r < convSize; r++) {
    for (let c = 0; c < convSize; c++) {
      add(`C1_${r}_${c}`, "conv", rand(0.05, 0.95), rand(-0.12, 0.12), rand(-0.02, 0.02));
    }
  }
  add("O_1", "dense", rand(0.2, 0.95), rand(-0.05, 0.05), rand(-0.03, 0.03));
  add("O_2", "dense", rand(0.2, 0.95), rand(-0.05, 0.05), rand(-0.03, 0.03));

  const kernel = [0.18, -0.04, 0.11, 0.09, 0.22, -0.07, -0.03, 0.14, 0.19];
  const makeEdge = (from: string, to: string, weight: number, gradient: number) => {
    const fromNeuron = idMap.get(from)!;
    const e: EdgeState = {
      id: `${from}_to_${to}`,
      from,
      to,
      weight,
      gradient,
      contribution: weight * fromNeuron.activation,
    };
    edges.push(e);
    idMap.get(from)!.outgoingEdges.push(e);
    idMap.get(to)!.incomingEdges.push(e);
  };

  for (let r = 0; r < convSize; r++) {
    for (let c = 0; c < convSize; c++) {
      const to = `C1_${r}_${c}`;
      let k = 0;
      for (let dr = 0; dr < 3; dr++) {
        for (let dc = 0; dc < 3; dc++) {
          const from = `C0_${r + dr}_${c + dc}`;
          makeEdge(from, to, kernel[k], rand(-0.012, 0.012));
          k += 1;
        }
      }
    }
  }

  const convIds = neurons.filter((n) => n.layerType === "conv").map((n) => n.id);
  convIds.forEach((cid) => {
    makeEdge(cid, "O_1", rand(-0.4, 0.4), rand(-0.015, 0.015));
    makeEdge(cid, "O_2", rand(-0.4, 0.4), rand(-0.015, 0.015));
  });

  return { neurons, edges };
}

export function generateMockRNN(): NeuroTopology {
  const neurons: NeuronState[] = [];
  const edges: EdgeState[] = [];
  const idMap = new Map<string, NeuronState>();

  const add = (id: string, layerType: NeuronState["layerType"], activation: number, bias = 0, gradient = 0) => {
    const n: NeuronState = { id, layerType, activation, bias, gradient, incomingEdges: [], outgoingEdges: [] };
    neurons.push(n);
    idMap.set(id, n);
  };
  const T = 5;
  for (let t = 0; t < T; t++) {
    add(`X_${t}`, "input", rand(0.1, 0.95), 0, 0);
    add(`H_${t}`, "recurrent", rand(0.05, 0.95), rand(-0.1, 0.1), rand(-0.02, 0.02));
    add(`O_${t}`, "dense", rand(0.1, 0.95), rand(-0.05, 0.05), rand(-0.03, 0.03));
  }

  const makeEdge = (from: string, to: string, weight: number, gradient: number) => {
    const fromNeuron = idMap.get(from)!;
    const e: EdgeState = {
      id: `${from}_to_${to}`,
      from,
      to,
      weight,
      gradient,
      contribution: weight * fromNeuron.activation,
    };
    edges.push(e);
    idMap.get(from)!.outgoingEdges.push(e);
    idMap.get(to)!.incomingEdges.push(e);
  };

  for (let t = 0; t < T; t++) {
    makeEdge(`X_${t}`, `H_${t}`, rand(-0.6, 0.6), rand(-0.02, 0.02));
    makeEdge(`H_${t}`, `O_${t}`, rand(-0.6, 0.6), rand(-0.02, 0.02));
    if (t > 0) {
      makeEdge(`H_${t - 1}`, `H_${t}`, rand(-0.6, 0.6), rand(-0.02, 0.02));
    }
  }
  return { neurons, edges };
}

const initialTopology = generateMockANN();

export const useNeurofluxStore = create<NeurofluxStore>((set) => ({
  mode: "prediction",
  currentArchitecture: "ann",
  topology: initialTopology,
  selectedNeuronId: initialTopology.neurons.find((n) => n.id.startsWith("H1_"))?.id ?? initialTopology.neurons[0]?.id ?? null,
  hoveredNeuronId: null,
  playbackState: "playing",
  currentEpoch: 0,
  history: [initialTopology],
  metricsHistory: [{ epoch: 0, loss: 1.2, accuracy: 0.35 }],
  showNeuronHealth: true,
  setMode: (mode) => set({ mode }),
  setCurrentArchitecture: (arch) =>
    set((state) => {
      if (arch === "ann") {
        const idx = clamp(state.currentEpoch, 0, state.history.length - 1);
        const annTopo = state.history[idx] ?? state.topology;
        return {
          currentArchitecture: arch,
          topology: annTopo,
          selectedNeuronId: annTopo.neurons[0]?.id ?? null,
          hoveredNeuronId: null,
        };
      }
      const topo = arch === "cnn" ? generateMockCNN() : generateMockRNN();
      return {
        currentArchitecture: arch,
        topology: topo,
        selectedNeuronId: topo.neurons[0]?.id ?? null,
        hoveredNeuronId: null,
      };
    }),
  setSelectedNeuron: (id) => set({ selectedNeuronId: id }),
  setHoveredNeuron: (id) => set({ hoveredNeuronId: id }),
  setTopology: (topology) => set({ topology }),
  addTopologySnapshot: (topology) =>
    set((state) => {
      const combined = [...state.history, topology];
      const overflow = Math.max(0, combined.length - MAX_HISTORY);
      const nextHistory = overflow > 0 ? combined.slice(overflow) : combined;
      const latestIndex = nextHistory.length - 1;

      if (state.currentArchitecture !== "ann") {
        const frozenIndex = Math.max(0, state.currentEpoch - overflow);
        return {
          history: nextHistory,
          currentEpoch: Math.min(frozenIndex, latestIndex),
        };
      }

      if (state.playbackState === "playing") {
        return {
          history: nextHistory,
          currentEpoch: latestIndex,
          topology,
        };
      }

      const frozenIndex = Math.max(0, state.currentEpoch - overflow);
      return {
        history: nextHistory,
        currentEpoch: Math.min(frozenIndex, latestIndex),
        topology: nextHistory[Math.min(frozenIndex, latestIndex)] ?? topology,
      };
    }),
  addMetricsSnapshot: (metric) =>
    set((state) => {
      const next = [...state.metricsHistory, metric];
      const overflow = Math.max(0, next.length - MAX_HISTORY);
      return {
        metricsHistory: overflow > 0 ? next.slice(overflow) : next,
      };
    }),
  play: () =>
    set((state) => {
      const latestIndex = Math.max(0, state.history.length - 1);
      return {
        playbackState: "playing",
        currentEpoch: latestIndex,
        topology:
          state.currentArchitecture === "ann"
            ? state.history[latestIndex] ?? state.topology
            : state.topology,
      };
    }),
  pause: () => set({ playbackState: "paused" }),
  stepForward: () =>
    set((state) => {
      if (state.playbackState !== "paused" || state.currentArchitecture !== "ann") return {};
      const idx = Math.min(state.history.length - 1, state.currentEpoch + 1);
      return {
        currentEpoch: idx,
        topology: state.history[idx] ?? state.topology,
      };
    }),
  stepBackward: () =>
    set((state) => {
      if (state.playbackState !== "paused" || state.currentArchitecture !== "ann") return {};
      const idx = Math.max(0, state.currentEpoch - 1);
      return {
        currentEpoch: idx,
        topology: state.history[idx] ?? state.topology,
      };
    }),
  scrubToEpoch: (epochIndex) =>
    set((state) => {
      if (state.currentArchitecture !== "ann") return {};
      const idx = clamp(epochIndex, 0, state.history.length - 1);
      return {
        currentEpoch: idx,
        topology: state.history[idx] ?? state.topology,
      };
    }),
  toggleNeuronHealth: () =>
    set((state) => ({
      showNeuronHealth: !state.showNeuronHealth,
    })),
  perturbWeight: (edgeId, newWeight) =>
    set((state) => {
      if (state.currentArchitecture !== "ann") return {};

      const working = recalculateForwardPass(
        (() => {
          const snapshot = cloneTopology(state.topology);
          const edge = snapshot.edges.find((e) => e.id === edgeId);
          if (!edge) return snapshot;
          edge.weight = clamp(newWeight, -2, 2);

          const byId = new Map(snapshot.edges.map((e) => [e.id, e]));
          snapshot.neurons.forEach((n) => {
            n.incomingEdges = n.incomingEdges.map((e) => byId.get(e.id) ?? e);
            n.outgoingEdges = n.outgoingEdges.map((e) => byId.get(e.id) ?? e);
          });
          return snapshot;
        })(),
      );

      const nextHistory = [...state.history];
      const idx = clamp(state.currentEpoch, 0, nextHistory.length - 1);
      nextHistory[idx] = working;

      return {
        topology: working,
        history: nextHistory,
      };
    }),
}));
