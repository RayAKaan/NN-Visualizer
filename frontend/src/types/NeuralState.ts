/* =====================================================
   Edge + Neural State Types (Robust & UI-safe)

/**
 * A weighted connection between two neurons.
 * `from` and `to` are indices (not positions).
 */
export type Edge = {
  from: number;
  to: number;
  strength: number; // signed, magnitude used for visuals
};

/**
 * Full neural state snapshot used by 3D + analytics views.
 * Designed to be:
 * - serializable
 * - partially available (progressive loading)
 * - safe for rendering
 */
export type NeuralState = {
  /** Raw input pixels (length = 784) */
  input: number[];

  /** Layer activations (normalized or raw depending on usage) */
  layers: {
    hidden1: number[];
    hidden2: number[];
    output: number[];
  };

  /** Final prediction */
  prediction: number;
  confidence: number;

  /** Sparse edge lists for visualization */
  edges: {
    /** Connections hidden1 → hidden2 */
    hidden1_hidden2: Edge[];

    /** Connections hidden2 → output (usually only to predicted digit) */
    hidden2_output: Edge[];
  };
};

/* =====================================================
   Optional helper types (recommended)

/**
 * Defensive empty state you can reuse to avoid undefined checks.
 */
export const EMPTY_NEURAL_STATE: NeuralState = {
  input: Array(28 * 28).fill(0),
  layers: {
    hidden1: [],
    hidden2: [],
    output: [],
  },
  prediction: -1,
  confidence: 0,
  edges: {
    hidden1_hidden2: [],
    hidden2_output: [],
  },
};
