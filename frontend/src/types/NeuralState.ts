/* =====================================================
   Edge + Neural State Types
   Robust, Serializable, UI-safe

/**
 * A weighted connection between two neurons.
 * `from` and `to` are neuron indices (NOT screen positions).
 */
export type Edge = {
  from: number;
  to: number;
  strength: number; // signed weight or gradient
};

/**
 * Complete neural state snapshot.
 * Used by:
 * - 2D connection view
 * - 3D network visualization
 * - Training playback
 * - Analytics overlays
 */
export type NeuralState = {
  /** Raw input pixels (784 for MNIST, empty during training) */
  input: number[];

  /** Layer activations */
  layers: {
    hidden1: number[];
    hidden2: number[];
    output: number[];
  };

  /** Prediction metadata (safe defaults during training) */
  prediction: number;
  confidence: number;

  /** Sparse edge lists for rendering */
  edges: {
    /** hidden1 → hidden2 connections */
    hidden1_hidden2: Edge[];

    /** hidden2 → output connections */
    hidden2_output: Edge[];
  };
};

/* =====================================================
   Safe empty fallback (IMPORTANT)

/**
 * Use this when:
 * - socket not ready
 * - model still loading
 * - switching modes
 *
 * Prevents 100% of "cannot read property" crashes.
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
