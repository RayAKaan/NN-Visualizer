import { Edge } from "../types/NeuralState";

export function matrixToEdges(matrix: number[][]): Edge[] {
  const edges: Edge[] = [];
  for (let i = 0; i < matrix.length; i += 1) {
    for (let j = 0; j < matrix[i].length; j += 1) {
      edges.push({
        from: i,
        to: j,
        strength: matrix[i][j],
      });
    }
  }
  return edges;
}
