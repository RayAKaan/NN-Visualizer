import { Edge } from "../types";

export function matrixToEdges(matrix: number[][]): Edge[] {
  const out: Edge[] = [];
  matrix.forEach((row, i) => row.forEach((v, j) => out.push({ from: i, to: j, strength: v })));
  return out;
}
