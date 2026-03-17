export interface KernelPreset {
  id: string;
  name: string;
  description: string;
  category: "edge" | "blur" | "sharpen" | "emboss" | "custom" | "identity";
  size: 3 | 5 | 7;
  values: number[];
  expectedEffect: string;
  icon: string;
}

export interface KernelLabState {
  activeKernel: number[];
  kernelSize: number;
  targetLayerId: string;
  previewFeatureMap: Float32Array | null;
  isEditing: boolean;
  history: number[][];
}
