export interface MathVariable {
  id: string;
  symbol: string;
  stageId: string;
  type: "input" | "output" | "weight" | "bias" | "gradient" | "gate";
  shape: number[];
  equationPosition?: {
    startChar: number;
    endChar: number;
  };
  visualTarget?: {
    selector: string;
    highlightStyle: "glow" | "border" | "pulse" | "overlay";
  };
}

export interface BindingState {
  activeVariable: string | null;
  hoveredSource: "math" | "visual" | null;
  connections: Array<{
    variableId: string;
    mathRect: DOMRect;
    visualRect: DOMRect;
  }>;
}
