export type ColorRamp = Array<[number, number, number]>;

export const ACTIVATION_RAMP_DARK: ColorRamp = [
  [10, 10, 20],
  [15, 30, 60],
  [20, 60, 120],
  [30, 100, 180],
  [45, 140, 220],
  [80, 200, 240],
  [140, 230, 250],
  [200, 245, 255],
  [240, 252, 255],
];

export const ACTIVATION_RAMP_LIGHT: ColorRamp = [
  [245, 247, 252],
  [210, 225, 250],
  [170, 200, 245],
  [120, 170, 235],
  [70, 130, 220],
  [40, 100, 200],
  [25, 75, 180],
  [18, 55, 140],
  [15, 40, 100],
];

export const WEIGHT_RAMP_DARK: ColorRamp = [
  [180, 50, 80],
  [150, 60, 90],
  [120, 65, 95],
  [90, 70, 100],
  [42, 42, 58],
  [60, 100, 95],
  [70, 140, 130],
  [80, 180, 170],
  [94, 234, 212],
];

export const WEIGHT_RAMP_LIGHT: ColorRamp = [
  [190, 24, 93],
  [210, 70, 120],
  [230, 120, 155],
  [245, 175, 195],
  [229, 231, 235],
  [170, 240, 230],
  [100, 210, 195],
  [40, 180, 165],
  [13, 148, 136],
];

export const GRADIENT_RAMP_DARK: ColorRamp = [
  [15, 15, 20],
  [50, 30, 15],
  [90, 50, 15],
  [140, 80, 20],
  [190, 110, 30],
  [220, 140, 40],
  [245, 170, 55],
  [255, 200, 80],
  [255, 240, 150],
];

export const GRADIENT_RAMP_LIGHT: ColorRamp = [
  [255, 251, 235],
  [254, 243, 199],
  [253, 224, 140],
  [252, 200, 80],
  [245, 170, 50],
  [220, 140, 35],
  [190, 110, 25],
  [160, 85, 15],
  [120, 60, 10],
];

export const SALIENCY_RAMP: Array<[number, number, number, number]> = [
  [0, 0, 0, 0],
  [30, 60, 180, 40],
  [60, 80, 200, 80],
  [120, 60, 220, 120],
  [200, 60, 180, 150],
  [240, 80, 80, 170],
  [250, 140, 40, 190],
  [255, 200, 50, 210],
  [255, 255, 120, 230],
];

export const GATE_COLORS = {
  dark: {
    forget: { low: "#2d1520", high: "#f472b6", label: "#f9a8d4" },
    input: { low: "#152d20", high: "#34d399", label: "#86efac" },
    output: { low: "#15202d", high: "#60a5fa", label: "#93c5fd" },
    cell: { low: "#201520", high: "#c084fc", label: "#d8b4fe" },
  },
  light: {
    forget: { low: "#fce7f3", high: "#db2777", label: "#be185d" },
    input: { low: "#d1fae5", high: "#059669", label: "#047857" },
    output: { low: "#dbeafe", high: "#2563eb", label: "#1d4ed8" },
    cell: { low: "#f3e8ff", high: "#9333ea", label: "#7c3aed" },
  },
};

export function sampleRamp(ramp: ColorRamp, value: number): [number, number, number] {
  const clamped = Math.max(0, Math.min(1, value));
  const scaled = clamped * (ramp.length - 1);
  const low = Math.floor(scaled);
  const high = Math.ceil(scaled);
  const t = scaled - low;

  if (low === high) return ramp[low];
  const [r1, g1, b1] = ramp[low];
  const [r2, g2, b2] = ramp[high];

  return [
    Math.round(r1 + (r2 - r1) * t),
    Math.round(g1 + (g2 - g1) * t),
    Math.round(b1 + (b2 - b1) * t),
  ];
}

export function renderHeatmap(
  data: Float32Array,
  width: number,
  height: number,
  ramp: ColorRamp,
  normalize = true,
): string {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";

  const img = ctx.createImageData(width, height);

  let min = 0;
  let max = 1;
  if (normalize) {
    min = Number.POSITIVE_INFINITY;
    max = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < data.length; i += 1) {
      const value = data[i];
      if (value < min) min = value;
      if (value > max) max = value;
    }
    if (max === min) max = min + 1;
  }

  for (let i = 0; i < data.length; i += 1) {
    const normalized = (data[i] - min) / (max - min);
    const [r, g, b] = sampleRamp(ramp, normalized);
    const offset = i * 4;
    img.data[offset] = r;
    img.data[offset + 1] = g;
    img.data[offset + 2] = b;
    img.data[offset + 3] = 255;
  }

  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL();
}

export function renderSaliencyOverlay(data: Float32Array, width: number, height: number): string {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";

  const img = ctx.createImageData(width, height);

  let maxAbs = 0;
  for (let i = 0; i < data.length; i += 1) {
    const value = Math.abs(data[i]);
    if (value > maxAbs) maxAbs = value;
  }
  if (maxAbs === 0) maxAbs = 1;

  for (let i = 0; i < data.length; i += 1) {
    const normalized = Math.abs(data[i]) / maxAbs;
    const scaled = normalized * (SALIENCY_RAMP.length - 1);
    const low = Math.floor(scaled);
    const high = Math.min(Math.ceil(scaled), SALIENCY_RAMP.length - 1);
    const t = scaled - low;
    const [r1, g1, b1, a1] = SALIENCY_RAMP[low];
    const [r2, g2, b2, a2] = SALIENCY_RAMP[high];
    const offset = i * 4;
    img.data[offset] = Math.round(r1 + (r2 - r1) * t);
    img.data[offset + 1] = Math.round(g1 + (g2 - g1) * t);
    img.data[offset + 2] = Math.round(b1 + (b2 - b1) * t);
    img.data[offset + 3] = Math.round(a1 + (a2 - a1) * t);
  }

  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL();
}
