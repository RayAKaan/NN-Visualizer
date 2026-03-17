export const neuralPalette = {
  void: '#08090D',
  abyss: '#0C0E14',
  obsidian: '#12141C',
  slate: '#1A1D28',
  graphite: '#242836',
  steel: '#3A3F52',
  silver: '#6B7394',
  cloud: '#9BA3C2',
  pearl: '#C8CEE4',
  white: '#E8ECF8',
  ash: '#4A5068',
  synapse: { dim: '#1A3A5C', base: '#3B82F6', bright: '#60A5FA', glow: '#93C5FD' },
  axon: { dim: '#0D3D3F', base: '#06B6D4', bright: '#22D3EE', glow: '#67E8F9' },
  dendrite: { dim: '#2D1B4E', base: '#8B5CF6', bright: '#A78BFA', glow: '#C4B5FD' },
  soma: { dim: '#0D2818', base: '#22C55E', bright: '#4ADE80', glow: '#86EFAC' },
  cortex: { dim: '#3D2008', base: '#F59E0B', bright: '#FBBF24', glow: '#FDE68A' },
  lesion: { dim: '#4A0810', base: '#E11D48', bright: '#F43F5E', glow: '#FB7185' },
  myelin: { dim: '#3B0D2E', base: '#EC4899', bright: '#F472B6', glow: '#F9A8D4' },
} as const;

export type NeuralAccent = keyof typeof neuralPalette | 'synapse' | 'axon' | 'dendrite' | 'soma' | 'cortex' | 'lesion' | 'myelin';

export function lerpColor(a: string, b: string, t: number): string {
  const clampT = Math.max(0, Math.min(1, t));
  const parse = (value: string) => {
    const cleaned = value.replace('#', '');
    const num = parseInt(cleaned, 16);
    return [num >> 16, (num >> 8) & 255, num & 255];
  };
  const [r1, g1, b1] = parse(a);
  const [r2, g2, b2] = parse(b);
  const resR = Math.round(r1 + (r2 - r1) * clampT);
  const resG = Math.round(g1 + (g2 - g1) * clampT);
  const resB = Math.round(b1 + (b2 - b1) * clampT);
  return `#${[resR, resG, resB].map((v) => v.toString(16).padStart(2, '0')).join('')}`;
}

export function activationColor(value: number, maxAbs: number = 1): string {
  const t = Math.abs(value) / maxAbs;
  if (value > 0) return lerpColor(neuralPalette.axon.dim, neuralPalette.axon.glow, t);
  if (value < 0) return lerpColor(neuralPalette.dendrite.dim, neuralPalette.dendrite.glow, t);
  return neuralPalette.steel;
}

export function weightColor(value: number, maxAbs: number = 1): string {
  const t = Math.min(Math.abs(value) / maxAbs, 1);
  if (value >= 0) return lerpColor(neuralPalette.graphite, neuralPalette.axon.bright, t);
  return lerpColor(neuralPalette.graphite, neuralPalette.dendrite.bright, t);
}

export function gradientHealthColor(norm: number): string {
  if (norm < 1e-7) return neuralPalette.lesion.bright;
  if (norm < 1e-4) return neuralPalette.cortex.bright;
  if (norm > 100) return neuralPalette.lesion.bright;
  if (norm > 10) return neuralPalette.cortex.bright;
  return neuralPalette.soma.bright;
}

export function glowStyle(color: string, intensity: number = 0.5): string {
  const clamped = Math.max(0, Math.min(1, intensity));
  const alpha = Math.round(clamped * 80).toString(16).padStart(2, '0');
  const alphaHalf = Math.round(clamped * 40).toString(16).padStart(2, '0');
  const alphaQuarter = Math.round(clamped * 20).toString(16).padStart(2, '0');
  return `0 0 ${8 + clamped * 16}px ${color}${alpha}, 0 0 ${20 + clamped * 30}px ${color}${alphaHalf}, 0 0 ${40 + clamped * 60}px ${color}${alphaQuarter}`;
}

export function neuralGlow(color: string, intensity: number = 0.5): string {
  const clamped = Math.max(0, Math.min(1, intensity));
  const tight = Math.round(clamped * 90).toString(16).padStart(2, '0');
  const mid = Math.round(clamped * 55).toString(16).padStart(2, '0');
  const wide = Math.round(clamped * 30).toString(16).padStart(2, '0');
  return `0 0 ${8 + clamped * 8}px ${color}${tight}, 0 0 ${20 + clamped * 20}px ${color}${mid}, 0 0 ${40 + clamped * 40}px ${color}${wide}`;
}
