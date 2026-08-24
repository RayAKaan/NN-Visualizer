export const neuralPalette = {
  void: '#FAF7F2',
  abyss: '#FFFFFF',
  obsidian: '#FFFFFF',
  slate: '#F3EEE5',
  graphite: '#D8CFC0',
  steel: '#C9BEAE',
  silver: '#57534E',
  cloud: '#44403C',
  pearl: '#292524',
  white: '#1C1917',
  ash: '#79716B',
  synapse: { dim: '#E3F0FA', base: '#0072B2', bright: '#004E7E', glow: '#8FC6E8' },
  axon: { dim: '#E0F3EE', base: '#00806A', bright: '#005947', glow: '#7FC7B4' },
  dendrite: { dim: '#F5E6F0', base: '#A64D85', bright: '#7C3560', glow: '#D9A8C6' },
  soma: { dim: '#E2F2E7', base: '#15803D', bright: '#0F5C2C', glow: '#86CB9E' },
  cortex: { dim: '#FBEDDD', base: '#B45309', bright: '#8A3E06', glow: '#E5B072' },
  lesion: { dim: '#FBE7E7', base: '#B91C1C', bright: '#8F1414', glow: '#E58C8C' },
  myelin: { dim: '#F5E6F0', base: '#A64D85', bright: '#7C3560', glow: '#D9A8C6' },
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
  if (value > 0) return lerpColor(neuralPalette.axon.base, neuralPalette.axon.bright, t);
  if (value < 0) return lerpColor(neuralPalette.dendrite.base, neuralPalette.dendrite.bright, t);
  return neuralPalette.steel;
}

export function weightColor(value: number, maxAbs: number = 1): string {
  const t = Math.min(Math.abs(value) / maxAbs, 1);
  if (value >= 0) return lerpColor('#EDE7DB', neuralPalette.axon.base, t);
  return lerpColor('#EDE7DB', neuralPalette.dendrite.base, t);
}

export function gradientHealthColor(norm: number): string {
  if (norm < 1e-7) return neuralPalette.lesion.base;
  if (norm < 1e-4) return neuralPalette.cortex.base;
  if (norm > 100) return neuralPalette.lesion.base;
  if (norm > 10) return neuralPalette.cortex.base;
  return neuralPalette.soma.base;
}

/** Soft warm shadow halo - replaces the neon bloom glows of the dark theme. */
export function glowStyle(color: string, intensity: number = 0.5): string {
  const clamped = Math.max(0, Math.min(1, intensity));
  const alpha = Math.round(clamped * 60).toString(16).padStart(2, '0');
  const alphaHalf = Math.round(clamped * 28).toString(16).padStart(2, '0');
  return `0 2px ${6 + clamped * 10}px ${color}${alpha}, 0 6px ${18 + clamped * 22}px ${color}${alphaHalf}`;
}

export function neuralGlow(color: string, intensity: number = 0.5): string {
  const clamped = Math.max(0, Math.min(1, intensity));
  const tight = Math.round(clamped * 70).toString(16).padStart(2, '0');
  const mid = Math.round(clamped * 35).toString(16).padStart(2, '0');
  const wide = Math.round(clamped * 16).toString(16).padStart(2, '0');
  return `0 0 ${4 + clamped * 6}px ${color}${tight}, 0 0 ${12 + clamped * 14}px ${color}${mid}, 0 4px ${24 + clamped * 24}px ${color}${wide}`;
}
