/**
 * Centralized ANN/CNN/RNN identity colors.
 * Okabe-Ito derived (Wong 2011, Nature Methods) - colorblind-safe on barley white.
 */
export interface ArchTheme {
  label: string;
  color: string;
  deep: string;
  tint: string;
  border: string;
}

export const ARCH_COLORS: Record<'ann' | 'cnn' | 'rnn', ArchTheme> = {
  ann: { label: 'ANN', color: '#0072B2', deep: '#00568A', tint: 'rgba(0,114,178,0.08)', border: 'rgba(0,114,178,0.35)' },
  cnn: { label: 'CNN', color: '#00806A', deep: '#005C4C', tint: 'rgba(0,128,106,0.08)', border: 'rgba(0,128,106,0.35)' },
  rnn: { label: 'RNN', color: '#A64D85', deep: '#7C3560', tint: 'rgba(166,77,133,0.08)', border: 'rgba(166,77,133,0.35)' },
};

export type ArchKey = keyof typeof ARCH_COLORS;

/** Okabe-Ito categorical series for charts */
export const CHART_SERIES = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9'] as const;

export function archTheme(key: ArchKey): ArchTheme {
  return ARCH_COLORS[key];
}
