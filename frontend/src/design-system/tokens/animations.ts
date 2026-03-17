export const animations = {
  easeOutExpo: 'cubic-bezier(0.16, 1, 0.3, 1)',
  durations: {
    fast: '150ms',
    normal: '300ms',
    slow: '600ms',
  },
  springs: {
    gentle: { stiffness: 240, damping: 26 },
    snappy: { stiffness: 320, damping: 22 },
  },
} as const;
