export const typography = {
  families: {
    display: '"Inter", system-ui, -apple-system, sans-serif',
    ui: '"Inter", system-ui, -apple-system, sans-serif',
    mono: '"JetBrains Mono", "Fira Code", "SF Mono", ui-monospace, monospace',
    equation: '"JetBrains Mono", ui-monospace, monospace',
  },
  sizes: {
    xs: '11px',
    sm: '12px',
    base: '13px',
    md: '14px',
    lg: '16px',
    xl: '20px',
    '2xl': '28px',
    '3xl': '36px',
  },
  lineHeights: {
    xs: '1.5',
    sm: '1.5',
    base: '1.6',
    md: '1.5',
    lg: '1.4',
    xl: '1.3',
    '2xl': '1.2',
    '3xl': '1.1',
  },
  weights: {
    normal: 400,
    medium: 500,
    semibold: 600,
    bold: 700,
  },
} as const;
