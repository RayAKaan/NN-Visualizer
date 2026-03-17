import React from 'react';
import { cn } from '@/utils/cn';

export interface NeuralPanelProps {
  variant?: 'base' | 'elevated' | 'sunken';
  glow?: string;
  depth?: number;
  className?: string;
  style?: React.CSSProperties;
  children: React.ReactNode;
}

export function NeuralPanel({
  variant = 'base',
  glow,
  depth = 0,
  className,
  style,
  children,
}: NeuralPanelProps) {
  return (
    <div
      className={cn('neural-panel', `neural-panel-${variant}`, className)}
      style={{
        ...(glow ? { boxShadow: `var(--neural-panel-shadow), 0 0 30px ${glow}1f` } : undefined),
        transform: depth ? `translate3d(${depth}px, ${depth}px, 0)` : undefined,
        ...style,
      }}
    >
      {children}
    </div>
  );
}
