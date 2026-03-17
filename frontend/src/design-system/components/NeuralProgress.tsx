import React from 'react';
import { cn } from '@/utils/cn';

export interface NeuralProgressProps {
  value: number;
  className?: string;
}

export function NeuralProgress({ value, className }: NeuralProgressProps) {
  const clamped = Math.max(0, Math.min(100, value));
  return (
    <div className={cn('neural-progress', className)}>
      <div className="neural-progress-bar" style={{ width: `${clamped}%` }} />
    </div>
  );
}
