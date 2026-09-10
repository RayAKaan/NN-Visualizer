import React from 'react';
import { cn } from '@/utils/cn';

export interface NeuralProgressProps {
  value: number;
  max?: number;
  label?: string;
  showValue?: boolean;
  className?: string;
}

export function NeuralProgress({ value, max = 100, label, showValue = false, className }: NeuralProgressProps) {
  const safeMax = max > 0 ? max : 100;
  const clamped = Math.max(0, Math.min(safeMax, value));
  const percent = (clamped / safeMax) * 100;
  return (
    <div className={cn('neural-progress-wrap', className)}>
      {label || showValue ? (
        <div className="neural-progress-label">
          {label ? <span>{label}</span> : <span />}
          {showValue ? <span className="font-mono">{Math.round(percent)}%</span> : null}
        </div>
      ) : null}
      <div
        className="neural-progress"
        role="progressbar"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={safeMax}
        aria-valuenow={clamped}
      >
        <div className="neural-progress-bar" style={{ width: `${percent}%` }} />
      </div>
    </div>
  );
}
