import React from 'react';
import { cn } from '@/utils/cn';
import { useAnimatedNumber } from '@/design-system/hooks/useAnimatedNumber';

export interface NeuralNumberProps {
  value: number;
  precision?: number;
  duration?: number;
  className?: string;
}

export function NeuralNumber({ value, precision = 2, duration = 300, className }: NeuralNumberProps) {
  const display = useAnimatedNumber(value, duration);
  return (
    <span className={cn('neural-number', className)}>
      {display.toFixed(precision)}
    </span>
  );
}
