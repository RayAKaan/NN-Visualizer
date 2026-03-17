import React from 'react';
import { cn } from '@/utils/cn';

export interface NeuralSelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {}

export function NeuralSelect({ className, children, ...props }: NeuralSelectProps) {
  return (
    <select className={cn('neural-select', className)} {...props}>
      {children}
    </select>
  );
}
