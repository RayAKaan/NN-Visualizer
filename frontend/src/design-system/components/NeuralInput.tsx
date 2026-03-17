import React from 'react';
import { cn } from '@/utils/cn';

export interface NeuralInputProps extends React.InputHTMLAttributes<HTMLInputElement> {}

export function NeuralInput({ className, ...props }: NeuralInputProps) {
  return <input className={cn('neural-input', className)} {...props} />;
}
