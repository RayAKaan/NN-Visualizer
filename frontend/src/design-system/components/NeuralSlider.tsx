import React from 'react';
import { cn } from '@/utils/cn';

export interface NeuralSliderProps extends React.InputHTMLAttributes<HTMLInputElement> {}

export function NeuralSlider({ className, ...props }: NeuralSliderProps) {
  return <input type="range" className={cn('neural-slider', className)} {...props} />;
}
