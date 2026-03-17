import React from 'react';
import { cn } from '@/utils/cn';

export interface NeuralButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
}

export function NeuralButton({ variant = 'secondary', className, ...props }: NeuralButtonProps) {
  return (
    <button
      className={cn('neural-button', `neural-button-${variant}`, className)}
      {...props}
    />
  );
}
