import React from 'react';
import { cn } from '@/utils/cn';

export interface NeuralButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  icon?: boolean;
}

export function NeuralButton({
  variant = 'secondary',
  size = 'md',
  icon = false,
  className,
  ...props
}: NeuralButtonProps) {
  return (
    <button
      className={cn(
        'neural-button',
        `neural-button-${variant}`,
        size === 'sm' && 'neural-button-sm',
        size === 'lg' && 'neural-button-lg',
        icon && 'neural-button-icon',
        className,
      )}
      {...props}
    />
  );
}
