import React from 'react';
import { Loader2 } from 'lucide-react';
import { cn } from '@/utils/cn';

export interface NeuralButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  icon?: boolean;
  loading?: boolean;
  loadingLabel?: string;
}

export function NeuralButton({
  variant = 'secondary',
  size = 'md',
  icon = false,
  loading = false,
  loadingLabel = 'Working…',
  className,
  children,
  disabled,
  type = 'button',
  ...props
}: NeuralButtonProps) {
  return (
    <button
      type={type}
      className={cn(
        'neural-button',
        `neural-button-${variant}`,
        size === 'sm' && 'neural-button-sm',
        size === 'lg' && 'neural-button-lg',
        icon && 'neural-button-icon',
        loading && 'neural-button-loading',
        className,
      )}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      {...props}
    >
      {loading ? <Loader2 size={14} className="neural-button-spinner" aria-hidden="true" /> : null}
      <span>{loading ? loadingLabel : children}</span>
    </button>
  );
}
