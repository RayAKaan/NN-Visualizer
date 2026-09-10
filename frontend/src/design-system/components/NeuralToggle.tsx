import React from 'react';
import { cn } from '@/utils/cn';

export interface NeuralToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label?: string;
  className?: string;
  disabled?: boolean;
}

export function NeuralToggle({ checked, onChange, label, className, disabled = false }: NeuralToggleProps) {
  return (
    <div className={cn('neural-toggle', className)}>
      <button
        type="button"
        className={cn('neural-toggle-track', checked && 'neural-toggle-track-on')}
        onClick={() => onChange(!checked)}
        role="switch"
        aria-checked={checked}
        aria-label={label ?? 'Toggle setting'}
        disabled={disabled}
      >
        <span className={cn('neural-toggle-thumb', checked && 'neural-toggle-thumb-on')} aria-hidden="true" />
      </button>
      {label ? <span className="neural-toggle-label">{label}</span> : null}
    </div>
  );
}
