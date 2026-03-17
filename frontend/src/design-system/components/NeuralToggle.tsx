import React from 'react';
import { cn } from '@/utils/cn';

export interface NeuralToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label?: string;
  className?: string;
}

export function NeuralToggle({ checked, onChange, label, className }: NeuralToggleProps) {
  return (
    <label className={cn('neural-toggle', className)}>
      <span
        className={cn('neural-toggle-track', checked && 'neural-toggle-track-on')}
        onClick={() => onChange(!checked)}
        role="switch"
        aria-checked={checked}
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            onChange(!checked);
          }
        }}
      >
        <span className={cn('neural-toggle-thumb', checked && 'neural-toggle-thumb-on')} />
      </span>
      {label ? <span className="neural-toggle-label">{label}</span> : null}
    </label>
  );
}
