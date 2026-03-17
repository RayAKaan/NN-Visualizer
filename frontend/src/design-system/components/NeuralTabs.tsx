import React from 'react';
import { cn } from '@/utils/cn';

export interface NeuralTabItem {
  id: string;
  label: string;
  disabled?: boolean;
}

export interface NeuralTabsProps {
  tabs: NeuralTabItem[];
  value: string;
  onChange: (value: string) => void;
  className?: string;
}

export function NeuralTabs({ tabs, value, onChange, className }: NeuralTabsProps) {
  return (
    <div className={cn('neural-tabs', className)} role="tablist">
      {tabs.map((tab) => {
        const isActive = tab.id === value;
        return (
          <button
            key={tab.id}
            type="button"
            role="tab"
            aria-selected={isActive}
            disabled={tab.disabled}
            className={cn('neural-tab', isActive && 'neural-tab-active')}
            onClick={() => onChange(tab.id)}
          >
            {tab.label}
          </button>
        );
      })}
    </div>
  );
}
