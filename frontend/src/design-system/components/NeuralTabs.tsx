import React, { useRef } from 'react';
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
  ariaLabel?: string;
}

export function NeuralTabs({ tabs, value, onChange, className, ariaLabel = 'Sections' }: NeuralTabsProps) {
  const refs = useRef<Record<string, HTMLButtonElement | null>>({});
  const enabledTabs = tabs.filter((tab) => !tab.disabled);

  const focusTab = (index: number) => {
    const next = enabledTabs[(index + enabledTabs.length) % enabledTabs.length];
    if (!next) return;
    onChange(next.id);
    refs.current[next.id]?.focus();
  };

  return (
    <div className={cn('neural-tabs', className)} role="tablist" aria-label={ariaLabel}>
      {tabs.map((tab) => {
        const isActive = tab.id === value;
        return (
          <button
            key={tab.id}
            ref={(node) => { refs.current[tab.id] = node; }}
            type="button"
            role="tab"
            id={`tab-${tab.id}`}
            aria-selected={isActive}
            aria-controls={`tabpanel-${tab.id}`}
            tabIndex={isActive ? 0 : -1}
            disabled={tab.disabled}
            className={cn('neural-tab', isActive && 'neural-tab-active')}
            onClick={() => onChange(tab.id)}
            onKeyDown={(event) => {
              const currentIndex = enabledTabs.findIndex((item) => item.id === tab.id);
              if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
                event.preventDefault();
                focusTab(currentIndex + 1);
              } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
                event.preventDefault();
                focusTab(currentIndex - 1);
              } else if (event.key === 'Home') {
                event.preventDefault();
                focusTab(0);
              } else if (event.key === 'End') {
                event.preventDefault();
                focusTab(enabledTabs.length - 1);
              }
            }}
          >
            {tab.label}
          </button>
        );
      })}
    </div>
  );
}
