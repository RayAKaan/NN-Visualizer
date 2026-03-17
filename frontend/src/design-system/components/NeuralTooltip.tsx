import React from 'react';
import { cn } from '@/utils/cn';

export interface NeuralTooltipProps {
  content: React.ReactNode;
  className?: string;
  children: React.ReactNode;
}

export function NeuralTooltip({ content, className, children }: NeuralTooltipProps) {
  return (
    <span className={cn('neural-tooltip', className)}>
      {children}
      <span className="neural-tooltip-content">{content}</span>
    </span>
  );
}
