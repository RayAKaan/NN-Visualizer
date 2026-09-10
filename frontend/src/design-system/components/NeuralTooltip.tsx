import React, { useId } from 'react';
import { cn } from '@/utils/cn';

export interface NeuralTooltipProps {
  content: React.ReactNode;
  className?: string;
  children: React.ReactNode;
}

export function NeuralTooltip({ content, className, children }: NeuralTooltipProps) {
  const tooltipId = useId();
  return (
    <span className={cn('neural-tooltip', className)} tabIndex={0} aria-describedby={tooltipId}>
      {children}
      <span id={tooltipId} className="neural-tooltip-content" role="tooltip">{content}</span>
    </span>
  );
}
