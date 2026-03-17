import React from 'react';
import { cn } from '@/utils/cn';

export interface NeuralBadgeProps {
  tone?: 'neutral' | 'success' | 'warning' | 'danger' | 'info';
  className?: string;
  children: React.ReactNode;
}

export function NeuralBadge({ tone = 'neutral', className, children }: NeuralBadgeProps) {
  return <span className={cn('neural-badge', `neural-badge-${tone}`, className)}>{children}</span>;
}
