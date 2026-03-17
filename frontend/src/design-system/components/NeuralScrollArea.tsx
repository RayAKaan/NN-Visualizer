import React from 'react';
import { cn } from '@/utils/cn';

export interface NeuralScrollAreaProps {
  className?: string;
  children: React.ReactNode;
}

export function NeuralScrollArea({ className, children }: NeuralScrollAreaProps) {
  return <div className={cn('neural-scroll-area', className)}>{children}</div>;
}
