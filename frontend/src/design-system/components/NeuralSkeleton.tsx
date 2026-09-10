import React from 'react';
import { cn } from '@/utils/cn';

export interface NeuralSkeletonProps {
  className?: string;
  style?: React.CSSProperties;
}

export function NeuralSkeleton({ className, style }: NeuralSkeletonProps) {
  return <div className={cn('neural-skeleton', className)} style={style} />;
}
