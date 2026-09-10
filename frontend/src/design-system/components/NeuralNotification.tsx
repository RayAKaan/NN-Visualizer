import React from 'react';
import { cn } from '@/utils/cn';

export interface NeuralNotificationProps {
  tone?: 'info' | 'success' | 'warning' | 'danger';
  className?: string;
  children: React.ReactNode;
}

export function NeuralNotification({ tone = 'info', className, children }: NeuralNotificationProps) {
  return <div className={cn('neural-notification', `neural-notification-${tone}`, className)}>{children}</div>;
}
