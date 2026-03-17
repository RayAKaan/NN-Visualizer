import React from 'react';
import { cn } from '@/utils/cn';

export interface NeuralModalProps {
  open: boolean;
  onClose?: () => void;
  className?: string;
  children: React.ReactNode;
}

export function NeuralModal({ open, onClose, className, children }: NeuralModalProps) {
  if (!open) return null;
  return (
    <div className="neural-modal-backdrop" onClick={onClose} role="presentation">
      <div
        className={cn('neural-modal', className)}
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
      >
        {children}
      </div>
    </div>
  );
}
