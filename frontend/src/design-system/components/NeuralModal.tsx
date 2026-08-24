import React, { useEffect, useRef } from 'react';
import { cn } from '@/utils/cn';

export interface NeuralModalProps {
  open: boolean;
  onClose?: () => void;
  className?: string;
  children: React.ReactNode;
  label?: string;
}

export function NeuralModal({ open, onClose, className, children, label }: NeuralModalProps) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const node = ref.current;
    node?.querySelector<HTMLElement>('input, button, select, textarea')?.focus();

    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose?.();
      if (event.key === 'Tab' && node) {
        const focusables = Array.from(
          node.querySelectorAll<HTMLElement>('a[href], button:not([disabled]), input, select, textarea, [tabindex]:not([tabindex="-1"])'),
        );
        if (focusables.length === 0) return;
        const first = focusables[0];
        const last = focusables[focusables.length - 1];
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first.focus();
        }
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div className="neural-modal-backdrop" onClick={onClose} role="presentation">
      <div
        ref={ref}
        className={cn('neural-modal', className)}
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label={label}
      >
        {children}
      </div>
    </div>
  );
}
