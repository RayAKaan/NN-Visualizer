import React, { useEffect, useRef } from 'react';
import { cn } from '@/utils/cn';

export interface NeuralModalProps {
  open: boolean;
  onClose?: () => void;
  className?: string;
  children: React.ReactNode;
  label?: string;
  labelledBy?: string;
  closeOnBackdrop?: boolean;
}

export function NeuralModal({
  open,
  onClose,
  className,
  children,
  label,
  labelledBy,
  closeOnBackdrop = true,
}: NeuralModalProps) {
  const ref = useRef<HTMLDivElement>(null);
  const previousFocus = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!open) {
      previousFocus.current?.focus?.();
      previousFocus.current = null;
      return;
    }
    previousFocus.current = document.activeElement as HTMLElement | null;
    const node = ref.current;
    const frame = window.requestAnimationFrame(() => {
      node?.querySelector<HTMLElement>('input, button, select, textarea, [tabindex]:not([tabindex="-1"])')?.focus();
    });

    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        onClose?.();
      }
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
    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener('keydown', onKey);
    };
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div
      className="neural-modal-backdrop"
      onClick={closeOnBackdrop ? onClose : undefined}
      role="presentation"
    >
      <div
        ref={ref}
        className={cn('neural-modal', className)}
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label={labelledBy ? undefined : label ?? "Dialog"}
        aria-labelledby={labelledBy}
      >
        {children}
      </div>
    </div>
  );
}
