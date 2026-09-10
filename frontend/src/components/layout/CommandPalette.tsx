import React, { useEffect, useId, useRef } from "react";
import { Check, Command, Search } from "lucide-react";
import { cn } from "@/utils/cn";

export interface AppCommand {
  id: string;
  label: string;
  description?: string;
  group: "Navigation" | "Actions" | "Models" | "Utility";
  shortcut?: string;
  run: () => void;
}

interface CommandPaletteProps {
  open: boolean;
  query: string;
  commands: AppCommand[];
  activeIndex: number;
  onQueryChange: (query: string) => void;
  onActiveIndexChange: (index: number) => void;
  onRun: (command: AppCommand) => void;
  onClose: () => void;
}

export function CommandPalette({
  open,
  query,
  commands,
  activeIndex,
  onQueryChange,
  onActiveIndexChange,
  onRun,
  onClose,
}: CommandPaletteProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);
  const titleId = useId();
  const listId = useId();

  useEffect(() => {
    if (!open) return;
    previousFocusRef.current = document.activeElement as HTMLElement | null;
    const frame = window.requestAnimationFrame(() => inputRef.current?.focus());
    return () => window.cancelAnimationFrame(frame);
  }, [open]);

  useEffect(() => {
    if (open) return;
    previousFocusRef.current?.focus?.();
    previousFocusRef.current = null;
  }, [open]);

  if (!open) return null;

  const handleKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Escape") {
      event.preventDefault();
      onClose();
      return;
    }
    if (event.key === "ArrowDown") {
      event.preventDefault();
      onActiveIndexChange(Math.min(activeIndex + 1, Math.max(commands.length - 1, 0)));
      return;
    }
    if (event.key === "ArrowUp") {
      event.preventDefault();
      onActiveIndexChange(Math.max(activeIndex - 1, 0));
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      const command = commands[activeIndex];
      if (command) onRun(command);
      return;
    }
    if (event.key === "Tab") {
      const root = dialogRef.current;
      if (!root) return;
      const focusable = Array.from(root.querySelectorAll<HTMLElement>("input, button:not([disabled])"));
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
  };

  return (
    <div className="command-palette-backdrop" onMouseDown={onClose}>
      <div
        ref={dialogRef}
        className="command-palette"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onMouseDown={(event) => event.stopPropagation()}
        onKeyDown={handleKeyDown}
      >
        <div className="command-palette-heading">
          <div>
            <div className="command-palette-eyebrow">Application command layer</div>
            <h2 id={titleId}>What do you want to do?</h2>
          </div>
          <kbd>Esc</kbd>
        </div>
        <label className="command-palette-search">
          <Search size={16} aria-hidden="true" />
          <input
            ref={inputRef}
            value={query}
            onChange={(event) => onQueryChange(event.target.value)}
            placeholder="Search workspaces and actions"
            role="combobox"
            aria-autocomplete="list"
            aria-expanded="true"
            aria-controls={listId}
            aria-activedescendant={commands[activeIndex] ? `command-${commands[activeIndex].id}` : undefined}
          />
          <Command size={14} aria-hidden="true" />
        </label>

        <div className="command-palette-results" id={listId} role="listbox" aria-label="Commands">
          {commands.length > 0 ? (
            commands.map((command, index) => (
              <button
                type="button"
                key={command.id}
                id={`command-${command.id}`}
                role="option"
                aria-selected={index === activeIndex}
                className={cn("command-palette-item", index === activeIndex && "is-active")}
                onMouseEnter={() => onActiveIndexChange(index)}
                onClick={() => onRun(command)}
                ref={(node) => {
                  if (node && index === activeIndex) node.scrollIntoView({ block: "nearest" });
                }}
              >
                <span className="command-palette-item-icon" aria-hidden="true">
                  {index === activeIndex ? <Check size={15} /> : <span />}
                </span>
                <span className="command-palette-item-copy">
                  <strong>{command.label}</strong>
                  {command.description ? <small>{command.description}</small> : null}
                </span>
                <span className="command-palette-item-meta">
                  <small>{command.group}</small>
                  {command.shortcut ? <kbd>{command.shortcut}</kbd> : null}
                </span>
              </button>
            ))
          ) : (
            <div className="command-palette-empty" role="status">
              <Search size={18} aria-hidden="true" />
              <strong>No matching commands</strong>
              <span>Try a workspace name or action.</span>
            </div>
          )}
        </div>
        <div className="command-palette-footer">
          <span><kbd>↑</kbd><kbd>↓</kbd> Navigate</span>
          <span><kbd>↵</kbd> Run</span>
          <span><kbd>Esc</kbd> Close</span>
        </div>
      </div>
    </div>
  );
}
