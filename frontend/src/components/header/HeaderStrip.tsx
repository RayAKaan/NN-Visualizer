import React from "react";

interface HeaderAction {
  label: string;
  onClick: () => void;
}

interface Props {
  title: string;
  subtitle?: string;
  actions?: HeaderAction[];
}

export function HeaderStrip({ title, subtitle, actions = [] }: Props) {
  return (
    <div className="rounded-xl border border-cyan-400/15 bg-slate-900/70 backdrop-blur px-4 py-3 flex flex-wrap items-center justify-between gap-3">
      <div>
        <h2 className="text-base md:text-lg font-semibold text-cyan-200">{title}</h2>
        {subtitle && <p className="text-xs text-slate-400 mt-0.5">{subtitle}</p>}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        {actions.map((a) => (
          <button
            key={a.label}
            onClick={a.onClick}
            className="h-9 px-3 rounded-lg border border-white/10 bg-white/5 text-xs text-slate-200 hover:border-cyan-400/40 hover:bg-cyan-500/10"
          >
            {a.label}
          </button>
        ))}
      </div>
    </div>
  );
}
