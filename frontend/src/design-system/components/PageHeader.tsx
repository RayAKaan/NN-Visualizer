import React from "react";
import { cn } from "@/utils/cn";

export interface PageHeaderProps {
  title: string;
  subtitle?: string;
  eyebrow?: string;
  actions?: React.ReactNode;
  status?: React.ReactNode;
  children?: React.ReactNode;
  className?: string;
}

export function PageHeader({ title, subtitle, eyebrow, actions, status, children, className }: PageHeaderProps) {
  return (
    <header className={cn("ph-root", className)}>
      <div className="ph-row">
        <div className="ph-heading">
          {eyebrow ? <div className="ph-eyebrow">{eyebrow}</div> : null}
          <h1 className="ph-title">{title}</h1>
          {subtitle ? <p className="ph-subtitle">{subtitle}</p> : null}
        </div>
        {status || actions ? (
          <div className="ph-actions">
            {status ? <div className="ph-status">{status}</div> : null}
            {actions ? <div className="ph-action-group">{actions}</div> : null}
          </div>
        ) : null}
      </div>
      {children ? <div className="ph-extra">{children}</div> : null}
    </header>
  );
}
