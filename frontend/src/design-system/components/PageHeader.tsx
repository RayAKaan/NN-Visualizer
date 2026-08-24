import React from "react";
import { cn } from "@/utils/cn";

export interface PageHeaderProps {
  title: string;
  subtitle?: string;
  actions?: React.ReactNode;
  children?: React.ReactNode;
  className?: string;
}

export function PageHeader({ title, subtitle, actions, children, className }: PageHeaderProps) {
  return (
    <header className={cn("ph-root", className)}>
      <div className="ph-row">
        <div>
          <h1 className="ph-title">{title}</h1>
          {subtitle ? <p className="ph-subtitle">{subtitle}</p> : null}
        </div>
        {actions ? <div className="ph-actions">{actions}</div> : null}
      </div>
      {children ? <div className="ph-extra">{children}</div> : null}
    </header>
  );
}
