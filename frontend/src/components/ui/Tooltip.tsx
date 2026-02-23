import { ReactNode } from "react";

export default function Tooltip({ children }: { content: string; children: ReactNode; position?: string; delay?: number }) {
  return <>{children}</>;
}
