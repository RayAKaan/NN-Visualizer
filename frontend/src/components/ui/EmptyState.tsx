export default function EmptyState({ icon, title, subtitle }: { icon: string; title: string; subtitle?: string }) {
  return <div className="empty-state"><div className="empty-state-icon">{icon}</div><div className="empty-state-title">{title}</div>{subtitle && <div className="empty-state-subtitle">{subtitle}</div>}</div>;
}
