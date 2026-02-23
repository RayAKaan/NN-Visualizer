export default function ErrorBanner({ message, type = "error", onRetry, retryCount }: { message: string; type?: "error" | "warning"; onRetry?: () => void; retryCount?: number }) {
  return <div className={type === "error" ? "error-banner" : "warning-banner"}>{message} {onRetry && <button onClick={onRetry}>Retry</button>} {retryCount ?? 0}</div>;
}
