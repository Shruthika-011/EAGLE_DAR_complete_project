import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { AlertCircle, Loader2, UploadCloud } from 'lucide-react'
import { ApiError } from '../api/client'
import { useAnalysis } from '../state/AnalysisContext'
import type { Analysis } from '../types'

export function Card({ title, icon, actions, children, className = '' }: { title?: string; icon?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`card ${className}`}>
      {(title || actions) && (
        <header className="card-head">
          <h2>{icon}{title}</h2>
          {actions && <div className="card-actions">{actions}</div>}
        </header>
      )}
      {children}
    </section>
  )
}

export function Spinner({ label }: { label?: string }) {
  return <div className="spinner" role="status"><Loader2 className="spin" size={20} />{label ?? 'Loading…'}</div>
}

export function ErrorBanner({ error, onRetry }: { error: ApiError | string; onRetry?: () => void }) {
  const e = typeof error === 'string' ? null : error
  return (
    <div className="banner banner-error" role="alert">
      <AlertCircle size={18} />
      <div>
        <b>{e ? e.code.replace(/_/g, ' ') : 'Error'}</b>
        <p>{e ? e.message : error as string}</p>
        {e?.details && <pre>{JSON.stringify(e.details, null, 2)}</pre>}
      </div>
      {onRetry && <button className="btn btn-ghost" onClick={onRetry}>Retry</button>}
    </div>
  )
}

export function EmptyState({ title, text, cta = true }: { title: string; text: string; cta?: boolean }) {
  return (
    <div className="empty">
      <UploadCloud size={34} />
      <h3>{title}</h3>
      <p>{text}</p>
      {cta && <Link className="btn btn-primary" to="/upload">Upload satellite images</Link>}
    </div>
  )
}

export function PageHeader({ title, subtitle, actions }: { title: string; subtitle?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="page-head">
      <div><h1>{title}</h1>{subtitle && <p className="muted">{subtitle}</p>}</div>
      {actions && <div className="page-actions">{actions}</div>}
    </div>
  )
}

/** Renders children only once an analysis is loaded; otherwise the right empty/loading/error state. */
export function RequireAnalysis({ children }: { children: (a: Analysis) => ReactNode }) {
  const { analysis, status, error, reload, setCurrent, currentId } = useAnalysis()
  if (status === 'loading') return <Spinner label="Loading analysis from the backend…" />
  if (status === 'error' && error)
    return (
      <>
        <ErrorBanner error={error} onRetry={error.code === 'analysis_not_found' ? undefined : reload} />
        <div className="stack-row">
          <button className="btn btn-ghost" onClick={() => setCurrent(null)}>Clear selection</button>
          <Link className="btn btn-primary" to="/upload">Start a new analysis</Link>
        </div>
      </>
    )
  if (!analysis || !currentId) return <EmptyState title="No analysis selected" text="Upload a before/after image pair to see results here." />
  return <>{children(analysis)}</>
}
