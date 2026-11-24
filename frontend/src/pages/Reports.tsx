import { useState } from 'react'
import { CheckCircle2, Download, FileText, Loader2, RefreshCw, XCircle } from 'lucide-react'
import { downloadBlob, url } from '../api/client'
import { Card, ErrorBanner, PageHeader, RequireAnalysis, Spinner } from '../components/common'
import { fmtDateTime, shortId } from '../lib/derive'
import { useAnalysis } from '../state/AnalysisContext'
import type { Analysis, ReportMeta } from '../types'

const fmtBytes = (n: number | null) => (n === null ? '—' : n > 1048576 ? `${(n / 1048576).toFixed(1)} MB` : `${Math.round(n / 1024)} KB`)

function StatusCard({ a }: { a: Analysis }) {
  const { pdf, generatePdf } = useAnalysis()
  const m = pdf.meta
  const busy = pdf.status === 'starting' || pdf.status === 'queued' || pdf.status === 'generating'
  const pct = Math.round((m?.progress ?? 0) * 100)
  return (
    <Card title="Disaster Damage Assessment Report (PDF)" icon={<FileText size={16} />}>
      <dl className="kv">
        <dt>Analysis ID</dt><dd className="mono wrap">{a.analysis_id}</dd>
        <dt>Report ID</dt><dd className="mono wrap">{m?.report_id ?? '—'}</dd>
        <dt>Status</dt>
        <dd role="status" data-testid="report-status">
          {pdf.status === 'idle' && <span className="muted">No report generated yet</span>}
          {pdf.status === 'starting' && <><Loader2 size={14} className="spin" /> Starting…</>}
          {(pdf.status === 'queued' || pdf.status === 'generating') && <><Loader2 size={14} className="spin" /> {pdf.status === 'queued' ? 'Queued' : 'Generating'}</>}
          {pdf.status === 'complete' && <span className="ok"><CheckCircle2 size={15} />Complete</span>}
          {pdf.status === 'failed' && <span className="bad"><XCircle size={15} />Failed</span>}
        </dd>
      </dl>
      {busy && (
        <div className="progress-block" aria-live="polite">
          <div className="progress-label"><b>{m?.stage ?? 'Starting'}</b><span className="muted small">{pct}% · rendering on the backend from this analysis's stored predictions</span></div>
          <div className="bar"><i style={{ width: `${Math.max(pct, 4)}%` }} /></div>
        </div>
      )}
      {pdf.status === 'failed' && <ErrorBanner error={pdf.error ?? 'Report generation failed.'} onRetry={generatePdf} />}
      {pdf.status === 'complete' && m && (
        <dl className="kv">
          <dt>Generated</dt><dd>{fmtDateTime(m.finished_utc)} · took {m.generation_seconds?.toFixed(1)} s</dd>
          <dt>Size</dt><dd>{m.pages} pages · {fmtBytes(m.bytes)}</dd>
          <dt>Explanation</dt><dd>{m.explanation_source}{m.fallback_reason && <span className="muted small"> (LLM not used: {m.fallback_reason})</span>}</dd>
        </dl>
      )}
      <div className="actions">
        <button className="btn btn-primary" onClick={generatePdf} disabled={busy}>
          {busy ? <Loader2 size={16} className="spin" /> : pdf.status === 'complete' ? <RefreshCw size={16} /> : <FileText size={16} />}
          {pdf.status === 'complete' || pdf.status === 'failed' ? 'Regenerate Report' : 'Generate Disaster Assessment Report'}
        </button>
        {pdf.status === 'complete' && m?.download_url && (
          <a className="btn btn-outline" href={url.reportPdf(m.report_id)} download><Download size={16} />Download PDF</a>
        )}
      </div>
      <p className="muted small">Every number comes from this analysis's predictions and the deterministic spatial reasoning engine. The explanation text is a template unless an LLM is explicitly enabled; either way it cannot add numbers.</p>
    </Card>
  )
}

function History({ reports, current }: { reports: ReportMeta[]; current: string | null }) {
  const [err, setErr] = useState<string | null>(null)
  if (reports.length === 0) return null
  return (
    <Card title="Reports for this analysis">
      <table className="tbl"><thead><tr><th>Report</th><th>Generated</th><th>Status</th><th className="n">Pages</th><th /></tr></thead>
        <tbody>{reports.map((r) => (
          <tr key={r.report_id}><td className="mono">{shortId(r.report_id)}{r.report_id === current && <b className="small"> · shown</b>}</td>
            <td>{fmtDateTime(r.finished_utc ?? r.created_utc)}</td><td>{r.status}</td><td className="n">{r.pages ?? '—'}</td>
            <td>{r.download_url && <button className="link" onClick={() => downloadBlob(`/api/reports/${r.report_id}/download`, `eagle-dar-report-${shortId(r.analysis_id)}-${shortId(r.report_id)}.pdf`).catch((e) => setErr(String(e.message ?? e)))}>Download</button>}</td></tr>))}
        </tbody></table>
      {err && <p className="inline-error" role="alert">{err}</p>}
    </Card>
  )
}

function Quick() {
  const { quick, loadQuick } = useAnalysis()
  const sections = quick.json?.sections ?? []
  return (
    <details className="advanced" onToggle={(e) => { if ((e.target as HTMLDetailsElement).open && quick.status === 'idle') loadQuick() }}>
      <summary>Quick view: key findings and recommendations</summary>
      {quick.status === 'loading' && <Spinner label="Loading…" />}
      {quick.status === 'error' && <ErrorBanner error={quick.error ?? 'Could not load the quick view.'} onRetry={() => loadQuick(true)} />}
      {quick.status === 'ready' && (
        <div className="grid-2">
          {sections.filter((s) => s.title !== 'Questions answered').map((s) => (
            <div key={s.title} className="finding"><h4>{s.title}</h4>{s.paragraphs.map((p, i) => <p key={i}>{p}</p>)}</div>
          ))}
        </div>
      )}
    </details>
  )
}

function Body({ a }: { a: Analysis }) {
  const { pdf, reports } = useAnalysis()
  const m = pdf.meta
  return (
    <>
      <PageHeader title="Reports" subtitle={<>Detailed, image-rich disaster assessment for analysis <code>{shortId(a.analysis_id)}…</code></>} />
      <StatusCard a={a} />
      {pdf.status === 'complete' && m && (
        <Card title="Report preview">
          <iframe title="Disaster assessment report (PDF preview)" className="pdf-frame" src={url.reportPdf(m.report_id, true)} />
          <p className="muted small">Preview uses your browser's PDF viewer. If nothing appears, use Download PDF.</p>
        </Card>
      )}
      <History reports={reports} current={m?.report_id ?? null} />
      <Card><Quick /></Card>
    </>
  )
}

export function Reports() {
  return <RequireAnalysis>{(a) => <Body a={a} />}</RequireAnalysis>
}
