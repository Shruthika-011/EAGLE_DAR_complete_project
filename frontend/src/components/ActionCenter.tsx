import { useState } from 'react'
import { Link } from 'react-router-dom'
import { CheckCircle2, Download, FileDown, FileText, Loader2, XCircle, Zap } from 'lucide-react'
import { ApiError, downloadBlob, url } from '../api/client'
import { fmtDateTime, shortId } from '../lib/derive'
import { useAnalysis } from '../state/AnalysisContext'
import { Card } from './common'

export function ActionCenter() {
  const { currentId, pdf, generatePdf } = useAnalysis()
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  if (!currentId) return <Card title="Action Center" icon={<Zap size={16} />}><p className="muted">Run an analysis to enable report actions.</p></Card>

  const working = pdf.status === 'starting' || pdf.status === 'queued' || pdf.status === 'generating'
  const m = pdf.meta
  const answers = async () => {
    setBusy(true); setErr(null)
    try { await downloadBlob(`/api/analysis/${currentId}/export?format=json`, `eagle-dar-official-answers-${shortId(currentId)}.json`) }
    catch (e) { setErr(e instanceof ApiError ? e.message : String(e)) } finally { setBusy(false) }
  }

  return (
    <Card title="Action Center" icon={<Zap size={16} />}>
      <div className={`status-line status-${pdf.status === 'complete' ? 'ready' : pdf.status === 'failed' ? 'error' : ''}`} role="status">
        {pdf.status === 'complete' && <><CheckCircle2 size={16} /> PDF report ready (confirmed by the backend){m?.finished_utc ? ` · ${fmtDateTime(m.finished_utc)}` : ''}</>}
        {working && <><Loader2 size={16} className="spin" /> {m?.stage ?? 'Starting'}… {m ? `${Math.round(m.progress * 100)}%` : ''}</>}
        {pdf.status === 'idle' && <>Report not generated yet.</>}
        {pdf.status === 'failed' && <><XCircle size={16} /> Report generation failed: {pdf.error}</>}
      </div>
      <div className="actions">
        {pdf.status === 'complete' && m ? (
          <>
            <a className="btn btn-primary" href={url.reportPdf(m.report_id)} download><Download size={16} />Download PDF Report</a>
            <Link className="btn btn-outline" to="/reports"><FileText size={16} />View Detailed Report</Link>
          </>
        ) : (
          <button className="btn btn-primary" disabled={working} onClick={generatePdf}>
            {working ? <Loader2 size={16} className="spin" /> : <FileText size={16} />}Generate PDF report
          </button>
        )}
        <button className="btn btn-outline" disabled={busy} onClick={answers}>
          {busy ? <Loader2 size={16} className="spin" /> : <FileDown size={16} />}Official answers (JSON)
        </button>
      </div>
      {err && <p className="inline-error" role="alert">{err}</p>}
      <p className="muted small">GeoJSON is not offered because the images carry no geographic coordinates.</p>
    </Card>
  )
}
