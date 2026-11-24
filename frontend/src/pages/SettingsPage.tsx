import { CheckCircle2, RefreshCw, XCircle } from 'lucide-react'
import { API_BASE, url } from '../api/client'
import { Card, PageHeader } from '../components/common'
import { fmtDateTime } from '../lib/derive'
import { useAnalysis } from '../state/AnalysisContext'
import { useHealth } from '../state/HealthContext'
import { usePrefs } from '../state/PrefsContext'

const Ok = ({ ok, yes, no }: { ok: boolean; yes: string; no: string }) =>
  ok ? <span className="ok"><CheckCircle2 size={15} />{yes}</span> : <span className="bad"><XCircle size={15} />{no}</span>

export function SettingsPage() {
  const { connection, health, error, checkedAt, refresh } = useHealth()
  const { prefs, setPrefs, reset } = usePrefs()
  const { recent, currentId, setCurrent } = useAnalysis()
  const gpu = health?.device.gpu

  return (
    <>
      <PageHeader title="Settings" actions={<button className="btn btn-outline" onClick={refresh}><RefreshCw size={16} />Re-check backend</button>} />
      <div className="grid-2">
        <Card title="Backend connection">
          <dl className="kv">
            <dt>Backend URL</dt><dd className="mono wrap">{API_BASE} <span className="muted small">(VITE_API_BASE_URL)</span></dd>
            <dt>Status</dt><dd><Ok ok={connection === 'online'} yes="Online" no={connection === 'degraded' ? 'Degraded' : connection === 'checking' ? 'Checking…' : 'Offline'} /></dd>
            <dt>API version</dt><dd>{health?.api_version ?? '—'}</dd>
            <dt>Last checked</dt><dd>{checkedAt ? fmtDateTime(checkedAt.toISOString()) : '—'}</dd>
            <dt>Interactive docs</dt><dd><a className="link" href={url.docs} target="_blank" rel="noreferrer">Swagger UI</a></dd>
          </dl>
          {error && <p className="inline-error" role="alert">{error}</p>}
        </Card>
        <Card title="Model availability">
          {health ? (
            <dl className="kv">
              <dt>YOLOv8-seg ({health.models.yolo.file})</dt><dd><Ok ok={health.models.yolo.loaded} yes="Loaded" no={health.models.yolo.exists ? 'Not loaded' : 'File missing'} /></dd>
              <dt>Siamese ResNet18 ({health.models.siamese.file})</dt><dd><Ok ok={health.models.siamese.loaded} yes="Loaded" no={health.models.siamese.exists ? 'Not loaded' : 'File missing'} /></dd>
              <dt>Device in use</dt><dd>{health.device.in_use ?? '—'} <span className="muted small">(requested: {health.device.requested})</span></dd>
              <dt>CUDA</dt><dd><Ok ok={health.device.cuda_available} yes="Available" no="Not available (CPU fallback)" /></dd>
              {gpu && <><dt>GPU</dt><dd>{gpu.name}</dd><dt>GPU memory</dt><dd>{gpu.memory_free_mb} MB free of {gpu.memory_total_mb} MB</dd></>}
              {health.reports && <><dt>Report explanation</dt><dd>{health.reports.explanation}{health.reports.llm_enabled && !health.reports.llm_key_present && <span className="inline-error"> (LLM enabled but no API key; the template is used)</span>}</dd></>}
              <dt>Upload limit</dt><dd>{health.limits.max_upload_mb} MB per image · {health.limits.max_pixels.toLocaleString()} px</dd>
            </dl>
          ) : <p className="muted">Unavailable while the backend cannot be reached.</p>}
          {health?.models.load_error && <p className="inline-error">{health.models.load_error}</p>}
        </Card>
        <Card title="Display preferences" actions={<button className="btn btn-ghost" onClick={reset}>Reset</button>}>
          <div className="form-grid one">
            <label className="inline"><input type="checkbox" checked={prefs.showIds} onChange={(e) => setPrefs({ showIds: e.target.checked })} />Show building IDs on maps when zoomed in</label>
            <label className="inline"><input type="checkbox" checked={prefs.compact} onChange={(e) => setPrefs({ compact: e.target.checked })} />Compact spacing</label>
            <label>Damage overlay opacity: {Math.round(prefs.overlayOpacity * 100)}%
              <input type="range" min={0.1} max={0.8} step={0.05} value={prefs.overlayOpacity} onChange={(e) => setPrefs({ overlayOpacity: Number(e.target.value) })} /></label>
          </div>
          <p className="muted small">Saved in this browser only. A dark theme is not implemented.</p>
        </Card>
        <Card title="Recent analyses (this browser)">
          {recent.length === 0 ? <p className="muted">None yet.</p> : (
            <ul className="recent-list">{recent.map((r) => (
              <li key={r.id}><button className="link" onClick={() => setCurrent(r.id)}>{r.label}</button>{r.id === currentId && <b className="small"> · selected</b>}
                <span className="muted small"> · {r.total} buildings · {fmtDateTime(r.created_utc)}</span></li>))}</ul>
          )}
        </Card>
      </div>
    </>
  )
}
