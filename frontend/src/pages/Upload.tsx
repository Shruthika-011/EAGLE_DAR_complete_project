import { useCallback, useEffect, useRef, useState, type DragEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { CheckCircle2, ImagePlus, Loader2, RefreshCw, Trash2, UploadCloud } from 'lucide-react'
import { analyze, ApiError, type AnalyzePhase } from '../api/client'
import { Card, ErrorBanner, PageHeader } from '../components/common'
import { useAnalysis } from '../state/AnalysisContext'
import { useHealth } from '../state/HealthContext'

const ALLOWED_EXT = ['jpg', 'jpeg', 'png']
const ALLOWED_MIME = ['image/jpeg', 'image/png']
const MIN_SIDE = 64

interface Picked { file: File; preview: string; width: number; height: number }

function readDims(file: File): Promise<Picked> {
  return new Promise((resolve, reject) => {
    const preview = URL.createObjectURL(file)
    const img = new Image()
    img.onload = () => resolve({ file, preview, width: img.naturalWidth, height: img.naturalHeight })
    img.onerror = () => { URL.revokeObjectURL(preview); reject(new Error('This file could not be read as an image (it may be corrupted).')) }
    img.src = preview
  })
}

function DropZone({ role, picked, error, onFile, onClear, maxMb }: {
  role: 'Before' | 'After'; picked: Picked | null; error: string | null; maxMb: number
  onFile: (f: File) => void; onClear: () => void
}) {
  const input = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)
  const drop = (e: DragEvent) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files?.[0]; if (f) onFile(f) }
  return (
    <div className="dz-wrap">
      <h3>{role}-disaster image</h3>
      <div
        className={`dropzone ${over ? 'over' : ''} ${picked ? 'has' : ''} ${error ? 'bad' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setOver(true) }} onDragLeave={() => setOver(false)} onDrop={drop}
        onClick={() => !picked && input.current?.click()} role="button" tabIndex={0} aria-label={`Choose ${role.toLowerCase()} image`}
        onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && !picked && input.current?.click()}
      >
        {picked ? <img src={picked.preview} alt={`${role} preview`} /> : (
          <div className="dz-empty"><ImagePlus size={34} /><b>Drop image here or click to browse</b><span>JPG, JPEG or PNG · up to {maxMb} MB</span></div>
        )}
      </div>
      <input ref={input} type="file" hidden accept=".jpg,.jpeg,.png,image/jpeg,image/png" onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); e.target.value = '' }} />
      {picked && (
        <div className="file-meta">
          <div><b title={picked.file.name}>{picked.file.name}</b><span>{picked.width} × {picked.height} px · {(picked.file.size / 1048576).toFixed(2)} MB</span></div>
          <div className="stack-row">
            <button className="btn btn-ghost" type="button" onClick={() => input.current?.click()}><RefreshCw size={14} />Replace</button>
            <button className="btn btn-ghost" type="button" onClick={onClear}><Trash2 size={14} />Remove</button>
          </div>
        </div>
      )}
      {error && <p className="inline-error" role="alert">{error}</p>}
    </div>
  )
}

export function Upload() {
  const nav = useNavigate()
  const { setCurrent } = useAnalysis()
  const { health, connection } = useHealth()
  const maxMb = health?.limits.max_upload_mb ?? 25
  const maxPixels = health?.limits.max_pixels ?? 4096 * 4096

  const [before, setBefore] = useState<Picked | null>(null)
  const [after, setAfter] = useState<Picked | null>(null)
  const [errB, setErrB] = useState<string | null>(null)
  const [errA, setErrA] = useState<string | null>(null)
  const [name, setName] = useState('')
  const [district, setDistrict] = useState('')
  const [state, setState] = useState('')
  const [conf, setConf] = useState(0.25)
  const [alignment, setAlignment] = useState<'check' | 'register'>('check')

  const [phase, setPhase] = useState<'idle' | AnalyzePhase | 'done'>('idle')
  const [progress, setProgress] = useState(0)
  const [apiError, setApiError] = useState<ApiError | string | null>(null)
  const [doneId, setDoneId] = useState<string | null>(null)
  const submitting = useRef(false)             // guards against double clicks / duplicate requests
  const ctl = useRef<AbortController | null>(null)
  const timer = useRef<number | undefined>(undefined)

  useEffect(() => () => { clearTimeout(timer.current); ctl.current?.abort() }, [])
  useEffect(() => () => { if (before) URL.revokeObjectURL(before.preview) }, [before])
  useEffect(() => () => { if (after) URL.revokeObjectURL(after.preview) }, [after])

  const validate = useCallback(async (file: File): Promise<{ picked?: Picked; error?: string }> => {
    const ext = file.name.split('.').pop()?.toLowerCase() ?? ''
    if (!ALLOWED_EXT.includes(ext) || (file.type && !ALLOWED_MIME.includes(file.type))) return { error: 'Unsupported format. Use JPG, JPEG or PNG.' }
    if (file.size === 0) return { error: 'The file is empty.' }
    if (file.size > maxMb * 1048576) return { error: `File is larger than the ${maxMb} MB limit.` }
    try {
      const p = await readDims(file)
      if (Math.min(p.width, p.height) < MIN_SIDE) { URL.revokeObjectURL(p.preview); return { error: `Image is too small (${p.width}×${p.height}); minimum side is ${MIN_SIDE}px.` } }
      if (p.width * p.height > maxPixels) { URL.revokeObjectURL(p.preview); return { error: `Image has ${(p.width * p.height).toLocaleString()} pixels; the limit is ${maxPixels.toLocaleString()}.` } }
      return { picked: p }
    } catch (e) { return { error: (e as Error).message } }
  }, [maxMb, maxPixels])

  const pick = (which: 'before' | 'after') => async (file: File) => {
    const r = await validate(file)
    const [set, setErr] = which === 'before' ? [setBefore, setErrB] : [setAfter, setErrA]
    setErr(r.error ?? null)
    if (r.picked) set(r.picked)
    setApiError(null)
  }

  const mismatch = before && after && (before.width !== after.width || before.height !== after.height)
    ? `Before (${before.width}×${before.height}) and after (${after.width}×${after.height}) must have identical dimensions.` : null
  const blocked = connection === 'offline' || connection === 'degraded'
  const canStart = !!before && !!after && !mismatch && !errA && !errB && phase === 'idle' && !blocked

  const start = async () => {
    if (!before || !after || submitting.current) return
    submitting.current = true
    setApiError(null); setPhase('uploading'); setProgress(0)
    const c = new AbortController(); ctl.current = c
    try {
      const a = await analyze(
        { before: before.file, after: after.file, disasterName: name, district, state, conf, alignment },
        (p, f) => { setPhase(p); setProgress(f) }, c.signal)
      setDoneId(a.analysis_id); setPhase('done')
      setCurrent(a.analysis_id)
      timer.current = window.setTimeout(() => nav('/'), 1800)
    } catch (e) {
      if ((e as Error).name === 'AbortError') { setPhase('idle'); return }
      setApiError(e instanceof ApiError ? e : String(e)); setPhase('idle')
    } finally { submitting.current = false }
  }

  const busy = phase === 'uploading' || phase === 'processing'

  return (
    <>
      <PageHeader title="Upload Satellite Images" subtitle="Provide a before and an after image of the same area. They must have identical dimensions." />
      {blocked && <div className="banner banner-warn" role="alert">The backend is {connection === 'offline' ? 'unreachable' : 'not ready (models not loaded)'}; analysis is disabled until it is available. See Settings.</div>}

      <Card>
        <div className="grid-2">
          <DropZone role="Before" picked={before} error={errB} maxMb={maxMb} onFile={pick('before')} onClear={() => { setBefore(null); setErrB(null) }} />
          <DropZone role="After" picked={after} error={errA} maxMb={maxMb} onFile={pick('after')} onClear={() => { setAfter(null); setErrA(null) }} />
        </div>
        {mismatch && <p className="inline-error" role="alert">{mismatch}</p>}
      </Card>

      <Card title="Disaster details (optional)">
        <div className="form-grid">
          <label>Disaster name<input value={name} maxLength={100} onChange={(e) => setName(e.target.value)} placeholder="e.g. Midwest flooding" /></label>
          <label>District<input value={district} maxLength={100} onChange={(e) => setDistrict(e.target.value)} placeholder="As you want it shown" /></label>
          <label>State<input value={state} maxLength={100} onChange={(e) => setState(e.target.value)} placeholder="As you want it shown" /></label>
        </div>
        <p className="muted small">These labels are stored with the analysis exactly as typed. They are not verified and are not derived from the imagery.</p>
        <details className="advanced">
          <summary>Advanced settings</summary>
          <div className="form-grid">
            <label>Building detection confidence: {conf.toFixed(2)}
              <input type="range" min={0.05} max={0.9} step={0.05} value={conf} onChange={(e) => setConf(Number(e.target.value))} />
              <small className="muted">Lower finds more buildings but adds false detections. 0.25 is the evaluated default.</small>
            </label>
            <label>Before/after alignment
              <select value={alignment} onChange={(e) => setAlignment(e.target.value as 'check' | 'register')}>
                <option value="check">Check and report offset</option>
                <option value="register">Check and correct translation</option>
              </select>
            </label>
          </div>
        </details>
      </Card>

      <div className="start-row">
        <button className="btn btn-primary btn-lg" disabled={!canStart} onClick={start}>
          {busy ? <Loader2 size={18} className="spin" /> : <UploadCloud size={18} />}
          {busy ? 'Analysing…' : 'Start Disaster Analysis'}
        </button>
        {(!before || !after) && phase === 'idle' && <span className="muted small">Add both images to continue.</span>}
      </div>

      {busy && (
        <Card>
          <div className="progress-block" role="status" aria-live="polite">
            <div className="progress-label">
              <b>{phase === 'uploading' ? `Uploading images… ${Math.round(progress * 100)}%` : 'Processing: detecting buildings and classifying damage…'}</b>
              <span className="muted small">This runs on the backend's models and can take from several seconds to a minute.</span>
            </div>
            <div className={`bar ${phase === 'processing' ? 'indeterminate' : ''}`}><i style={{ width: phase === 'uploading' ? `${progress * 100}%` : '100%' }} /></div>
            <button className="btn btn-ghost" onClick={() => ctl.current?.abort()}>Cancel</button>
          </div>
        </Card>
      )}
      {apiError && <ErrorBanner error={apiError} />}
      {phase === 'done' && doneId && (
        <div className="banner banner-ok" role="status">
          <CheckCircle2 size={20} />
          <div><b>Analysis complete</b><p>Analysis ID: <code>{doneId}</code></p><p className="muted small">Opening the dashboard…</p></div>
          <button className="btn btn-primary" onClick={() => nav('/')}>Open dashboard now</button>
        </div>
      )}
    </>
  )
}
