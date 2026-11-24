import { useState } from 'react'
import { Link } from 'react-router-dom'
import { AlertTriangle, Building2, CheckCircle2, Clock, Info, Lightbulb, Map as MapIcon, MapPin, ShieldAlert, Home as HomeIcon, ScanSearch } from 'lucide-react'
import { url } from '../api/client'
import { ActionCenter } from '../components/ActionCenter'
import { CompareSlider } from '../components/CompareSlider'
import { Card, EmptyState, ErrorBanner, Spinner } from '../components/common'
import { DamageDonut } from '../components/DamageDonut'
import { DamageLegend, useVisibleClasses } from '../components/DamageLegend'
import { DamageMapViewer } from '../components/DamageMapViewer'
import { DAMAGE_META } from '../lib/damage'
import { buildInsights, fmtDateTime, fmtInt, fmtNum, fmtPct, kpis } from '../lib/derive'
import { useAnalysis } from '../state/AnalysisContext'
import { useHealth } from '../state/HealthContext'
import type { Analysis } from '../types'

function Hero() {
  const { connection, health, checkedAt, error } = useHealth()
  const label = { checking: 'Checking…', online: 'System Online', degraded: 'Degraded', offline: 'Backend Offline' }[connection]
  const detail = connection === 'degraded'
    ? (health?.models.load_error ?? 'Models are not loaded.')
    : connection === 'offline' ? (error ?? '') : `Last checked: ${checkedAt ? fmtDateTime(checkedAt.toISOString()) : '—'}`
  return (
    <section className="hero">
      <div className="hero-text">
        <h1>Rapid. Accurate. Actionable.</h1>
        <p>AI-Powered Disaster Assessment for a Safer India</p>
      </div>
      <div className={`sys sys-${connection}`} role="status">
        <span className="sys-dot">{connection === 'online' ? <CheckCircle2 size={22} /> : <AlertTriangle size={22} />}</span>
        <div><b>{label}</b><small>{detail}</small></div>
      </div>
      <span className="credit">Imagery: xBD / Maxar Open Data · CC BY-NC-SA 4.0</span>
    </section>
  )
}

function Kpi({ icon, tone, title, value, sub }: { icon: React.ReactNode; tone: string; title: string; value: string; sub: string }) {
  return (
    <div className="card kpi">
      <div className={`kpi-icon tone-${tone}`}>{icon}</div>
      <div><div className="kpi-title">{title}</div><div className="kpi-value">{value}</div><div className="kpi-sub">{sub}</div></div>
    </div>
  )
}

function Loaded({ a }: { a: Analysis }) {
  const k = kpis(a)
  const { visible, toggle } = useVisibleClasses()
  const [sel, setSel] = useState<number | null>(null)
  const selected = a.buildings.find((b) => b.building_id === sel) ?? null
  const insights = buildInsights(a)
  const { context: ctx, statistics: s } = a
  const [w, h] = [a.image.width, a.image.height]

  return (
    <>
      <div className="meta-strip">
        <span><b>{ctx.disaster_name ?? a.image.after.display_name}</b></span>
        <span className="muted">Analysis {a.analysis_id.slice(0, 8)}… · {fmtDateTime(a.created_utc)}</span>
        <Link to="/results" className="link">Details</Link>
      </div>

      <div className="grid-kpi">
        <Kpi icon={<Building2 size={30} />} tone="blue" title="Total Buildings" value={fmtInt(k.total)} sub="Detected in the after image" />
        <Kpi icon={<AlertTriangle size={30} />} tone="red" title="Affected Buildings" value={fmtInt(k.affected)} sub={`${fmtPct(k.affectedShare)} of total · major + destroyed`} />
        <Kpi icon={<HomeIcon size={30} />} tone="orange" title="Severe Damage" value={fmtInt(k.severe)} sub={`${fmtPct(k.severeShare)} of total · destroyed`} />
        <Kpi icon={<Clock size={30} />} tone="blue" title="Inference Time" value={`${k.seconds.toFixed(1)} sec`} sub={`Model time on ${a.parameters.device}`} />
      </div>

      <div className="grid-main">
        <Card title="Satellite Analysis Results" icon={<ScanSearch size={16} />} className="span-compare">
          <CompareSlider beforeSrc={url.before(a.analysis_id)} afterSrc={url.after(a.analysis_id)} width={w} height={h} />
          <p className="muted small">Drag the divider to compare · Ctrl + scroll to zoom · drag to pan when zoomed.</p>
        </Card>
        <Card title="Damage Classification" icon={<ShieldAlert size={16} />}>
          <DamageDonut analysis={a} />
          <p className="muted small">{fmtInt(k.total)} detected buildings. Affected = major damage + destroyed.</p>
          <h3 className="sub">Prediction confidence</h3>
          <div className="conf-levels">
            {(['high', 'medium', 'low'] as const).map((l) => <div key={l}><b>{fmtInt(s.confidence.levels[l])}</b><span>{l} confidence</span></div>)}
          </div>
          <p className="muted small">Building confidence = detection score × damage-class probability (uncalibrated; low &lt; {s.confidence.low_confidence_threshold}).</p>
        </Card>
        <Card title="Damage Map" icon={<MapIcon size={16} />} actions={<Link className="link" to="/map">Open full map</Link>}>
          <DamageMapViewer analysis={a} visible={visible} selectedId={sel} onSelect={setSel} />
          <DamageLegend visible={visible} toggle={toggle} compact />
          <p className="map-sel">{selected
            ? <>Building #{selected.building_id} · <i style={{ background: DAMAGE_META[selected.predicted_damage].color }} className="dot" />{DAMAGE_META[selected.predicted_damage].label} · confidence {fmtNum(selected.building_confidence)}</>
            : <span className="muted">Hover or click a building for details.</span>}</p>
        </Card>
      </div>

      <div className="grid-bottom">
        <Card title="Key Insights" icon={<Lightbulb size={16} />}>
          <ul className="insights">
            {insights.map((i) => (
              <li key={i.id} className={`tone-${i.tone}`}>{i.tone === 'warn' ? <AlertTriangle size={16} /> : i.tone === 'ok' ? <CheckCircle2 size={16} /> : <Info size={16} />}<span>{i.text}</span></li>
            ))}
          </ul>
          <Link className="btn btn-primary" to="/reports">View Detailed Report →</Link>
        </Card>

        <Card title="Location Summary" icon={<MapPin size={16} />}>
          <dl className="kv">
            <dt>Disaster</dt><dd>{ctx.disaster_name ?? <span className="muted">Not provided</span>}</dd>
            <dt>District</dt><dd>{ctx.district ?? <span className="muted">Not provided</span>}</dd>
            <dt>State</dt><dd>{ctx.state ?? <span className="muted">Not provided</span>}</dd>
            <dt>Affected area</dt><dd><span className="muted">Unavailable (not georeferenced)</span></dd>
            <dt>Mean building confidence</dt><dd>{fmtNum(s.confidence.mean)} <span className="muted small">(uncalibrated score)</span></dd>
            <dt>Image size</dt><dd>{w} × {h} px</dd>
          </dl>
          <p className="notice"><MapPin size={14} /> Geographic location unavailable. District and state, if shown, are exactly what was typed at upload; they are not derived from the imagery.</p>
        </Card>

        <ActionCenter />
      </div>
    </>
  )
}

export function Dashboard() {
  const { status, error, analysis, reload, recent, setCurrent } = useAnalysis()
  return (
    <>
      <Hero />
      {status === 'loading' && <Spinner label="Loading analysis from the backend…" />}
      {status === 'error' && error && (
        <>
          <ErrorBanner error={error} onRetry={error.code === 'analysis_not_found' ? undefined : reload} />
          <div className="stack-row"><button className="btn btn-ghost" onClick={() => setCurrent(null)}>Clear selection</button></div>
        </>
      )}
      {status === 'idle' && (
        <>
          <div className="grid-kpi">
            <Kpi icon={<Building2 size={30} />} tone="blue" title="Total Buildings" value="—" sub="No analysis yet" />
            <Kpi icon={<AlertTriangle size={30} />} tone="red" title="Affected Buildings" value="—" sub="No analysis yet" />
            <Kpi icon={<HomeIcon size={30} />} tone="orange" title="Severe Damage" value="—" sub="No analysis yet" />
            <Kpi icon={<Clock size={30} />} tone="blue" title="Inference Time" value="—" sub="No analysis yet" />
          </div>
          <Card>
            <EmptyState title="No analysis yet" text="Upload a before and after satellite image of the same area. The backend will detect buildings, classify damage and fill this dashboard with real results." />
            {recent.length > 0 && (
              <div className="recent">
                <h3>Recent analyses on this browser</h3>
                <ul>{recent.map((r) => (
                  <li key={r.id}><button className="link" onClick={() => setCurrent(r.id)}>{r.label}</button><span className="muted"> · {fmtInt(r.total)} buildings · {fmtDateTime(r.created_utc)}</span></li>
                ))}</ul>
              </div>
            )}
          </Card>
        </>
      )}
      {status === 'ready' && analysis && <Loaded a={analysis} />}
    </>
  )
}
