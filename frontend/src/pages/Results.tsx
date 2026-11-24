import { useMemo, useState } from 'react'
import { ArrowDown, ArrowUp, Search } from 'lucide-react'
import { url } from '../api/client'
import { BuildingDetails } from '../components/BuildingDetails'
import { Card, PageHeader, RequireAnalysis } from '../components/common'
import { DamageLegend, useVisibleClasses } from '../components/DamageLegend'
import { DAMAGE_META } from '../lib/damage'
import { classRows, fmtDateTime, fmtInt, fmtNum, fmtPct, kpis } from '../lib/derive'
import type { Analysis, Building } from '../types'

type SortKey = 'building_id' | 'class_id' | 'building_confidence' | 'area_px'
const PAGE = 15

function Table({ a, onPick, selectedId }: { a: Analysis; onPick: (id: number) => void; selectedId: number | null }) {
  const { visible, toggle, all } = useVisibleClasses()
  const [q, setQ] = useState('')
  const [key, setKey] = useState<SortKey>('building_id')
  const [asc, setAsc] = useState(true)
  const [page, setPage] = useState(0)

  const rows = useMemo(() => {
    const term = q.trim()
    const f = a.buildings.filter((b) => visible.has(b.predicted_damage) && (!term || String(b.building_id).startsWith(term)))
    const val = (b: Building) => (b[key] ?? -1) as number
    return f.sort((x, y) => (asc ? 1 : -1) * (val(x) - val(y)))
  }, [a.buildings, visible, q, key, asc])
  const pages = Math.max(1, Math.ceil(rows.length / PAGE))
  const cur = Math.min(page, pages - 1)
  const slice = rows.slice(cur * PAGE, cur * PAGE + PAGE)
  const sort = (k: SortKey) => { k === key ? setAsc(!asc) : (setKey(k), setAsc(true)); setPage(0) }
  const Th = ({ k, children }: { k: SortKey; children: string }) => (
    <th><button className="th-btn" onClick={() => sort(k)}>{children}{key === k && (asc ? <ArrowUp size={12} /> : <ArrowDown size={12} />)}</button></th>
  )

  return (
    <>
      <div className="table-tools">
        <label className="search"><Search size={14} /><input placeholder="Building ID starts with…" value={q} inputMode="numeric" onChange={(e) => { setQ(e.target.value.replace(/\D/g, '')); setPage(0) }} /></label>
        <DamageLegend visible={visible} toggle={(n) => { toggle(n); setPage(0) }} counts={a.statistics.damage_counts} compact />
        <button className="btn btn-ghost" onClick={all}>Show all</button>
      </div>
      <div className="table-wrap">
        <table className="tbl">
          <thead><tr><Th k="building_id">ID</Th><Th k="class_id">Damage class</Th><Th k="building_confidence">Confidence</Th><th>Level</th><Th k="area_px">Area px²</Th><th>Quadrant</th><th>Bounding box</th></tr></thead>
          <tbody>
            {slice.map((b) => (
              <tr key={b.building_id} className={b.building_id === selectedId ? 'sel' : ''} onClick={() => onPick(b.building_id)} tabIndex={0} onKeyDown={(e) => e.key === 'Enter' && onPick(b.building_id)}>
                <td>#{b.building_id}</td>
                <td><i className="dot" style={{ background: DAMAGE_META[b.predicted_damage].color }} />{DAMAGE_META[b.predicted_damage].label}</td>
                <td className="n">{fmtNum(b.building_confidence)}</td>
                <td>{b.confidence_level ? <span className={`lvl lvl-${b.confidence_level}`}>{b.confidence_level}</span> : '—'}</td>
                <td className="n">{b.area_px === null ? '—' : b.area_px.toFixed(0)}</td>
                <td>{b.quadrant}</td>
                <td className="mono">{b.bbox_xyxy.join(', ')}</td>
              </tr>
            ))}
            {slice.length === 0 && <tr><td colSpan={7} className="muted pad">No buildings match the current filters.</td></tr>}
          </tbody>
        </table>
      </div>
      <div className="pager">
        <span className="muted small">{fmtInt(rows.length)} of {fmtInt(a.buildings.length)} buildings</span>
        <div className="stack-row">
          <button className="btn btn-ghost" disabled={cur === 0} onClick={() => setPage(cur - 1)}>Previous</button>
          <span className="small">Page {cur + 1} / {pages}</span>
          <button className="btn btn-ghost" disabled={cur >= pages - 1} onClick={() => setPage(cur + 1)}>Next</button>
        </div>
      </div>
    </>
  )
}

function Body({ a }: { a: Analysis }) {
  const [sel, setSel] = useState<number | null>(null)
  const building = a.buildings.find((b) => b.building_id === sel) ?? null
  const k = kpis(a)
  const rows = classRows(a.statistics.damage_counts, k.total)
  const al = a.alignment

  return (
    <>
      <PageHeader title="View Results" subtitle={<>{a.context.disaster_name ?? a.image.after.display_name} · Analysis <code>{a.analysis_id}</code></>} />
      <div className="grid-2">
        <Card title="Before disaster"><img className="photo" src={url.before(a.analysis_id)} alt="Before disaster" /></Card>
        <Card title="After disaster"><img className="photo" src={url.after(a.analysis_id)} alt="After disaster" /></Card>
      </div>

      <div className="grid-2-1">
        <Card title="Damage class distribution">
          <table className="tbl"><thead><tr><th>Class</th><th className="n">Buildings</th><th className="n">Share</th><th /></tr></thead>
            <tbody>{rows.map((r) => (
              <tr key={r.name}><td><i className="dot" style={{ background: r.color }} />{r.label}</td><td className="n">{fmtInt(r.count)}</td><td className="n">{fmtPct(r.share)}</td>
                <td className="bar-cell"><div className="mini-bar"><i style={{ width: `${(r.share ?? 0) * 100}%`, background: r.color }} /></div></td></tr>))}
              <tr className="total"><td>Total detected</td><td className="n">{fmtInt(k.total)}</td><td /><td /></tr>
              <tr><td>Affected (major + destroyed)</td><td className="n">{fmtInt(k.affected)}</td><td className="n">{fmtPct(k.affectedShare)}</td><td /></tr></tbody></table>
        </Card>
        <Card title="Analysis metadata">
          <dl className="kv">
            <dt>Analysis ID</dt><dd className="mono wrap">{a.analysis_id}</dd>
            <dt>Created</dt><dd>{fmtDateTime(a.created_utc)}</dd>
            <dt>Image size</dt><dd>{a.image.width} × {a.image.height} px</dd>
            <dt>Before file</dt><dd>{a.image.before.display_name}</dd>
            <dt>After file</dt><dd>{a.image.after.display_name}</dd>
            <dt>Detection conf.</dt><dd>{a.parameters.conf} · imgsz {a.parameters.imgsz}</dd>
            <dt>Device</dt><dd>{a.parameters.device}</dd>
            <dt>Inference time</dt><dd>{a.parameters.inference_seconds.toFixed(2)} s</dd>
            <dt>Alignment</dt><dd>{al.status} · {al.shift_px.toFixed(1)} px via {al.method}{al.registered ? ' (registered)' : ''}</dd>
          </dl>
          {al.note && <p className="notice">{al.note}</p>}
          {a.warnings.map((w) => <p key={w} className="notice">{w}</p>)}
        </Card>
      </div>

      <div className="grid-table">
        <Card title={`Building predictions (${fmtInt(a.buildings.length)})`}>
          <Table a={a} onPick={setSel} selectedId={sel} />
        </Card>
        <Card title="Inspector"><BuildingDetails analysis={a} building={building} onClose={() => setSel(null)} /></Card>
      </div>
      <p className="muted small">{a.disclaimer}</p>
    </>
  )
}

export function Results() {
  return <RequireAnalysis>{(a) => <Body a={a} />}</RequireAnalysis>
}
