import { useMemo } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Clock } from 'lucide-react'
import { Card, PageHeader, RequireAnalysis } from '../components/common'
import { DamageDonut } from '../components/DamageDonut'
import { DAMAGE_META, DAMAGE_ORDER } from '../lib/damage'
import { classRows, confidenceHistogram, fmtInt, fmtNum, fmtPct, kpis, quadrantRows } from '../lib/derive'
import type { Analysis } from '../types'

const AXIS = { fontSize: 12, fill: '#9dc2b0' }

function Body({ a }: { a: Analysis }) {
  const k = kpis(a)
  const rows = classRows(a.statistics.damage_counts, k.total)
  const hist = useMemo(() => confidenceHistogram(a.buildings), [a.buildings])
  const quads = quadrantRows(a).map((q) => ({ name: q.quadrant, ...q }))
  const totalVsAffected = [
    { name: 'Total detected', count: k.total, color: '#22a05a' },
    { name: 'Affected', count: k.affected, color: '#f97316' },
    { name: 'Destroyed', count: k.severe, color: DAMAGE_META.destroyed.color },
  ]
  const confByClass = DAMAGE_ORDER.map((n) => {
    const v = a.buildings.filter((b) => b.predicted_damage === n).map((b) => b.building_confidence)
    return { name: n, n: v.length, mean: v.length ? v.reduce((s, x) => s + x, 0) / v.length : null }
  })

  return (
    <>
      <PageHeader title="Analytics" subtitle="Computed from this analysis's building-level predictions. With a single analysis there is nothing to trend or compare over time." />
      <div className="grid-kpi">
        <div className="card kpi"><div className="kpi-icon tone-blue"><Clock size={28} /></div><div><div className="kpi-title">Inference time</div><div className="kpi-value">{k.seconds.toFixed(1)} sec</div><div className="kpi-sub">Model time on {a.parameters.device}, one image pair</div></div></div>
        <div className="card kpi"><div><div className="kpi-title">Buildings detected</div><div className="kpi-value">{fmtInt(k.total)}</div><div className="kpi-sub">All three classes</div></div></div>
        <div className="card kpi"><div><div className="kpi-title">Affected share</div><div className="kpi-value">{fmtPct(k.affectedShare)}</div><div className="kpi-sub">major + destroyed</div></div></div>
        <div className="card kpi"><div><div className="kpi-title">Mean confidence</div><div className="kpi-value">{fmtNum(a.statistics.confidence.mean)}</div><div className="kpi-sub">uncalibrated building score</div></div></div>
      </div>

      <div className="grid-2">
        <Card title="Damage distribution"><DamageDonut analysis={a} /></Card>
        <Card title="Total versus affected buildings">
          <div className="chart"><ResponsiveContainer><BarChart data={totalVsAffected} layout="vertical" margin={{ left: 20, right: 24 }}>
            <CartesianGrid horizontal={false} stroke="#1d5039" /><XAxis type="number" tick={AXIS} allowDecimals={false} /><YAxis type="category" dataKey="name" tick={AXIS} width={110} />
            <Tooltip /><Bar dataKey="count" name="Buildings" isAnimationActive={false}>{totalVsAffected.map((d) => <Cell key={d.name} fill={d.color} />)}</Bar>
          </BarChart></ResponsiveContainer></div>
        </Card>
        <Card title="Damage class comparison">
          <div className="chart"><ResponsiveContainer><BarChart data={rows} margin={{ right: 16 }}>
            <CartesianGrid vertical={false} stroke="#1d5039" /><XAxis dataKey="label" tick={AXIS} interval={0} /><YAxis tick={AXIS} allowDecimals={false} />
            <Tooltip /><Bar dataKey="count" name="Buildings" isAnimationActive={false}>{rows.map((r) => <Cell key={r.name} fill={r.color} />)}</Bar>
          </BarChart></ResponsiveContainer></div>
        </Card>
        <Card title="Confidence distribution">
          <div className="chart"><ResponsiveContainer><BarChart data={hist} margin={{ right: 16 }}>
            <CartesianGrid vertical={false} stroke="#1d5039" /><XAxis dataKey="label" tick={{ ...AXIS, fontSize: 10 }} interval={0} /><YAxis tick={AXIS} allowDecimals={false} />
            <Tooltip /><Bar dataKey="count" name="Buildings" fill="#22a05a" isAnimationActive={false} />
          </BarChart></ResponsiveContainer></div>
          <p className="muted small">Building confidence = detection score × damage-class probability. Low values are common with the current models (thresholds: low &lt; {a.statistics.confidence.low_confidence_threshold}).</p>
        </Card>
      </div>

      <Card title="Quadrant-wise damage statistics">
        <div className="chart chart-tall"><ResponsiveContainer><BarChart data={quads} margin={{ right: 16 }}>
          <CartesianGrid vertical={false} stroke="#1d5039" /><XAxis dataKey="name" tick={AXIS} /><YAxis tick={AXIS} allowDecimals={false} /><Tooltip /><Legend />
          {DAMAGE_ORDER.map((n) => <Bar key={n} dataKey={n} name={DAMAGE_META[n].label} stackId="q" fill={DAMAGE_META[n].color} isAnimationActive={false} />)}
        </BarChart></ResponsiveContainer></div>
        <div className="table-wrap"><table className="tbl"><thead><tr><th>Quadrant</th>{DAMAGE_ORDER.map((n) => <th key={n} className="n">{DAMAGE_META[n].short}</th>)}<th className="n">Affected</th></tr></thead>
          <tbody>{quads.map((q) => <tr key={q.name}><td>{q.name}</td>{DAMAGE_ORDER.map((n) => <td key={n} className="n">{q[n]}</td>)}<td className="n"><b>{q.damaged}</b></td></tr>)}</tbody></table></div>
        <p className="muted small">A building belongs to the quadrant containing its centre; the image is split at its centre.</p>
      </Card>

      <Card title="Mean confidence by predicted class">
        <table className="tbl"><thead><tr><th>Predicted class</th><th className="n">Buildings</th><th className="n">Mean confidence</th></tr></thead>
          <tbody>{confByClass.map((c) => <tr key={c.name}><td><i className="dot" style={{ background: DAMAGE_META[c.name].color }} />{DAMAGE_META[c.name].label}</td><td className="n">{c.n}</td><td className="n">{fmtNum(c.mean)}</td></tr>)}</tbody></table>
      </Card>
    </>
  )
}

export function Analytics() {
  return <RequireAnalysis>{(a) => <Body a={a} />}</RequireAnalysis>
}
