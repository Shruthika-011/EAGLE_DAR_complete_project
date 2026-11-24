import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from 'recharts'
import { classRows, fmtInt, fmtPct } from '../lib/derive'
import type { Analysis } from '../types'

export function DamageDonut({ analysis }: { analysis: Analysis }) {
  const { total_buildings: total } = analysis.statistics.scene
  const rows = classRows(analysis.statistics.damage_counts, total)
  const affected = analysis.statistics.damaged_fraction
  const data = rows.filter((r) => r.count > 0)

  return (
    <div className="donut-stack">
      <div className="donut" aria-label="Damage class distribution">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie data={total ? data : [{ name: 'none', count: 1, color: '#1d5039' }]} dataKey="count" nameKey="label" innerRadius="62%" outerRadius="92%" startAngle={90} endAngle={-270} stroke="#0c3022" strokeWidth={2} isAnimationActive={false}>
              {(total ? data : [{ color: '#1d5039' }]).map((d, i) => <Cell key={i} fill={d.color} />)}
            </Pie>
            {total > 0 && <Tooltip formatter={(v: number, _n, p) => [`${fmtInt(v)} (${fmtPct(v / total)})`, (p.payload as { label: string }).label]} />}
          </PieChart>
        </ResponsiveContainer>
        <div className="donut-centre"><b>{fmtPct(affected)}</b><span>Affected</span></div>
      </div>
      <table className="class-table">
        <thead><tr><th>Damage class</th><th className="n">Buildings</th><th className="n">Share</th></tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.name}>
              <td><i style={{ background: r.color }} />{r.label}</td>
              <td className="n">{fmtInt(r.count)}</td><td className="n">{fmtPct(r.share)}</td>
            </tr>
          ))}
        </tbody>
        <tfoot>
          <tr><td>Total detected</td><td className="n">{fmtInt(total)}</td><td className="n">{total ? '100.0%' : '—'}</td></tr>
          <tr><td>Affected (major + destroyed)</td><td className="n">{fmtInt(analysis.statistics.damaged_total)}</td><td className="n">{fmtPct(affected)}</td></tr>
        </tfoot>
      </table>
    </div>
  )
}
