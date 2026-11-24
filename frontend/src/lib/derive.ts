// Everything displayed is derived from the backend's analysis. Missing values stay missing (null / '—'); they are never replaced.
import type { Analysis, Building, ClassCounts, Quadrant } from '../types'
import { AFFECTED_CLASSES, DAMAGE_META, DAMAGE_ORDER } from './damage'

export const share = (n: number, d: number): number | null => (d > 0 ? n / d : null)

export function fmtPct(x: number | null | undefined, digits = 1): string {
  return x === null || x === undefined || Number.isNaN(x) ? '—' : `${(x * 100).toFixed(digits)}%`
}
export const fmtInt = (n: number | null | undefined) => (n === null || n === undefined ? '—' : n.toLocaleString('en-IN'))
export const fmtNum = (n: number | null | undefined, digits = 3) => (n === null || n === undefined ? '—' : n.toFixed(digits))

export interface Kpis {
  total: number
  affected: number
  affectedShare: number | null
  severe: number
  severeShare: number | null
  seconds: number
}

export function kpis(a: Analysis): Kpis {
  const c = a.statistics.damage_counts
  const total = a.statistics.scene.total_buildings
  const affected = AFFECTED_CLASSES.reduce((s, k) => s + c[k], 0)
  return {
    total,
    affected,
    affectedShare: share(affected, total),
    severe: c.destroyed,
    severeShare: share(c.destroyed, total),
    seconds: a.parameters.inference_seconds,
  }
}

export interface ClassRow { name: keyof ClassCounts; label: string; color: string; count: number; share: number | null }

export function classRows(counts: ClassCounts, total: number): ClassRow[] {
  return DAMAGE_ORDER.map((name) => ({
    name, label: DAMAGE_META[name].label, color: DAMAGE_META[name].color, count: counts[name], share: share(counts[name], total),
  }))
}

export function confidenceHistogram(buildings: Pick<Building, 'building_confidence'>[], bins = 10) {
  const out = Array.from({ length: bins }, (_, i) => ({
    lo: i / bins, hi: (i + 1) / bins, label: `${(i / bins).toFixed(1)}–${((i + 1) / bins).toFixed(1)}`, count: 0,
  }))
  for (const b of buildings) {
    const c = b.building_confidence
    if (c === null || c === undefined || Number.isNaN(c)) continue
    out[Math.min(bins - 1, Math.max(0, Math.floor(c * bins)))].count += 1
  }
  return out
}

export const QUADRANTS: Quadrant[] = ['top-left', 'top-right', 'bottom-left', 'bottom-right']

export function quadrantRows(a: Analysis) {
  return QUADRANTS.map((q) => ({ quadrant: q, ...a.statistics.quadrants[q] }))
}

export interface Insight { id: string; tone: 'ok' | 'info' | 'warn'; text: string }

/** Plain-language findings computed from the analysis. Each sentence states only what the data supports. */
export function buildInsights(a: Analysis): Insight[] {
  const s = a.statistics
  const k = kpis(a)
  const out: Insight[] = []

  if (k.total === 0) {
    return [{ id: 'none', tone: 'warn', text: 'No buildings were detected. This does not show that the area has no buildings; the detector misses many.' }]
  }
  out.push({ id: 'affected', tone: k.affected > 0 ? 'warn' : 'ok',
    text: `${fmtPct(k.affectedShare)} of detected buildings (${fmtInt(k.affected)} of ${fmtInt(k.total)}) are classified as affected (major or destroyed).` })
  out.push({ id: 'severe', tone: k.severe > 0 ? 'warn' : 'ok',
    text: `${fmtInt(k.severe)} building${k.severe === 1 ? ' is' : 's are'} classified as destroyed (${fmtPct(k.severeShare)} of total).` })

  const m = s.most_damaged_quadrant
  if (s.damaged_total === 0) out.push({ id: 'quad', tone: 'ok', text: 'No affected buildings were detected in any quadrant.' })
  else if (m.answer && m.answer !== 'cannot be determined') {
    const n = m.quadrant_counts?.[m.answer as Quadrant]
    out.push({ id: 'quad', tone: 'info', text: `Most impacted quadrant: ${m.answer}${n !== undefined ? ` (${n} affected buildings)` : ''}.` })
  } else if (m.tie_unresolved) out.push({ id: 'quad', tone: 'info', text: 'No single quadrant has the most affected buildings (tie).' })

  const top = classRows(s.damage_counts, k.total).reduce((p, c) => (c.count > p.count ? c : p))
  out.push({ id: 'dist', tone: 'info', text: `Largest class: ${top.label} — ${fmtPct(top.share)} of detected buildings.` })

  const low = s.confidence.low_confidence_building_ids.length
  if (low > 0) out.push({ id: 'conf', tone: low / k.total > 0.5 ? 'warn' : 'info',
    text: `${fmtInt(low)} building${low === 1 ? ' has' : 's have'} low prediction confidence (< ${s.confidence.low_confidence_threshold}), ${fmtPct(share(low, k.total), 0)} of detected.` })

  if (['misaligned', 'unverified'].includes(a.alignment.status))
    out.push({ id: 'align', tone: 'warn', text: `Before/after alignment is ${a.alignment.status}; damage labels may be less reliable.` })
  return out
}

export function fmtDateTime(iso: string | null | undefined): string {
  if (!iso) return '—'
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? '—' : d.toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' })
}

export const shortId = (id: string) => id.slice(0, 8)
