import { describe, expect, it } from 'vitest'
import { buildInsights, classRows, confidenceHistogram, fmtPct, kpis, share } from './derive'
import { clampView, cropStyle, imageToBox, zoomAt } from './geometry'
import { AFFECTED_CLASSES, DAMAGE_META, DAMAGE_ORDER } from './damage'
import type { Analysis, ClassCounts } from '../types'

// Synthetic shapes only to exercise pure functions; the UI itself never uses sample data.
const counts = (o: Partial<ClassCounts>): ClassCounts => ({ 'no-damage': 0, 'major-damage': 0, destroyed: 0, ...o })
function fake(c: ClassCounts, extra: Partial<Analysis['statistics']> = {}, low: number[] = []): Analysis {
  const total = Object.values(c).reduce((a, b) => a + b, 0)
  const damaged = c['major-damage'] + c.destroyed
  return {
    parameters: { inference_seconds: 4.2, conf: 0.25, imgsz: 640, alignment: 'check', device: 'cpu' },
    alignment: { status: 'aligned' },
    statistics: {
      scene: { total_buildings: total, width: 1024, height: 1024, scene_id: 'x' }, damage_counts: c, damaged_total: damaged,
      damaged_fraction: total ? damaged / total : null,
      most_damaged_quadrant: { answer: 'cannot be determined', region: null, supporting_building_ids: [], tie_unresolved: false, quadrant_counts: null },
      confidence: { low_confidence_threshold: 0.35, low_confidence_building_ids: low, mean: 0.3, min: 0.1, max: 0.6, levels: { high: 0, medium: 0, low: low.length } },
      ...extra,
    },
  } as unknown as Analysis
}

describe('damage definitions', () => {
  it('has the three classes in project order with distinct colours', () => {
    expect(DAMAGE_ORDER).toEqual(['no-damage', 'major-damage', 'destroyed'])
    expect(new Set(DAMAGE_ORDER.map((k) => DAMAGE_META[k].color)).size).toBe(3)
  })
  it('no-damage is not affected', () => {
    expect(AFFECTED_CLASSES).toEqual(['major-damage', 'destroyed'])
  })
})

describe('kpis', () => {
  it('affected = major+destroyed', () => {
    const k = kpis(fake(counts({ 'no-damage': 80, 'major-damage': 15, destroyed: 5 })))
    expect(k.total).toBe(100)
    expect(k.affected).toBe(20)
    expect(k.affectedShare).toBeCloseTo(0.2)
    expect(k.severe).toBe(5)
    expect(k.seconds).toBe(4.2)
  })
  it('has null shares for an empty analysis instead of NaN or 0', () => {
    const k = kpis(fake(counts({})))
    expect(k.affectedShare).toBeNull()
    expect(fmtPct(k.affectedShare)).toBe('—')
  })
  it('share/fmtPct', () => {
    expect(share(1, 0)).toBeNull()
    expect(fmtPct(0.12345)).toBe('12.3%')
    expect(fmtPct(undefined)).toBe('—')
  })
})

describe('classRows', () => {
  it('lists all three classes with counts and shares that sum to 1', () => {
    const rows = classRows(counts({ 'no-damage': 7, destroyed: 3 }), 10)
    expect(rows.map((r) => r.name)).toEqual(DAMAGE_ORDER)
    expect(rows.reduce((s, r) => s + (r.share ?? 0), 0)).toBeCloseTo(1)
    expect(rows[1].count).toBe(0)
  })
})

describe('confidenceHistogram', () => {
  it('bins real values, including 1.0 in the last bin, and counts every building once', () => {
    const h = confidenceHistogram([{ building_confidence: 0 }, { building_confidence: 0.05 }, { building_confidence: 0.1 }, { building_confidence: 0.99 }, { building_confidence: 1 }])
    expect(h).toHaveLength(10)
    expect(h[0].count).toBe(2)
    expect(h[1].count).toBe(1)
    expect(h[9].count).toBe(2)
    expect(h.reduce((s, b) => s + b.count, 0)).toBe(5)
  })
})

describe('buildInsights', () => {
  it('reports an empty detection honestly', () => {
    const i = buildInsights(fake(counts({})))
    expect(i).toHaveLength(1)
    expect(i[0].tone).toBe('warn')
  })
  it('derives sentences from the data', () => {
    const a = fake(counts({ 'no-damage': 8, destroyed: 2 }), {
      most_damaged_quadrant: { answer: 'top-left', region: null, supporting_building_ids: [], tie_unresolved: false, quadrant_counts: { 'top-left': 2, 'top-right': 0, 'bottom-left': 0, 'bottom-right': 0 } },
    }, [1, 2, 3])
    const text = buildInsights(a).map((x) => x.text).join(' | ')
    expect(text).toContain('20.0%')
    expect(text).toContain('2 buildings are classified as destroyed')
    expect(text).toContain('top-left (2 affected buildings)')
    expect(text).toContain('3 buildings have low prediction confidence')
  })
  it('does not invent a most-impacted quadrant when nothing is damaged', () => {
    const text = buildInsights(fake(counts({ 'no-damage': 5 }))).map((x) => x.text).join(' ')
    expect(text).toContain('No affected buildings')
    expect(text).not.toContain('Most impacted')
  })
  it('warns when alignment is unverified', () => {
    const a = fake(counts({ 'no-damage': 5 })); a.alignment.status = 'unverified'
    expect(buildInsights(a).some((i) => i.id === 'align')).toBe(true)
  })
})

describe('zoom / pan / coordinate maths', () => {
  it('maps image pixels onto the displayed box proportionally', () => {
    expect(imageToBox(512, 256, 1024, 1024, 500, 400)).toEqual({ x: 250, y: 100 })
    expect(imageToBox(1024, 1024, 1024, 1024, 500, 400)).toEqual({ x: 500, y: 400 })
  })
  it('zooming keeps the point under the cursor fixed', () => {
    const v = zoomAt({ scale: 1, x: 0, y: 0 }, 2, 300, 200, 600, 400)
    expect(v.scale).toBe(2)
    // content point under cursor: (cx - x) / scale
    expect((300 - v.x) / v.scale).toBeCloseTo(300)
    expect((200 - v.y) / v.scale).toBeCloseTo(200)
  })
  it('never pans past the content or zooms out below 1x', () => {
    expect(clampView({ scale: 2, x: 50, y: 50 }, 600, 400)).toEqual({ scale: 2, x: 0, y: 0 })
    expect(clampView({ scale: 2, x: -9999, y: -9999 }, 600, 400)).toEqual({ scale: 2, x: -600, y: -400 })
    expect(zoomAt({ scale: 1, x: 0, y: 0 }, 0.5, 10, 10, 600, 400).scale).toBe(1)
    expect(zoomAt({ scale: 9, x: 0, y: 0 }, 5, 10, 10, 600, 400).scale).toBe(10)
  })
  it('crop thumbnail shows the bbox region scaled to the thumbnail width', () => {
    const s = cropStyle([100, 100, 200, 150], 1000, 1000, 100, 0)
    expect(s.width).toBe(100)
    expect(s.height).toBe(50)
    expect(s.backgroundSize).toBe('1000px 1000px')
    expect(s.backgroundPosition).toBe('-100px -100px')
  })
})
