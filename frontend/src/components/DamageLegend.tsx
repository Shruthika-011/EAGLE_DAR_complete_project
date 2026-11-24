import { useCallback, useState } from 'react'
import { DAMAGE_META, DAMAGE_ORDER } from '../lib/damage'
import type { ClassCounts, DamageName } from '../types'

export function useVisibleClasses() {
  const [visible, setVisible] = useState<Set<DamageName>>(() => new Set(DAMAGE_ORDER))
  const toggle = useCallback((n: DamageName) => setVisible((s) => { const x = new Set(s); x.has(n) ? x.delete(n) : x.add(n); return x }), [])
  const all = useCallback(() => setVisible(new Set(DAMAGE_ORDER)), [])
  return { visible, toggle, all }
}

/** Legend that doubles as per-class show/hide toggles. */
export function DamageLegend({ visible, toggle, counts, compact = false }: { visible: Set<DamageName>; toggle: (n: DamageName) => void; counts?: ClassCounts; compact?: boolean }) {
  return (
    <div className={`legend ${compact ? 'legend-compact' : ''}`} role="group" aria-label="Damage classes (click to show or hide)">
      {DAMAGE_ORDER.map((n) => {
        const on = visible.has(n)
        return (
          <button key={n} type="button" className={`chip ${on ? '' : 'chip-off'}`} aria-pressed={on} onClick={() => toggle(n)} title={on ? 'Hide this class' : 'Show this class'}>
            <i style={{ background: DAMAGE_META[n].color }} />
            {compact ? DAMAGE_META[n].short : DAMAGE_META[n].label}
            {counts && <span className="chip-n">{counts[n]}</span>}
          </button>
        )
      })}
    </div>
  )
}
