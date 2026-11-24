import type { DamageName } from '../types'

// Same three classes and colours as the backend's damage map (reasoning/damage_map.py).
export const DAMAGE_ORDER: DamageName[] = ['no-damage', 'major-damage', 'destroyed']

export const DAMAGE_META: Record<DamageName, { label: string; color: string; short: string }> = {
  'no-damage': { label: 'No Damage', short: 'Undamaged', color: '#4caf50' },
  'major-damage': { label: 'Major Damage', short: 'Major', color: '#ff9800' },
  destroyed: { label: 'Destroyed', short: 'Destroyed', color: '#f44336' },
}

// Affected = major + destroyed (minor damage is merged into major damage).
export const AFFECTED_CLASSES: DamageName[] = ['major-damage', 'destroyed']
