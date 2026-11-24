import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'

export interface Prefs { overlayOpacity: number; showIds: boolean; compact: boolean; highlightLow: boolean }
const DEFAULTS: Prefs = { overlayOpacity: 0.4, showIds: false, compact: false, highlightLow: true }
const KEY = 'eagle-dar.prefs'

function load(): Prefs {
  try {
    return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(KEY) || '{}') }
  } catch {
    return DEFAULTS
  }
}

const Ctx = createContext<{ prefs: Prefs; setPrefs: (p: Partial<Prefs>) => void; reset: () => void } | null>(null)

export function PrefsProvider({ children }: { children: ReactNode }) {
  const [prefs, set] = useState<Prefs>(load)
  useEffect(() => {
    try { localStorage.setItem(KEY, JSON.stringify(prefs)) } catch { /* storage unavailable: preferences just won't persist */ }
    document.documentElement.dataset.density = prefs.compact ? 'compact' : 'comfortable'
  }, [prefs])
  const value = useMemo(() => ({ prefs, setPrefs: (p: Partial<Prefs>) => set((x) => ({ ...x, ...p })), reset: () => set(DEFAULTS) }), [prefs])
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function usePrefs() {
  const v = useContext(Ctx)
  if (!v) throw new Error('usePrefs outside PrefsProvider')
  return v
}
