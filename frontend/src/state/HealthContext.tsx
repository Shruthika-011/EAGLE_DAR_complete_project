import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { ApiError, getHealth } from '../api/client'
import type { Health } from '../types'

export type Connection = 'checking' | 'online' | 'degraded' | 'offline'
interface HealthState {
  connection: Connection
  health: Health | null
  error: string | null
  checkedAt: Date | null
  refresh: () => void
}

const Ctx = createContext<HealthState | null>(null)
const POLL_MS = 20_000

export function HealthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<Omit<HealthState, 'refresh'>>({ connection: 'checking', health: null, error: null, checkedAt: null })
  const ctl = useRef<AbortController | null>(null)

  const refresh = useCallback(() => {
    ctl.current?.abort()
    const c = new AbortController()
    ctl.current = c
    getHealth(c.signal)
      .then((h) => setState({ connection: h.status === 'ok' ? 'online' : 'degraded', health: h, error: null, checkedAt: new Date() }))
      .catch((e) => {
        if ((e as Error).name === 'AbortError') return
        setState({ connection: 'offline', health: null, error: e instanceof ApiError ? e.message : String(e), checkedAt: new Date() })
      })
  }, [])

  useEffect(() => {
    refresh()
    const t = setInterval(refresh, POLL_MS)
    return () => { clearInterval(t); ctl.current?.abort() }
  }, [refresh])

  const value = useMemo(() => ({ ...state, refresh }), [state, refresh])
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useHealth(): HealthState {
  const v = useContext(Ctx)
  if (!v) throw new Error('useHealth outside HealthProvider')
  return v
}
