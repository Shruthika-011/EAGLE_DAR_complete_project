import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { useLocation, useSearchParams } from 'react-router-dom'
import { ApiError, createReport, getAnalysis, getQuestions, getReportHtml, getReportJson, getReportMeta, listReports } from '../api/client'
import type { Analysis, QuestionRef, ReportJson, ReportMeta } from '../types'

const KEY = 'eagle-dar.currentAnalysis'
const RECENT = 'eagle-dar.recentAnalyses'
const ID_RE = /^[0-9a-f]{32}$/
const POLL_MS = 700

export interface RecentItem { id: string; created_utc: string; label: string; total: number }
export type LoadStatus = 'idle' | 'loading' | 'ready' | 'error'

/** The PDF report (Novelty 3). `meta` is exactly what the backend reports; nothing is inferred client-side. */
export interface PdfState { status: 'idle' | 'starting' | 'queued' | 'generating' | 'complete' | 'failed'; meta: ReportMeta | null; error: string | null }
/** Lightweight HTML/JSON quick view (same findings and recommendations as the PDF). */
export interface QuickState { status: 'idle' | 'loading' | 'ready' | 'error'; html: string | null; json: ReportJson | null; error: string | null }

interface AnalysisState {
  currentId: string | null
  analysis: Analysis | null
  status: LoadStatus
  error: ApiError | null
  recent: RecentItem[]
  setCurrent: (id: string | null) => void
  reload: () => void
  pdf: PdfState
  reports: ReportMeta[]
  generatePdf: () => Promise<void>
  quick: QuickState
  loadQuick: (refresh?: boolean) => Promise<void>
  questions: QuestionRef[]
  refreshQuestions: () => void
}

const Ctx = createContext<AnalysisState | null>(null)
const idlePdf: PdfState = { status: 'idle', meta: null, error: null }
const idleQuick: QuickState = { status: 'idle', html: null, json: null, error: null }

const readJson = <T,>(k: string, fallback: T): T => {
  try { return JSON.parse(localStorage.getItem(k) || '') as T } catch { return fallback }
}
const writeStore = (k: string, v: string | null) => {
  try { v === null ? localStorage.removeItem(k) : localStorage.setItem(k, v) } catch { /* storage unavailable */ }
}
const msg = (e: unknown) => (e instanceof ApiError ? e.message : String(e))
const inFlight = (s: ReportMeta['status']) => s === 'queued' || s === 'generating'

export function AnalysisProvider({ children }: { children: ReactNode }) {
  const [params, setParams] = useSearchParams()
  const location = useLocation()
  const [currentId, setId] = useState<string | null>(() => {
    const fromUrl = params.get('a')
    const stored = localStorage.getItem(KEY)
    return fromUrl && ID_RE.test(fromUrl) ? fromUrl : stored && ID_RE.test(stored) ? stored : null
  })
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [status, setStatus] = useState<LoadStatus>(currentId ? 'loading' : 'idle')
  const [error, setError] = useState<ApiError | null>(null)
  const [recent, setRecent] = useState<RecentItem[]>(() => readJson<RecentItem[]>(RECENT, []))
  const [pdf, setPdf] = useState<PdfState>(idlePdf)
  const [reports, setReports] = useState<ReportMeta[]>([])
  const [quick, setQuick] = useState<QuickState>(idleQuick)
  const [questions, setQuestions] = useState<QuestionRef[]>([])
  const [nonce, setNonce] = useState(0)
  const [qNonce, setQNonce] = useState(0)
  const starting = useRef(false)                     // guards against duplicate "generate" requests

  const remember = useCallback((a: Analysis) => {
    const item: RecentItem = { id: a.analysis_id, created_utc: a.created_utc, total: a.statistics.scene.total_buildings, label: a.context.disaster_name || a.image.after.display_name }
    setRecent((r) => {
      const next = [item, ...r.filter((x) => x.id !== item.id)].slice(0, 8)
      writeStore(RECENT, JSON.stringify(next))
      return next
    })
  }, [])

  // ---- load the current analysis (always with polygons: POST /api/analyze returns buildings without them) ----------
  useEffect(() => {
    setPdf(idlePdf); setReports([]); setQuick(idleQuick); setQuestions([])
    if (!currentId) { setAnalysis(null); setStatus('idle'); setError(null); return }
    const ctl = new AbortController()
    setStatus('loading'); setError(null)
    getAnalysis(currentId, ctl.signal)
      .then((a) => { setAnalysis(a); setStatus('ready'); remember(a) })
      .catch((e) => {
        if ((e as Error).name === 'AbortError') return
        setAnalysis(null); setStatus('error')
        setError(e instanceof ApiError ? e : new ApiError('unknown', String(e)))
      })
    return () => ctl.abort()
  }, [currentId, nonce, remember])

  // ---- existing reports for this analysis: resume an in-flight one, or show the newest finished one --------------
  useEffect(() => {
    if (!currentId || status !== 'ready') return
    const ctl = new AbortController()
    listReports(currentId, ctl.signal).then((list) => {
      setReports(list)
      const latest = list[0]
      if (latest) setPdf({ status: latest.status, meta: latest, error: latest.error })
    }).catch(() => { /* the list is optional; the Reports page shows a real error when the user acts */ })
    return () => ctl.abort()
  }, [currentId, status])

  // ---- evidence references (which questions cite which buildings) ------------------------------------------------
  useEffect(() => {
    if (!currentId || status !== 'ready') return
    const ctl = new AbortController()
    getQuestions(currentId, ctl.signal).then((r) => setQuestions(r.questions)).catch(() => setQuestions([]))
    return () => ctl.abort()
  }, [currentId, status, qNonce])

  // ---- poll while a report is being generated (real backend progress) ------------------------------------------
  const pollId = pdf.meta && inFlight(pdf.meta.status) ? pdf.meta.report_id : null
  useEffect(() => {
    if (!pollId) return
    let stop = false
    const tick = async () => {
      try {
        const m = await getReportMeta(pollId)
        if (stop) return
        setPdf({ status: m.status, meta: m, error: m.error })
        if (!inFlight(m.status)) setReports((l) => [m, ...l.filter((x) => x.report_id !== m.report_id)])
      } catch (e) {
        if (!stop) setPdf((p) => ({ ...p, status: 'failed', error: msg(e) }))
      }
    }
    const t = setInterval(tick, POLL_MS)
    return () => { stop = true; clearInterval(t) }
  }, [pollId])

  // ---- keep the id in the URL (shareable) and localStorage (survives reload) ------------------------------------
  useEffect(() => {
    writeStore(KEY, currentId)
    if (currentId && params.get('a') !== currentId) setParams((p) => { p.set('a', currentId); return p }, { replace: true })
    else if (!currentId && params.has('a')) setParams((p) => { p.delete('a'); return p }, { replace: true })
  }, [currentId, location.pathname]) // eslint-disable-line react-hooks/exhaustive-deps

  const setCurrent = useCallback((id: string | null) => setId(id), [])
  const reload = useCallback(() => setNonce((n) => n + 1), [])
  const refreshQuestions = useCallback(() => { setQNonce((n) => n + 1); setQuick(idleQuick) }, [])

  const generatePdf = useCallback(async () => {
    if (!currentId || starting.current) return
    starting.current = true
    setPdf({ status: 'starting', meta: null, error: null })
    try {
      const m = await createReport(currentId)             // 202 now; the PDF is rendered in the background
      setPdf({ status: m.status, meta: m, error: m.error })
    } catch (e) {
      setPdf({ status: 'failed', meta: null, error: msg(e) })
    } finally { starting.current = false }
  }, [currentId])

  const loadQuick = useCallback(async (refresh = false) => {
    if (!currentId) return
    setQuick((q) => ({ ...q, status: 'loading', error: null }))
    try {
      const html = await getReportHtml(currentId, refresh)
      const json = await getReportJson(currentId)
      setQuick({ status: 'ready', html, json, error: null })
    } catch (e) { setQuick({ status: 'error', html: null, json: null, error: msg(e) }) }
  }, [currentId])

  const value = useMemo(
    () => ({ currentId, analysis, status, error, recent, setCurrent, reload, pdf, reports, generatePdf, quick, loadQuick, questions, refreshQuestions }),
    [currentId, analysis, status, error, recent, setCurrent, reload, pdf, reports, generatePdf, quick, loadQuick, questions, refreshQuestions],
  )
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useAnalysis(): AnalysisState {
  const v = useContext(Ctx)
  if (!v) throw new Error('useAnalysis outside AnalysisProvider')
  return v
}
