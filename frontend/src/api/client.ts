// Reusable API layer. Every call goes to the FastAPI backend; nothing here fabricates data.
import type { Analysis, Health, QuestionRef, QuestionResult, ReportJson, ReportMeta } from '../types'

export const API_BASE: string = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/+$/, '') || 'http://127.0.0.1:8000'

export class ApiError extends Error {
  constructor(
    public code: string,
    message: string,
    public status = 0,
    public details?: Record<string, unknown>,
  ) {
    super(message)
  }
}

const networkError = () =>
  new ApiError('network_error', `Cannot reach the backend at ${API_BASE}. Is it running, and is this origin allowed by its CORS list?`)

async function toError(res: Response): Promise<ApiError> {
  try {
    const body = await res.json()
    const e = body?.error
    if (e?.code) return new ApiError(e.code, e.message, res.status, e.details)
  } catch {
    /* not the backend's error envelope */
  }
  return new ApiError('http_error', `Backend returned HTTP ${res.status}.`, res.status)
}

async function request(path: string, init?: RequestInit): Promise<Response> {
  let res: Response
  try {
    res = await fetch(`${API_BASE}${path}`, init)
  } catch (e) {
    if ((e as Error).name === 'AbortError') throw e
    throw networkError()
  }
  if (!res.ok) throw await toError(res)
  return res
}

const json = async <T,>(path: string, init?: RequestInit) => (await request(path, init)).json() as Promise<T>

// ---- URLs (used directly as <img src> / download links) -------------------
export const url = {
  before: (id: string) => `${API_BASE}/api/analysis/${id}/image/before`,
  after: (id: string) => `${API_BASE}/api/analysis/${id}/image/after`,
  map: (id: string) => `${API_BASE}/api/analysis/${id}/map`,
  comparison: (id: string) => `${API_BASE}/api/analysis/${id}/comparison`,
  export: (id: string) => `${API_BASE}/api/analysis/${id}/export`,
  docs: `${API_BASE}/docs`,
  reportPdf: (reportId: string, inline = false) => `${API_BASE}/api/reports/${reportId}/download${inline ? '?inline=true' : ''}`,
}

// ---- endpoints -------------------------------------------------------------
export const getHealth = (signal?: AbortSignal) => {
  // /api/health answers 503 with a full body when degraded; that body is still the status we want.
  return fetch(`${API_BASE}/api/health`, { signal })
    .then(async (r) => {
      if (r.status === 200 || r.status === 503) return (await r.json()) as Health
      throw await toError(r)
    })
    .catch((e) => {
      if ((e as Error).name === 'AbortError' || e instanceof ApiError) throw e
      throw networkError()
    })
}

export const getAnalysis = (id: string, signal?: AbortSignal) =>
  json<Analysis>(`/api/analysis/${id}?include_polygons=true`, { signal })

export const getReportJson = (id: string, refresh = false, signal?: AbortSignal) =>
  json<ReportJson>(`/api/analysis/${id}/report?format=json${refresh ? '&refresh=true' : ''}`, { signal })

export const getReportHtml = async (id: string, refresh = false, signal?: AbortSignal) =>
  (await request(`/api/analysis/${id}/report?format=html${refresh ? '&refresh=true' : ''}`, { signal })).text()

export const askQuestions = (analysisId: string, questions: { id?: string; question: string }[], signal?: AbortSignal) =>
  json<{ analysis_id: string; results: QuestionResult[] }>('/api/questions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ analysis_id: analysisId, questions }),
    signal,
  })

export const createReport = (analysisId: string) =>
  json<ReportMeta>(`/api/analysis/${analysisId}/reports`, { method: 'POST' })

export const getReportMeta = (reportId: string, signal?: AbortSignal) => json<ReportMeta>(`/api/reports/${reportId}`, { signal })

export const listReports = (analysisId: string, signal?: AbortSignal) => json<ReportMeta[]>(`/api/analysis/${analysisId}/reports`, { signal })

export const getQuestions = (analysisId: string, signal?: AbortSignal) =>
  json<{ analysis_id: string; questions: QuestionRef[] }>(`/api/analysis/${analysisId}/questions`, { signal })

export const getExport = (id: string) => json<{ id: string; answer: number | string; region: unknown }[]>(`/api/analysis/${id}/export`)

export async function downloadBlob(path: string, filename: string): Promise<void> {
  const blob = await (await request(path)).blob()
  const href = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = href
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(href), 1000)
}

// ---- analyze: XHR (fetch cannot report upload progress) ---------------------
export interface AnalyzeInput {
  before: File
  after: File
  disasterName?: string
  district?: string
  state?: string
  conf?: number
  alignment?: 'check' | 'register'
}
export type AnalyzePhase = 'uploading' | 'processing'

export function analyze(
  input: AnalyzeInput,
  onProgress: (phase: AnalyzePhase, uploadFraction: number) => void,
  signal?: AbortSignal,
): Promise<Analysis> {
  return new Promise((resolve, reject) => {
    const form = new FormData()
    form.append('before', input.before)
    form.append('after', input.after)
    const optional: Record<string, string | undefined> = {
      disaster_name: input.disasterName?.trim(),
      district: input.district?.trim(),
      state: input.state?.trim(),
      conf: input.conf !== undefined ? String(input.conf) : undefined,
      alignment: input.alignment,
    }
    for (const [k, v] of Object.entries(optional)) if (v) form.append(k, v)

    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${API_BASE}/api/analyze`)
    xhr.responseType = 'json'
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress('uploading', e.loaded / e.total)
    xhr.upload.onload = () => onProgress('processing', 1)
    xhr.onload = () => {
      const body = xhr.response
      if (xhr.status === 201 && body) return resolve(body as Analysis)
      const e = body?.error
      reject(e?.code ? new ApiError(e.code, e.message, xhr.status, e.details)
                     : new ApiError('http_error', `Backend returned HTTP ${xhr.status}.`, xhr.status))
    }
    xhr.onerror = () => reject(networkError())
    xhr.ontimeout = () => reject(new ApiError('timeout', 'The analysis request timed out.'))
    xhr.onabort = () => reject(new DOMException('Aborted', 'AbortError'))
    signal?.addEventListener('abort', () => xhr.abort())
    xhr.timeout = 10 * 60 * 1000
    onProgress('uploading', 0)
    xhr.send(form)
  })
}
