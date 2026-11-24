import { useState } from 'react'
import { Loader2, Send } from 'lucide-react'
import { ApiError, askQuestions, url } from '../api/client'
import { BuildingDetails } from '../components/BuildingDetails'
import { Card, ErrorBanner, PageHeader, RequireAnalysis } from '../components/common'
import { DamageLegend, useVisibleClasses } from '../components/DamageLegend'
import { DamageMapViewer } from '../components/DamageMapViewer'
import { fmtInt } from '../lib/derive'
import { useAnalysis } from '../state/AnalysisContext'
import { usePrefs } from '../state/PrefsContext'
import type { Analysis, Box, QuestionResult } from '../types'

const EXAMPLES = [
  'Count the destroyed buildings in the image.',
  'Count the damaged buildings (major or destroyed) in the image.',
  'Which quadrant has the most damaged buildings?',
  'Is the left half of the image more damaged than the right half?',
  'Which building in the image is a hospital?',
]

function Questions({ a, onEvidence }: { a: Analysis; onEvidence: (region: Box | null, ids: number[]) => void }) {
  const { refreshQuestions } = useAnalysis()
  const [text, setText] = useState(EXAMPLES[0])
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<ApiError | string | null>(null)
  const [res, setRes] = useState<QuestionResult | null>(null)

  const ask = async () => {
    if (busy || !text.trim()) return
    setBusy(true); setErr(null)
    try {
      const r = await askQuestions(a.analysis_id, [{ question: text.trim() }])
      setRes(r.results[0]); refreshQuestions()
      const ev = r.results[0].internal.evidence
      onEvidence(ev.region, ev.supporting_building_ids)
    } catch (e) { setErr(e instanceof ApiError ? e : String(e)) } finally { setBusy(false) }
  }

  return (
    <Card title="Ask a spatial question">
      <p className="muted small">Answered by deterministic spatial reasoning over the detected buildings (official PS-04 question types). No language model is involved.</p>
      <div className="chips">{EXAMPLES.map((e) => <button key={e} className="chip" onClick={() => setText(e)}>{e}</button>)}</div>
      <div className="ask">
        <input value={text} onChange={(e) => setText(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && ask()} maxLength={500} aria-label="Question" />
        <button className="btn btn-primary" onClick={ask} disabled={busy || !text.trim()}>{busy ? <Loader2 size={16} className="spin" /> : <Send size={16} />}Ask</button>
      </div>
      {err && <ErrorBanner error={err} />}
      {res && (
        <div className="answer">
          <div className="answer-main"><span className="muted small">Answer</span><b>{String(res.official.answer)}</b></div>
          <p className="small">{res.internal.evidence.supporting_building_ids.length > 0
            ? <>Supported by {fmtInt(res.internal.evidence.supporting_building_ids.length)} building(s); highlighted on the map with the evidence region (dashed).</>
            : res.internal.evidence.region ? 'Evidence region shown on the map (no individual buildings support this answer).' : 'No evidence region: the question cannot be answered from the images.'}</p>
          <details><summary>Official record (as submitted)</summary><pre>{JSON.stringify(res.official, null, 2)}</pre></details>
          <button className="btn btn-ghost" onClick={() => { setRes(null); onEvidence(null, []) }}>Clear evidence</button>
        </div>
      )}
    </Card>
  )
}

function Body({ a }: { a: Analysis }) {
  const { visible, toggle, all } = useVisibleClasses()
  const { prefs, setPrefs } = usePrefs()
  const [sel, setSel] = useState<number | null>(null)
  const [ev, setEv] = useState<{ region: Box | null; ids: number[] }>({ region: null, ids: [] })
  const building = a.buildings.find((b) => b.building_id === sel) ?? null

  return (
    <>
      <PageHeader title="Damage Map" subtitle={<>Predicted building footprints over the after-disaster image. Coordinates are image pixels, not geographic.</>}
        actions={<a className="btn btn-outline" href={url.map(a.analysis_id)} target="_blank" rel="noreferrer">Backend-rendered map (PNG)</a>} />
      <div className="map-layout">
        <Card className="map-card">
          <DamageMapViewer analysis={a} visible={visible} selectedId={sel} onSelect={setSel} wheel="always" highlightIds={ev.ids} evidence={ev.region} />
          <div className="map-bar">
            <DamageLegend visible={visible} toggle={toggle} counts={a.statistics.damage_counts} />
            <div className="stack-row wrap">
              <button className="btn btn-ghost" onClick={all}>Show all classes</button>
              <label className="inline"><input type="checkbox" checked={prefs.highlightLow} onChange={(e) => setPrefs({ highlightLow: e.target.checked })} />Highlight low-confidence buildings</label>
              <label className="inline"><input type="checkbox" checked={prefs.showIds} onChange={(e) => setPrefs({ showIds: e.target.checked })} />Building IDs (when zoomed)</label>
              <label className="inline">Overlay opacity<input type="range" min={0.1} max={0.8} step={0.05} value={prefs.overlayOpacity} onChange={(e) => setPrefs({ overlayOpacity: Number(e.target.value) })} /></label>
            </div>
          </div>
          <p className="muted small">Scroll to zoom · drag to pan · click a building for details.</p>
        </Card>
        <div className="side-col">
          <Card title="Building details"><BuildingDetails analysis={a} building={building} onClose={() => setSel(null)} /></Card>
          <Questions a={a} onEvidence={(region, ids) => setEv({ region, ids })} />
        </div>
      </div>
    </>
  )
}

export function DamageMapPage() {
  return <RequireAnalysis>{(a) => <Body a={a} />}</RequireAnalysis>
}
