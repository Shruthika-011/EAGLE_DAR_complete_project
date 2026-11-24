import { X } from 'lucide-react'
import { url } from '../api/client'
import { DAMAGE_META } from '../lib/damage'
import { fmtNum } from '../lib/derive'
import { useAnalysis } from '../state/AnalysisContext'
import { cropStyle } from '../lib/geometry'
import type { Analysis, Building } from '../types'

function Crop({ src, b, w, h, label }: { src: string; b: Building; w: number; h: number; label: string }) {
  return (
    <figure className="crop">
      <div style={{ ...cropStyle(b.bbox_xyxy, w, h, 150), backgroundImage: `url(${src})` }} role="img" aria-label={`${label} crop of building ${b.building_id}`} />
      <figcaption>{label}</figcaption>
    </figure>
  )
}

export function BuildingDetails({ analysis, building, onClose }: { analysis: Analysis; building: Building | null; onClose?: () => void }) {
  const { questions } = useAnalysis()
  if (!building) return <p className="muted pad">Select a building (click a polygon or a table row) to see its prediction.</p>
  const b = building
  const { width: w, height: h } = analysis.image
  const probs = b.class_probabilities
  const cited = questions.filter((q) => q.supporting_building_ids.includes(b.building_id))
  return (
    <div className="details">
      <div className="details-head">
        <h3>Building #{b.building_id}</h3>
        {onClose && <button className="icon-btn" onClick={onClose} aria-label="Close details"><X size={16} /></button>}
      </div>
      <p className="badge" style={{ borderColor: DAMAGE_META[b.predicted_damage].color }}>
        <i style={{ background: DAMAGE_META[b.predicted_damage].color }} />{DAMAGE_META[b.predicted_damage].label}
      </p>
      <div className="crops">
        <Crop src={url.before(analysis.analysis_id)} b={b} w={w} h={h} label="Before" />
        <Crop src={url.after(analysis.analysis_id)} b={b} w={w} h={h} label="After" />
      </div>
      <dl className="kv">
        <dt>Model confidence</dt><dd>{fmtNum(b.building_confidence)} {b.confidence_level && <span className={`lvl lvl-${b.confidence_level}`}>{b.confidence_level}</span>}</dd>
        <dt>Detection score</dt><dd>{fmtNum(b.detection_confidence)}</dd>
        <dt>Damage-class probability</dt><dd>{fmtNum(b.damage_confidence)}</dd>
        <dt>Margin over 2nd class</dt><dd>{fmtNum(b.damage_margin)}</dd>
        <dt>Evidence</dt><dd>{cited.length ? <>Cited by {cited.map((q) => q.id).join(', ')}</> : <span className="muted">Not cited by any question</span>}</dd>
        <dt>Quadrant</dt><dd>{b.quadrant}</dd>
        <dt>Centre (px)</dt><dd>{b.centre_xy[0].toFixed(1)}, {b.centre_xy[1].toFixed(1)}</dd>
        <dt>Bounding box (px)</dt><dd>{b.bbox_xyxy.join(', ')}</dd>
        <dt>Area (px²)</dt><dd>{b.area_px === null ? '—' : b.area_px.toFixed(0)}</dd>
        <dt>Geometry</dt><dd>{b.geometry_source ?? '—'}{b.polygon_xy ? ` · ${b.polygon_xy.length} vertices` : ''}</dd>
        <dt>Local alignment</dt><dd>{b.local_alignment?.status ?? '—'}</dd>
      </dl>
      {probs && (
        <>
          <h4>Class probabilities</h4>
          <ul className="probs">
            {Object.entries(probs).map(([k, v]) => (
              <li key={k}><span>{DAMAGE_META[k as keyof typeof DAMAGE_META]?.short ?? k}</span><div><i style={{ width: `${v * 100}%`, background: DAMAGE_META[k as keyof typeof DAMAGE_META]?.color }} /></div><b>{v.toFixed(3)}</b></li>
            ))}
          </ul>
        </>
      )}
      <p className="muted small">Model confidence is an uncalibrated score (detection score × damage-class probability), not the probability that this prediction is correct. Coordinates are image pixels, not geographic.</p>
    </div>
  )
}
