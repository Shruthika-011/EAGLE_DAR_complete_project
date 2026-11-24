import { useMemo, useState, type PointerEvent } from 'react'
import { url } from '../api/client'
import { DAMAGE_META } from '../lib/damage'
import { fmtNum } from '../lib/derive'
import { usePrefs } from '../state/PrefsContext'
import type { Analysis, Building, DamageName } from '../types'
import { useZoomPan } from './useZoomPan'
import { ZoomControls } from './ZoomControls'

interface Props {
  analysis: Analysis
  visible: Set<DamageName>
  selectedId: number | null
  onSelect: (id: number | null) => void
  wheel?: 'always' | 'ctrl'
  highlightIds?: number[]
  evidence?: [number, number, number, number] | null
}

const points = (b: Building) =>
  (b.polygon_xy && b.polygon_xy.length >= 3
    ? b.polygon_xy
    : [[b.bbox_xyxy[0], b.bbox_xyxy[1]], [b.bbox_xyxy[2], b.bbox_xyxy[1]], [b.bbox_xyxy[2], b.bbox_xyxy[3]], [b.bbox_xyxy[0], b.bbox_xyxy[3]]]
  ).map(([x, y]) => `${x},${y}`).join(' ')

/**
 * Draws the backend's building polygons over the uploaded after-image.
 * The SVG uses viewBox = image pixels, so polygon coordinates are used as returned (no conversion error);
 * the browser scales the whole layer with the image. These are IMAGE coordinates, not geographic ones.
 */
export function DamageMapViewer({ analysis, visible, selectedId, onSelect, wheel = 'ctrl', highlightIds, evidence }: Props) {
  const { prefs } = usePrefs()
  const zp = useZoomPan(wheel, 12)
  const { width: W, height: H } = analysis.image
  const lowThreshold = analysis.statistics.confidence.low_confidence_threshold
  const [hover, setHover] = useState<{ b: Building; x: number; y: number } | null>(null)
  const [state, setState] = useState<'loading' | 'ready' | 'error'>('loading')
  const hl = useMemo(() => new Set(highlightIds ?? []), [highlightIds])
  const dim = hl.size > 0
  const shown = useMemo(() => analysis.buildings.filter((b) => visible.has(b.predicted_damage)), [analysis.buildings, visible])
  const noPolys = analysis.buildings.length > 0 && analysis.buildings.every((b) => !b.polygon_xy)

  const track = (e: PointerEvent, b: Building) => {
    const r = zp.ref.current!.getBoundingClientRect()
    setHover({ b, x: e.clientX - r.left, y: e.clientY - r.top })
  }
  const fontSize = (11 * W) / (Math.max(zp.size.w, 1) * zp.view.scale)

  return (
    <div className="mapview" style={{ aspectRatio: `${W} / ${H}` }} ref={zp.ref} {...zp.bind} onPointerLeave={() => setHover(null)}>
      {state === 'loading' && <div className="skeleton" aria-busy="true">Loading image…</div>}
      {state === 'error' && <div className="img-error">The after-image could not be loaded from the backend.</div>}
      <div style={zp.contentStyle}>
        <img src={url.after(analysis.analysis_id)} alt="After-disaster image" draggable={false} onLoad={() => setState('ready')} onError={() => setState('error')} />
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label="Predicted building damage overlay (image coordinates)">
          {shown.map((b) => {
            const c = DAMAGE_META[b.predicted_damage].color
            const sel = b.building_id === selectedId
            const faded = dim && !hl.has(b.building_id)
            const low = prefs.highlightLow && b.building_confidence < lowThreshold
            return (
              <polygon
                key={b.building_id} points={points(b)} fill={c} fillOpacity={faded ? 0.08 : sel ? Math.min(0.8, prefs.overlayOpacity + 0.3) : prefs.overlayOpacity}
                stroke={sel ? '#fff' : c} strokeWidth={sel ? 3 : hl.has(b.building_id) ? 3 : 1.5} vectorEffect="non-scaling-stroke" strokeOpacity={faded ? 0.3 : 1}
                strokeDasharray={low && !sel ? '4 3' : undefined} data-low-confidence={low ? 'true' : undefined}
                style={{ cursor: 'pointer' }}
                onPointerMove={(e) => track(e, b)} onPointerLeave={() => setHover(null)}
                onClick={() => { if (!zp.wasDrag()) onSelect(sel ? null : b.building_id) }}
              />
            )
          })}
          {prefs.showIds && zp.view.scale >= 1.8 && shown.map((b) => (
            <text key={`t${b.building_id}`} x={b.centre_xy[0]} y={b.centre_xy[1]} fontSize={fontSize} textAnchor="middle" dominantBaseline="central"
                  fill="#fff" stroke="#000" strokeWidth={fontSize / 5} paintOrder="stroke" pointerEvents="none">{b.building_id}</text>
          ))}
          {evidence && <rect x={evidence[0]} y={evidence[1]} width={evidence[2] - evidence[0]} height={evidence[3] - evidence[1]} fill="none" stroke="#d946ef" strokeWidth={3} strokeDasharray="8 5" vectorEffect="non-scaling-stroke" pointerEvents="none" />}
        </svg>
      </div>
      {hover && (
        <div className="tooltip" style={{ left: Math.min(hover.x + 14, zp.size.w - 190), top: Math.min(hover.y + 14, zp.size.h - 90) }}>
          <b>Building #{hover.b.building_id}</b>
          <span><i style={{ background: DAMAGE_META[hover.b.predicted_damage].color }} />{DAMAGE_META[hover.b.predicted_damage].label}</span>
          <span>Model confidence {fmtNum(hover.b.building_confidence)}{hover.b.building_confidence < lowThreshold ? ' (low)' : ''}</span>
        </div>
      )}
      {noPolys && <div className="map-note">Backend returned no polygons for this analysis.</div>}
      <ZoomControls zoomIn={zp.zoomIn} zoomOut={zp.zoomOut} reset={zp.reset} scale={zp.view.scale} />
    </div>
  )
}
