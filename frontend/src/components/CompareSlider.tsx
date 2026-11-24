import { useRef, useState, type KeyboardEvent, type PointerEvent } from 'react'
import { useZoomPan } from './useZoomPan'
import { ZoomControls } from './ZoomControls'

interface Props { beforeSrc: string; afterSrc: string; width: number; height: number }

/** Before/after comparison with a draggable divider. Both layers share one zoom/pan transform. */
export function CompareSlider({ beforeSrc, afterSrc, width, height }: Props) {
  const zp = useZoomPan('ctrl', 8)
  const [pos, setPos] = useState(0.5)
  const [loaded, setLoaded] = useState({ before: false, after: false })
  const [failed, setFailed] = useState(false)
  const dragging = useRef(false)

  const move = (clientX: number) => {
    const r = zp.ref.current?.getBoundingClientRect()
    if (r) setPos(Math.min(1, Math.max(0, (clientX - r.left) / r.width)))
  }
  const down = (e: PointerEvent) => { dragging.current = true; (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId); e.stopPropagation() }
  const key = (e: KeyboardEvent) => {
    if (e.key === 'ArrowLeft') setPos((p) => Math.max(0, p - 0.05))
    if (e.key === 'ArrowRight') setPos((p) => Math.min(1, p + 0.05))
  }
  const ready = loaded.before && loaded.after

  return (
    <div className="compare" style={{ aspectRatio: `${width} / ${height}` }} ref={zp.ref} {...zp.bind}>
      {!ready && !failed && <div className="skeleton" aria-busy="true">Loading images from the backend…</div>}
      {failed && <div className="img-error">One of the images could not be loaded from the backend.</div>}
      <div style={zp.contentStyle}>
        <img src={beforeSrc} alt="Before disaster" draggable={false} onLoad={() => setLoaded((s) => ({ ...s, before: true }))} onError={() => setFailed(true)} />
      </div>
      <div className="compare-after" style={{ clipPath: `inset(0 0 0 ${pos * 100}%)` }}>
        <div style={zp.contentStyle}>
          <img src={afterSrc} alt="After disaster" draggable={false} onLoad={() => setLoaded((s) => ({ ...s, after: true }))} onError={() => setFailed(true)} />
        </div>
      </div>
      {ready && (
        <>
          <span className="tag tag-before">Before Disaster</span>
          <span className="tag tag-after">After Disaster</span>
          <div className="divider" style={{ left: `${pos * 100}%` }} data-nodrag>
            <div
              className="handle" role="slider" tabIndex={0} aria-label="Comparison divider" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(pos * 100)}
              onPointerDown={down} onPointerMove={(e) => dragging.current && move(e.clientX)} onPointerUp={() => (dragging.current = false)} onKeyDown={key}
            >⇆</div>
          </div>
          <ZoomControls zoomIn={zp.zoomIn} zoomOut={zp.zoomOut} reset={zp.reset} scale={zp.view.scale} />
        </>
      )}
    </div>
  )
}
