import { Maximize2, ZoomIn, ZoomOut } from 'lucide-react'

export function ZoomControls({ zoomIn, zoomOut, reset, scale }: { zoomIn: () => void; zoomOut: () => void; reset: () => void; scale: number }) {
  return (
    <div className="zoom-controls" data-nodrag>
      <button type="button" onClick={zoomIn} aria-label="Zoom in" title="Zoom in"><ZoomIn size={16} /></button>
      <button type="button" onClick={zoomOut} aria-label="Zoom out" title="Zoom out"><ZoomOut size={16} /></button>
      <button type="button" onClick={reset} aria-label="Reset view" title={`Reset view (${scale.toFixed(1)}×)`}><Maximize2 size={15} /></button>
    </div>
  )
}
