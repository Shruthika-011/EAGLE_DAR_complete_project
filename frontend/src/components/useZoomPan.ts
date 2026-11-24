import { useCallback, useEffect, useRef, useState, type CSSProperties, type PointerEvent } from 'react'
import { clampView, zoomAt, type View } from '../lib/geometry'

type WheelMode = 'always' | 'ctrl'

/** Zoom/pan for a fixed-aspect viewport. Wheel zoom is opt-in so cards don't hijack page scrolling. */
export function useZoomPan(wheel: WheelMode = 'ctrl', max = 10) {
  const ref = useRef<HTMLDivElement>(null)
  const [view, setView] = useState<View>({ scale: 1, x: 0, y: 0 })
  const [size, setSize] = useState({ w: 1, h: 1 })
  const sizeRef = useRef(size)
  const drag = useRef<{ x: number; y: number; vx: number; vy: number; moved: number } | null>(null)
  const dragged = useRef(false)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(() => {
      const w = el.clientWidth || 1, h = el.clientHeight || 1
      sizeRef.current = { w, h }
      setSize({ w, h })
      setView((v) => clampView(v, w, h))
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const zoomBy = useCallback((factor: number, cx?: number, cy?: number) => {
    const { w, h } = sizeRef.current
    setView((v) => zoomAt(v, factor, cx ?? w / 2, cy ?? h / 2, w, h, 1, max))
  }, [max])

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      if (wheel === 'ctrl' && !(e.ctrlKey || e.metaKey)) return
      e.preventDefault()
      const r = el.getBoundingClientRect()
      zoomBy(e.deltaY < 0 ? 1.15 : 1 / 1.15, e.clientX - r.left, e.clientY - r.top)
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [wheel, zoomBy])

  const onPointerDown = (e: PointerEvent) => {
    if ((e.target as HTMLElement).closest('[data-nodrag]')) return
    drag.current = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y, moved: 0 }
    dragged.current = false
  }
  const onPointerMove = (e: PointerEvent) => {
    const d = drag.current
    if (!d) return
    const dx = e.clientX - d.x, dy = e.clientY - d.y
    d.moved = Math.max(d.moved, Math.hypot(dx, dy))
    if (d.moved > 4) {
      dragged.current = true
      if (!(e.currentTarget as HTMLElement).hasPointerCapture(e.pointerId)) (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
      const { w, h } = sizeRef.current
      setView((v) => clampView({ scale: v.scale, x: d.vx + dx, y: d.vy + dy }, w, h))
    }
  }
  const end = (e: PointerEvent) => {
    drag.current = null
    if ((e.currentTarget as HTMLElement).hasPointerCapture?.(e.pointerId)) (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId)
  }

  const contentStyle: CSSProperties = {
    transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})`, transformOrigin: '0 0', width: '100%', height: '100%', position: 'absolute', inset: 0,
  }

  return {
    ref, view, size, contentStyle,
    zoomIn: () => zoomBy(1.4), zoomOut: () => zoomBy(1 / 1.4),
    reset: () => setView({ scale: 1, x: 0, y: 0 }),
    wasDrag: () => dragged.current,
    bind: { onPointerDown, onPointerMove, onPointerUp: end, onPointerCancel: end },
  }
}
