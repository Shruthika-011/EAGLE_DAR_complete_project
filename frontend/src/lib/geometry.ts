// Zoom/pan maths for the image viewers. The content box has size (w, h) at scale 1 and is
// drawn with transform: translate(x, y) scale(s), origin top-left; the viewport has the same size.
export interface View { scale: number; x: number; y: number }

export const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

/** Keep the zoomed content covering the viewport. */
export function clampView(v: View, w: number, h: number): View {
  return { scale: v.scale, x: clamp(v.x, w - w * v.scale, 0), y: clamp(v.y, h - h * v.scale, 0) }
}

/** Zoom by `factor` keeping the content point under (cx, cy) fixed. */
export function zoomAt(v: View, factor: number, cx: number, cy: number, w: number, h: number, min = 1, max = 10): View {
  const scale = clamp(v.scale * factor, min, max)
  const k = scale / v.scale
  return clampView({ scale, x: cx - (cx - v.x) * k, y: cy - (cy - v.y) * k }, w, h)
}

/** Image pixel -> position inside the unscaled content box. */
export const imageToBox = (px: number, py: number, imgW: number, imgH: number, boxW: number, boxH: number) =>
  ({ x: (px / imgW) * boxW, y: (py / imgH) * boxH })

/** CSS to show only a bbox region of an image as a thumbnail of the given width. */
export function cropStyle(bbox: [number, number, number, number], imgW: number, imgH: number, thumbW: number, pad = 12) {
  const x1 = Math.max(0, bbox[0] - pad), y1 = Math.max(0, bbox[1] - pad)
  const x2 = Math.min(imgW, bbox[2] + pad), y2 = Math.min(imgH, bbox[3] + pad)
  const cw = Math.max(1, x2 - x1), ch = Math.max(1, y2 - y1)
  const s = thumbW / cw
  return {
    width: thumbW, height: Math.round(ch * s), backgroundSize: `${imgW * s}px ${imgH * s}px`,
    backgroundPosition: `${-x1 * s}px ${-y1 * s}px`, backgroundRepeat: 'no-repeat' as const,
  }
}
