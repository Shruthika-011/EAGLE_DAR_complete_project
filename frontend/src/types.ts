// Mirrors backend/schemas/*.py (Phase 3). Fields the backend may legitimately leave empty are `| null`.

export type DamageName = 'no-damage' | 'major-damage' | 'destroyed'
export type Quadrant = 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right'
export type Box = [number, number, number, number]

export interface Building {
  building_id: number
  predicted_damage: DamageName
  class_id: number
  building_confidence: number
  detection_confidence: number | null
  damage_confidence: number | null
  damage_margin: number | null
  confidence_level: 'high' | 'medium' | 'low' | null
  class_probabilities: Record<string, number> | null
  centre_xy: [number, number]
  bbox_xyxy: Box
  quadrant: Quadrant
  area_px: number | null
  geometry_source: string | null
  local_alignment: { status: string; dx?: number; dy?: number; response?: number } | null
  polygon_xy?: [number, number][] | null
}

export interface Alignment {
  status: string
  dx: number
  dy: number
  shift_px: number
  phase_corr_response: number
  orb_inliers: number
  method: string
  registered: boolean
  note: string
}

export type ClassCounts = Record<DamageName, number>

export interface Statistics {
  scene: { scene_id: string | null; width: number; height: number; total_buildings: number }
  damage_counts: ClassCounts
  damaged_total: number
  damaged_fraction: number | null
  class_fractions: ClassCounts | null
  quadrants: Record<Quadrant, ClassCounts & { damaged: number; region: Box }>
  halves: Record<string, { damaged: number; region: Box }>
  most_damaged_quadrant: {
    answer: string
    region: Box | null
    supporting_building_ids: number[]
    tie_unresolved: boolean
    quadrant_counts: Record<Quadrant, number> | null
  }
  half_comparisons: Record<string, { answer: string; damaged_counts: Record<string, number> | null }>
  confidence: {
    mean: number | null
    min: number | null
    max: number | null
    low_confidence_threshold: number
    low_confidence_building_ids: number[]
    levels: { high: number; medium: number; low: number }
  }
}

export interface Analysis {
  analysis_id: string
  created_utc: string
  status: 'complete'
  image: {
    width: number
    height: number
    before: { display_name: string; bytes: number; sha256: string }
    after: { display_name: string; bytes: number; sha256: string }
  }
  context: { disaster_name: string | null; district: string | null; state: string | null }
  parameters: { conf: number; imgsz: number; alignment: 'check' | 'register'; device: string; inference_seconds: number }
  alignment: Alignment
  statistics: Statistics
  buildings: Building[]
  warnings: string[]
  links: Record<string, string>
  disclaimer: string
}

export interface Health {
  status: 'ok' | 'degraded'
  api_version: string
  models: {
    yolo: { file: string; exists: boolean; loaded: boolean }
    siamese: { file: string; exists: boolean; loaded: boolean }
    load_error: string | null
  }
  device: {
    in_use: string | null
    requested: string
    cuda_available: boolean
    gpu: { name: string; memory_free_mb: number; memory_total_mb: number } | null
  }
  storage: { writable: boolean }
  reports?: { pdf: boolean; llm_enabled: boolean; llm_key_present: boolean; explanation: string }
  limits: { max_upload_mb: number; max_pixels: number; max_pending_requests: number }
}

export interface OfficialRecord {
  id: string
  answer: number | string
  region: number[] | string | null
}

export interface EvidenceInfo {
  region: Box | null
  region_name: string | null
  supporting_building_ids: number[]
  supporting_buildings: { building_id: number; damage: string; confidence: number | null }[]
  map_url: string | null
  highlight_truncated: boolean
}

export interface QuestionResult {
  official: OfficialRecord
  internal: {
    question_type: 1 | 2 | 3 | 4
    question_text: string | null
    parsed: Record<string, unknown>
    details: Record<string, unknown>
    evidence: EvidenceInfo
  }
}

export interface ReportJson {
  analysis_id: string
  generated_utc: string
  generation: string
  sections: { title: string; paragraphs: string[] }[]
  facts: Record<string, unknown>
}

export interface ReportMeta {
  report_id: string
  analysis_id: string
  status: 'queued' | 'generating' | 'complete' | 'failed'
  stage: string
  progress: number
  created_utc: string
  started_utc: string | null
  finished_utc: string | null
  generation_seconds: number | null
  bytes: number | null
  sha256: string | null
  pages: number | null
  explanation_source: string | null
  fallback_reason: string | null
  error: string | null
  download_url: string | null
}

export interface QuestionRef {
  id: string
  question_text: string | null
  official: OfficialRecord
  supporting_building_ids: number[]
  evidence_region: Box | null
  answered_utc: string | null
}
