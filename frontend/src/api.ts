export type PredictionResponse = {
  label: 'FRESH' | 'STALE' | 'UNKNOWN'
  confidence: number
  shelf_days: number
  fresh_score: number
  produce: string
  is_unknown: boolean
  unknown_reason?: string | null
  source: string
}

export type FeedbackRequest = {
  file: File
  produceName: string
  freshnessLabel: 'fresh' | 'stale'
  notes?: string
  predictedLabel?: 'FRESH' | 'STALE' | 'UNKNOWN'
  predictedConfidence?: number
  isUnknown?: boolean
}

const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api'

export async function predictImage(file: File): Promise<PredictionResponse> {
  const form = new FormData()
  form.append('file', file)

  const response = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    body: form,
  })

  if (!response.ok) {
    const contentType = response.headers.get('content-type') || ''
    if (contentType.includes('application/json')) {
      const payload = (await response.json()) as { detail?: string }
      throw new Error(payload.detail || `Prediction failed (${response.status})`)
    }

    const text = await response.text()
    throw new Error(text || `Prediction failed (${response.status})`)
  }

  return (await response.json()) as PredictionResponse
}

export async function submitFeedback(payload: FeedbackRequest): Promise<void> {
  const form = new FormData()
  form.append('file', payload.file)
  form.append('produce_name', payload.produceName)
  form.append('freshness_label', payload.freshnessLabel)
  form.append('notes', payload.notes ?? '')
  form.append('predicted_label', payload.predictedLabel ?? '')
  form.append('predicted_confidence', String(payload.predictedConfidence ?? ''))
  form.append('is_unknown', String(payload.isUnknown ?? true))

  const response = await fetch(`${API_BASE}/feedback`, {
    method: 'POST',
    body: form,
  })

  if (!response.ok) {
    const contentType = response.headers.get('content-type') || ''
    if (contentType.includes('application/json')) {
      const body = (await response.json()) as { detail?: string }
      throw new Error(body.detail || `Feedback upload failed (${response.status})`)
    }
    throw new Error(`Feedback upload failed (${response.status})`)
  }
}
