import { useMemo, useRef, useState } from 'react'
import type { DragEventHandler } from 'react'
import './App.css'
import { predictImage, submitFeedback } from './api'

type Prediction = {
  label: 'FRESH' | 'STALE' | 'UNKNOWN'
  confidence: number
  shelf_days: number
  fresh_score: number
  produce: string
  is_unknown: boolean
  unknown_reason?: string | null
  source: string
}

type ScanRecord = {
  id: string
  at: string
  fileName: string
  result: Prediction
}

const HISTORY_KEY = 'freshor_history_v1'

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`
}

function actionFromPrediction(result: Prediction): string {
  if (result.label === 'UNKNOWN') {
    return 'Retake photo'
  }

  if (result.label === 'STALE') {
    return result.shelf_days <= 0 ? 'Discard now' : 'Use urgently'
  }
  return result.shelf_days <= 3 ? 'Monitor daily' : 'No action needed'
}

function loadHistory(): ScanRecord[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY)
    return raw ? (JSON.parse(raw) as ScanRecord[]) : []
  } catch {
    return []
  }
}

function persistHistory(next: ScanRecord[]): void {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(next))
}

function App() {
  const cameraInputRef = useRef<HTMLInputElement | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const [dragActive, setDragActive] = useState(false)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [currentFile, setCurrentFile] = useState<File | null>(null)
  const [fileName, setFileName] = useState<string>('')
  const [result, setResult] = useState<Prediction | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string>('')
  const [feedbackProduce, setFeedbackProduce] = useState('')
  const [feedbackFreshness, setFeedbackFreshness] = useState<'fresh' | 'stale'>('fresh')
  const [feedbackNotes, setFeedbackNotes] = useState('')
  const [feedbackStatus, setFeedbackStatus] = useState('')
  const [isSavingFeedback, setIsSavingFeedback] = useState(false)
  const [history, setHistory] = useState<ScanRecord[]>(() => loadHistory())

  const freshnessTone = result?.label === 'FRESH' ? 'fresh' : result?.label === 'STALE' ? 'stale' : 'unknown'
  const todaysScans = useMemo(() => {
    const day = new Date().toISOString().slice(0, 10)
    return history.filter((item) => item.at.startsWith(day)).length
  }, [history])

  const handleFile = async (file?: File) => {
    if (!file) return

    setError('')
    setFeedbackStatus('')
    setResult(null)
    setCurrentFile(file)
    setFileName(file.name)

    const nextPreview = URL.createObjectURL(file)
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev)
      return nextPreview
    })

    try {
      setIsLoading(true)
      const prediction = await predictImage(file)
      setResult(prediction)
      setFeedbackFreshness(prediction.label === 'STALE' ? 'stale' : 'fresh')
      setFeedbackProduce(prediction.produce === 'unknown produce' ? '' : prediction.produce)
      setFeedbackNotes('')
      const record: ScanRecord = {
        id: crypto.randomUUID(),
        at: new Date().toISOString(),
        fileName: file.name,
        result: prediction,
      }
      const nextHistory = [record, ...history].slice(0, 30)
      setHistory(nextHistory)
      persistHistory(nextHistory)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed')
    } finally {
      setIsLoading(false)
    }
  }

  const onSubmitFeedback = async () => {
    if (!currentFile || !result) return

    if (!feedbackProduce.trim()) {
      setFeedbackStatus('Please add produce name before submitting feedback.')
      return
    }

    try {
      setIsSavingFeedback(true)
      setFeedbackStatus('')
      await submitFeedback({
        file: currentFile,
        produceName: feedbackProduce.trim(),
        freshnessLabel: feedbackFreshness,
        notes: feedbackNotes,
        predictedLabel: result.label,
        predictedConfidence: result.confidence,
        isUnknown: result.is_unknown,
      })
      setFeedbackStatus('Thanks. Saved for future training updates.')
    } catch (err) {
      setFeedbackStatus(err instanceof Error ? err.message : 'Failed to save feedback.')
    } finally {
      setIsSavingFeedback(false)
    }
  }

  const onDrop: DragEventHandler<HTMLDivElement> = async (event) => {
    event.preventDefault()
    setDragActive(false)
    const file = event.dataTransfer.files?.[0]
    await handleFile(file)
  }

  return (
    <main className="app-shell">
      <header className="hero">
        <div className="brand-row">
          <img src="/freshor-mark.svg" alt="FreshOrNot" className="brand-mark" />
          <div>
            <p className="eyebrow">Produce Intelligence</p>
            <h1>FreshOrNot</h1>
          </div>
        </div>
        <p className="hero-subtitle">
          Scan produce quality with your trained model. Works on iPhone, Android, and desktop browsers.
        </p>
      </header>

      <section className="stat-grid">
        <article>
          <p className="label">Scans today</p>
          <p className="value">{todaysScans}</p>
        </article>
        <article>
          <p className="label">Model source</p>
          <p className="value small">{result?.source ?? 'Awaiting scan'}</p>
        </article>
        <article>
          <p className="label">Last result</p>
          <p className={`value ${freshnessTone}`}>{result?.label ?? 'N/A'}</p>
        </article>
      </section>

      <section
        className={`drop-zone ${dragActive ? 'active' : ''}`}
        onDragOver={(event) => {
          event.preventDefault()
          setDragActive(true)
        }}
        onDragLeave={(event) => {
          event.preventDefault()
          setDragActive(false)
        }}
        onDrop={onDrop}
      >
        <p className="drop-title">Drop image here on web, or choose camera/gallery below</p>
        <div className="cta-row">
          <button type="button" className="btn primary" onClick={() => cameraInputRef.current?.click()}>
            Take photo
          </button>
          <button type="button" className="btn" onClick={() => fileInputRef.current?.click()}>
            Upload image
          </button>
        </div>
        <input
          ref={cameraInputRef}
          className="hidden-input"
          type="file"
          accept="image/*"
          capture="environment"
          onChange={async (event) => handleFile(event.target.files?.[0])}
        />
        <input
          ref={fileInputRef}
          className="hidden-input"
          type="file"
          accept="image/*"
          onChange={async (event) => handleFile(event.target.files?.[0])}
        />
      </section>

      {previewUrl && (
        <section className="scan-grid">
          <article className="panel">
            <p className="panel-title">Image preview</p>
            <img src={previewUrl} alt="Uploaded produce" className="preview-image" />
            <p className="file-name">{fileName}</p>
          </article>

          <article className="panel">
            <p className="panel-title">Assessment</p>
            {isLoading && <p className="loading">Analyzing produce...</p>}
            {error && <p className="error">{error}</p>}
            {result && (
              <div className="result-stack">
                <p className={`status-pill ${freshnessTone}`}>{result.label}</p>
                <p className="metric">Confidence: {formatPercent(result.confidence)}</p>
                <p className="metric">Estimated shelf life: {result.shelf_days} days</p>
                <p className="metric">Produce: {result.produce || 'unknown'}</p>
                <p className="metric">Action: {actionFromPrediction(result)}</p>
                {result.is_unknown && (
                  <p className="unknown-banner">Unknown produce detected. You can add it to training data.</p>
                )}
              </div>
            )}

            {result?.is_unknown && currentFile && (
              <div className="feedback-form">
                <p className="feedback-title">Add this item to training data</p>
                <label htmlFor="feedback-produce">Produce name</label>
                <input
                  id="feedback-produce"
                  value={feedbackProduce}
                  onChange={(event) => setFeedbackProduce(event.target.value)}
                  placeholder="e.g., eggplant"
                />
                <label htmlFor="feedback-freshness">Freshness label</label>
                <select
                  id="feedback-freshness"
                  value={feedbackFreshness}
                  onChange={(event) => setFeedbackFreshness(event.target.value as 'fresh' | 'stale')}
                >
                  <option value="fresh">Fresh</option>
                  <option value="stale">Stale</option>
                </select>
                <label htmlFor="feedback-notes">Notes (optional)</label>
                <textarea
                  id="feedback-notes"
                  value={feedbackNotes}
                  onChange={(event) => setFeedbackNotes(event.target.value)}
                  placeholder="Lighting, angle, context, etc."
                />
                <button type="button" className="btn primary" onClick={onSubmitFeedback} disabled={isSavingFeedback}>
                  {isSavingFeedback ? 'Saving...' : 'Save for training'}
                </button>
                {feedbackStatus && <p className="feedback-status">{feedbackStatus}</p>}
              </div>
            )}
          </article>
        </section>
      )}

      <section className="history panel">
        <div className="history-head">
          <p className="panel-title">Recent scans</p>
          <button
            className="btn ghost"
            type="button"
            onClick={() => {
              setHistory([])
              persistHistory([])
            }}
          >
            Clear
          </button>
        </div>
        <div className="history-list">
          {history.length === 0 && <p className="empty">No scans yet.</p>}
          {history.map((item) => (
            <div key={item.id} className="history-item">
              <div>
                <p className="history-label">{item.result.label}</p>
                <p className="history-meta">{item.result.produce || 'unknown'} · {formatPercent(item.result.confidence)}</p>
              </div>
              <div>
                <p className="history-meta">{new Date(item.at).toLocaleString()}</p>
                <p className="history-file">{item.fileName}</p>
              </div>
            </div>
          ))}
        </div>
      </section>
    </main>
  )
}

export default App
