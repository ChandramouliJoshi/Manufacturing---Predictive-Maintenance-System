import { useEffect, useMemo, useRef, useState } from 'react'

const sensorConfig = {
  air_temperature_k: {
    label: 'Air Temperature',
    unit: 'K',
    min: 295,
    max: 315,
    base: 305,
    noise: 0.18,
    accent: '#38bdf8',
  },
  process_temperature_k: {
    label: 'Process Temperature',
    unit: 'K',
    min: 300,
    max: 325,
    base: 310,
    noise: 0.16,
    accent: '#f59e0b',
  },
  rotational_speed_rpm: {
    label: 'Rotational Speed',
    unit: 'RPM',
    min: 1150,
    max: 1850,
    base: 1500,
    noise: 14,
    accent: '#22c55e',
  },
  torque_nm: {
    label: 'Torque',
    unit: 'Nm',
    min: 20,
    max: 75,
    base: 45,
    noise: 1.2,
    accent: '#a78bfa',
  },
  tool_wear_min: {
    label: 'Tool Wear',
    unit: 'min',
    min: 0,
    max: 260,
    base: 100,
    noise: 0.9,
    accent: '#fb7185',
  },
}

const sensorKeys = Object.keys(sensorConfig)
const historyLength = 42

const clamp = (value, min, max) => Math.min(max, Math.max(min, value))

const createInitialHistory = () =>
  Object.fromEntries(
    sensorKeys.map((key) => {
      const config = sensorConfig[key]
      return [
        key,
        Array.from({ length: historyLength }, (_, index) => {
          const wave = Math.sin(index / 5) * config.noise * 2
          return clamp(config.base + wave, config.min, config.max)
        }),
      ]
    }),
  )

const formatValue = (key, value) => {
  if (key === 'rotational_speed_rpm' || key === 'tool_wear_min') {
    return Math.round(value).toLocaleString()
  }
  return value.toFixed(1)
}

function Sparkline({ values, color, min, max }) {
  const width = 320
  const height = 96
  const points = values
    .map((value, index) => {
      const x = (index / (values.length - 1)) * width
      const y = height - ((value - min) / (max - min)) * height
      return `${x.toFixed(1)},${clamp(y, 4, height - 4).toFixed(1)}`
    })
    .join(' ')

  return (
    <svg className="sparkline" viewBox={`0 0 ${width} ${height}`} role="img">
      <defs>
        <linearGradient id={`fill-${color.replace('#', '')}`} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.32" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polyline className="grid-line" points={`0,24 ${width},24`} />
      <polyline className="grid-line" points={`0,48 ${width},48`} />
      <polyline className="grid-line" points={`0,72 ${width},72`} />
      <polygon points={`0,${height} ${points} ${width},${height}`} fill={`url(#fill-${color.replace('#', '')})`} />
      <polyline points={points} fill="none" stroke={color} strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

function App() {
  const [history, setHistory] = useState(createInitialHistory)
  const [response, setResponse] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [isStreaming, setIsStreaming] = useState(true)
  const [lastUpdated, setLastUpdated] = useState(null)
  const latestFeaturesRef = useRef(null)

  const features = useMemo(
    () =>
      Object.fromEntries(
        sensorKeys.map((key) => {
          const values = history[key]
          return [key, values[values.length - 1]]
        }),
      ),
    [history],
  )

  const riskPercent = response ? Math.round(response.failure_probability * 100) : 0
  const riskLabel = riskPercent >= 60 ? 'Critical' : riskPercent >= 25 ? 'Elevated' : 'Stable'

  useEffect(() => {
    latestFeaturesRef.current = features
  }, [features])

  const predict = async (currentFeatures = features) => {
    setLoading(true)
    setError(null)

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ features: currentFeatures }),
      })
      const data = await res.json()

      if (!res.ok) {
        throw new Error(data.detail || 'Prediction request failed')
      }

      setResponse(data)
      setLastUpdated(new Date())
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (!isStreaming) {
      return undefined
    }

    const streamTimer = window.setInterval(() => {
      setHistory((current) => {
        const nextHistory = {}
        sensorKeys.forEach((key) => {
          const config = sensorConfig[key]
          const values = current[key]
          const latest = values[values.length - 1]
          const drift = key === 'tool_wear_min' ? 0.45 : 0
          const wave = Math.sin(Date.now() / 1800 + values.length) * config.noise
          const nextValue = clamp(latest + wave + drift + (Math.random() - 0.5) * config.noise, config.min, config.max)
          nextHistory[key] = [...values.slice(1), nextValue]
        })
        return nextHistory
      })
    }, 1200)

    return () => window.clearInterval(streamTimer)
  }, [isStreaming])

  useEffect(() => {
    predict(latestFeaturesRef.current || features)

    const predictionTimer = window.setInterval(() => {
      predict(latestFeaturesRef.current || features)
    }, 4500)

    return () => window.clearInterval(predictionTimer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const handleManualChange = (key, value) => {
    const config = sensorConfig[key]
    const numericValue = clamp(Number(value), config.min, config.max)
    setHistory((current) => ({
      ...current,
      [key]: [...current[key].slice(1), numericValue],
    }))
  }

  const resetStream = () => {
    setHistory(createInitialHistory())
    setResponse(null)
    setError(null)
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">FactoryGuard AI</p>
          <h1>Live Machine Health Monitor</h1>
        </div>
        <div className={`status-pill ${isStreaming ? 'online' : 'paused'}`}>
          <span />
          {isStreaming ? 'Streaming' : 'Paused'}
        </div>
      </header>

      <main>
        <section className="overview-grid">
          <div className="summary-panel">
            <span className="panel-label">Failure Risk</span>
            <div className="risk-readout">
              <strong>{riskPercent}%</strong>
              <small>{riskLabel}</small>
            </div>
            <div className="risk-meter">
              <span style={{ width: `${riskPercent}%` }} />
            </div>
          </div>

          <div className="summary-panel">
            <span className="panel-label">Prediction</span>
            <strong className={response?.prediction ? 'danger-text' : 'ok-text'}>
              {response?.prediction ? 'Maintenance Needed' : 'Normal Operation'}
            </strong>
            <small>{lastUpdated ? `Updated ${lastUpdated.toLocaleTimeString()}` : 'Waiting for API'}</small>
          </div>

          <div className="summary-panel actions-panel">
            <button type="button" onClick={() => setIsStreaming((current) => !current)}>
              {isStreaming ? 'Pause Stream' : 'Resume Stream'}
            </button>
            <button type="button" className="secondary-button" onClick={() => predict(features)} disabled={loading}>
              {loading ? 'Checking' : 'Run Now'}
            </button>
            <button type="button" className="ghost-button" onClick={resetStream}>
              Reset
            </button>
          </div>
        </section>

        {error && <p className="alert error">{error}</p>}

        <section className="sensor-grid">
          {sensorKeys.map((key) => {
            const config = sensorConfig[key]
            const values = history[key]
            const latest = values[values.length - 1]
            const previous = values[values.length - 2]
            const delta = latest - previous
            const rangePercent = ((latest - config.min) / (config.max - config.min)) * 100

            return (
              <article className="sensor-card" key={key}>
                <div className="sensor-heading">
                  <div>
                    <span className="sensor-name">{config.label}</span>
                    <span className="sensor-key">{key}</span>
                  </div>
                  <span className="sensor-dot" style={{ background: config.accent }} />
                </div>

                <div className="sensor-value">
                  <strong>{formatValue(key, latest)}</strong>
                  <span>{config.unit}</span>
                </div>

                <Sparkline values={values} color={config.accent} min={config.min} max={config.max} />

                <div className="range-track">
                  <span style={{ width: `${rangePercent}%`, background: config.accent }} />
                </div>

                <div className="sensor-controls">
                  <small>
                    {delta >= 0 ? '+' : ''}
                    {formatValue(key, delta)} {config.unit}
                  </small>
                  <input
                    aria-label={config.label}
                    type="range"
                    min={config.min}
                    max={config.max}
                    step={key.includes('temperature') ? 0.1 : 1}
                    value={latest}
                    onChange={(event) => handleManualChange(key, event.target.value)}
                  />
                </div>
              </article>
            )
          })}
        </section>

        <section className="payload-panel">
          <div>
            <span className="panel-label">Current API Payload</span>
            <pre>{JSON.stringify({ features }, null, 2)}</pre>
          </div>
          <div>
            <span className="panel-label">Model Response</span>
            <pre>{response ? JSON.stringify(response, null, 2) : 'Waiting for prediction...'}</pre>
          </div>
        </section>
      </main>
    </div>
  )
}

export default App
