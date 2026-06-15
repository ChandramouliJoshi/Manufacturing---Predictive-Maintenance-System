import { useState } from 'react'

const defaultPayload = {
  air_temperature_k: 305.0,
  process_temperature_k: 310.0,
  rotational_speed_rpm: 1500.0,
  torque_nm: 45.0,
  tool_wear_min: 100.0,
}

function App() {
  const [payload, setPayload] = useState(JSON.stringify(defaultPayload, null, 2))
  const [response, setResponse] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (event) => {
    event.preventDefault()
    setLoading(true)
    setError(null)
    setResponse(null)

    try {
      const parsed = JSON.parse(payload)
      const res = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ features: parsed }),
      })
      const data = await res.json()

      if (!res.ok) {
        throw new Error(data.detail || 'Prediction request failed')
      }

      setResponse(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">FactoryGuard AI</p>
          <h1>IoT Predictive Maintenance Engine</h1>
          <p>
            Predict catastrophic robotic-arm failures up to 24 hours before they happen with
            time-series classification and explainability.
          </p>
        </div>
      </header>

      <main>
        <section className="card">
          <h2>Predict Failure Probability</h2>
          <p>Send engineered sensor features to the API and receive a failure probability response.</p>
          <form onSubmit={handleSubmit}>
            <label htmlFor="feature-json">Feature payload (JSON)</label>
            <textarea
              id="feature-json"
              value={payload}
              onChange={(event) => setPayload(event.target.value)}
            />
            <div className="button-row">
              <button type="submit" disabled={loading}>
                {loading ? 'Predicting…' : 'Run Prediction'}
              </button>
            </div>
          </form>

          {error && <p className="alert error">{error}</p>}
          {response && (
            <div className="result-box">
              <h3>Prediction Result</h3>
              <pre>{JSON.stringify(response, null, 2)}</pre>
            </div>
          )}
        </section>

        <section className="card">
          <h2>Application Notes</h2>
          <ul>
            <li>Frontend built with React and Vite.</li>
            <li>API host should be running at <strong>http://localhost:8000</strong>.</li>
            <li>Payload must contain engineered feature values for the model.</li>
          </ul>
        </section>
      </main>
    </div>
  )
}

export default App
