import { useState } from 'react'

const defaultValues = {
  air_temperature_k: 305.0,
  process_temperature_k: 310.0,
  rotational_speed_rpm: 1500.0,
  torque_nm: 45.0,
  tool_wear_min: 100.0,
}

const featureDescriptions = {
  air_temperature_k: 'Air Temperature (Kelvin)',
  process_temperature_k: 'Process Temperature (Kelvin)',
  rotational_speed_rpm: 'Rotational Speed (RPM)',
  torque_nm: 'Torque (Newton-meters)',
  tool_wear_min: 'Tool Wear (minutes)',
}

function App() {
  const [features, setFeatures] = useState(defaultValues)
  const [response, setResponse] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleChange = (key, value) => {
    setFeatures((current) => ({
      ...current,
      [key]: Number(value),
    }))
  }

  const handleSubmit = async (event) => {
    event.preventDefault()
    setLoading(true)
    setError(null)
    setResponse(null)

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ features }),
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
          <p>Enter each raw sensor reading below and click the button to get a failure prediction.</p>
          <form onSubmit={handleSubmit}>
            {Object.entries(features).map(([name, value]) => (
              <div className="input-row" key={name}>
                <label htmlFor={name}>{featureDescriptions[name] || name}</label>
                <div className="input-wrapper">
                  <input
                    id={name}
                    type="number"
                    value={value}
                    step="any"
                    onChange={(event) => handleChange(name, event.target.value)}
                  />
                  <span className="field-name">{name}</span>
                </div>
              </div>
            ))}

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
            <li>Enter raw sensor values directly in the input fields.</li>
            <li>The backend will engineer features automatically before prediction.</li>
          </ul>
        </section>
      </main>
    </div>
  )
}

export default App
