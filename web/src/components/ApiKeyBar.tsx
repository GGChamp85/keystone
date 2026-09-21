import { useState } from 'react'
import { getApiKey, setApiKey } from '../api'

export function ApiKeyBar() {
  const [key, setKeyState] = useState(getApiKey())
  const [saved, setSaved] = useState(false)

  function save() {
    setApiKey(key.trim())
    setSaved(true)
    setTimeout(() => setSaved(false), 1500)
  }

  return (
    <div className="api-key-bar">
      <label htmlFor="api-key">API key</label>
      <input
        id="api-key"
        type="password"
        placeholder="ks-...."
        value={key}
        onChange={(e) => setKeyState(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && save()}
      />
      <button onClick={save}>{saved ? 'Saved' : 'Save'}</button>
    </div>
  )
}
