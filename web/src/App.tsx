import { useState } from 'react'
import { ApiKeyBar } from './components/ApiKeyBar'
import { MemoryPanel } from './components/MemoryPanel'
import { TaskSubmitForm } from './components/TaskSubmitForm'
import { TaskStreamView } from './components/TaskStreamView'

function taskIdFromUrl(): string | null {
  return new URLSearchParams(window.location.search).get('task')
}

function viewFromUrl(): 'memory' | null {
  return new URLSearchParams(window.location.search).get('view') === 'memory' ? 'memory' : null
}

export default function App() {
  const [taskId, setTaskIdState] = useState<string | null>(taskIdFromUrl)
  const [view, setViewState] = useState<'memory' | null>(viewFromUrl)

  function setTaskId(id: string | null) {
    const url = new URL(window.location.href)
    if (id) url.searchParams.set('task', id)
    else url.searchParams.delete('task')
    url.searchParams.delete('view')
    window.history.pushState({}, '', url)
    setTaskIdState(id)
    setViewState(null)
  }

  function setView(next: 'memory' | null) {
    const url = new URL(window.location.href)
    if (next) url.searchParams.set('view', next)
    else url.searchParams.delete('view')
    url.searchParams.delete('task')
    window.history.pushState({}, '', url)
    setViewState(next)
    setTaskIdState(null)
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>Keystone Agents</h1>
        <div className="api-key-bar">
          {view !== 'memory' && !taskId && (
            <button className="nav-link" onClick={() => setView('memory')}>
              Memory
            </button>
          )}
          <ApiKeyBar />
        </div>
      </header>

      <main>
        {view === 'memory' ? (
          <MemoryPanel onBack={() => setView(null)} />
        ) : taskId ? (
          <TaskStreamView taskId={taskId} onBack={() => setTaskId(null)} />
        ) : (
          <TaskSubmitForm onSubmitted={setTaskId} />
        )}
      </main>
    </div>
  )
}
