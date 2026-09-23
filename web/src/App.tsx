import { useState } from 'react'
import { ApiKeyBar } from './components/ApiKeyBar'
import { FineTunePanel } from './components/FineTunePanel'
import { BenchmarksPanel } from './components/BenchmarksPanel'
import { MemoryPanel } from './components/MemoryPanel'
import { ModelLibrary } from './components/ModelLibrary'
import { Playground } from './components/Playground'
import { SpendPanel } from './components/SpendPanel'
import { TaskSubmitForm } from './components/TaskSubmitForm'
import { TaskStreamView } from './components/TaskStreamView'
import { TeamTaskList } from './components/TeamTaskList'

type View = 'memory' | 'team' | 'finetune' | 'spend' | 'models' | 'playground' | 'benchmarks'
const VIEWS: View[] = ['memory', 'team', 'finetune', 'spend', 'models', 'playground', 'benchmarks']

function taskIdFromUrl(): string | null {
  return new URLSearchParams(window.location.search).get('task')
}

function viewFromUrl(): View | null {
  const v = new URLSearchParams(window.location.search).get('view')
  return VIEWS.includes(v as View) ? (v as View) : null
}

export default function App() {
  const [taskId, setTaskIdState] = useState<string | null>(taskIdFromUrl)
  const [view, setViewState] = useState<View | null>(viewFromUrl)

  function setTaskId(id: string | null) {
    const url = new URL(window.location.href)
    if (id) url.searchParams.set('task', id)
    else url.searchParams.delete('task')
    url.searchParams.delete('view')
    window.history.pushState({}, '', url)
    setTaskIdState(id)
    setViewState(null)
  }

  function setView(next: View | null, params: Record<string, string> = {}) {
    const url = new URL(window.location.href)
    if (next) url.searchParams.set('view', next)
    else url.searchParams.delete('view')
    url.searchParams.delete('task')
    url.searchParams.delete('model')
    url.searchParams.delete('base_model')
    for (const [k, v] of Object.entries(params)) url.searchParams.set(k, v)
    window.history.pushState({}, '', url)
    setViewState(next)
    setTaskIdState(null)
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>Keystone Agents</h1>
        <div className="api-key-bar">
          {!taskId && (
            <>
              {view !== 'team' && (
                <button className="nav-link" onClick={() => setView('team')}>
                  Tasks
                </button>
              )}
              {view !== 'memory' && (
                <button className="nav-link" onClick={() => setView('memory')}>
                  Memory
                </button>
              )}
              {view !== 'finetune' && (
                <button className="nav-link" onClick={() => setView('finetune')}>
                  Fine-tune
                </button>
              )}
              {view !== 'spend' && (
                <button className="nav-link" onClick={() => setView('spend')}>
                  Spend
                </button>
              )}
              {view !== 'models' && (
                <button className="nav-link" onClick={() => setView('models')}>
                  Models
                </button>
              )}
              {view !== 'playground' && (
                <button className="nav-link" onClick={() => setView('playground')}>
                  Playground
                </button>
              )}
              {view !== 'benchmarks' && (
                <button className="nav-link" onClick={() => setView('benchmarks')}>
                  Benchmarks
                </button>
              )}
            </>
          )}
          <ApiKeyBar />
        </div>
      </header>

      <main>
        {view === 'memory' ? (
          <MemoryPanel onBack={() => setView(null)} />
        ) : view === 'team' ? (
          <TeamTaskList onOpenTask={setTaskId} />
        ) : view === 'finetune' ? (
          <FineTunePanel onBack={() => setView(null)} />
        ) : view === 'spend' ? (
          <SpendPanel onBack={() => setView(null)} />
        ) : view === 'models' ? (
          <ModelLibrary
            onBack={() => setView(null)}
            onPlayground={(modelId) => setView('playground', { model: modelId })}
            onFineTune={(baseModel) => setView('finetune', { base_model: baseModel })}
          />
        ) : view === 'benchmarks' ? (
          <BenchmarksPanel onBack={() => setView(null)} />
        ) : view === 'playground' ? (
          <Playground
            initialModel={new URLSearchParams(window.location.search).get('model')}
            onBack={() => setView(null)}
            onLibrary={() => setView('models')}
          />
        ) : taskId ? (
          <TaskStreamView taskId={taskId} onBack={() => setTaskId(null)} />
        ) : (
          <TaskSubmitForm onSubmitted={setTaskId} />
        )}
      </main>
    </div>
  )
}
