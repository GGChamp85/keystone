import { useEffect, useState } from 'react'
import { getUsage, type UsageSummary } from '../api'

const RANGES = [7, 30, 90] as const

function usd(v: number): string {
  return `$${v.toFixed(v < 1 ? 4 : 2)}`
}

export function SpendPanel({ onBack }: { onBack: () => void }) {
  const [days, setDays] = useState<(typeof RANGES)[number]>(30)
  const [summary, setSummary] = useState<UsageSummary | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setError(null)
    getUsage(days)
      .then(setSummary)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
  }, [days])

  return (
    <div className="spend-panel">
      <button className="back-link" onClick={onBack}>
        ← Back
      </button>
      <h2>Spend</h2>
      <p className="steps-hint">
        Every token this tenant has spent — gateway requests and agent tasks — per day and model role, from the
        durable ledger, not a rolling counter.
      </p>
      <div className="spend-range">
        {RANGES.map((r) => (
          <button key={r} className={`nav-link${r === days ? ' nav-link-active' : ''}`} onClick={() => setDays(r)}>
            {r} days
          </button>
        ))}
      </div>

      {error && <p className="error-text">{error}</p>}

      {summary && (
        <>
          <div className="stats-grid">
            <div className="stat">
              <span className="stat-label">Tokens</span>
              <span className="stat-value">{summary.totals.total_tokens.toLocaleString()}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Prompt / completion</span>
              <span className="stat-value">
                {summary.totals.prompt_tokens.toLocaleString()} / {summary.totals.completion_tokens.toLocaleString()}
              </span>
            </div>
            <div className="stat">
              <span className="stat-label">Gateway requests</span>
              <span className="stat-value">{summary.totals.request_count.toLocaleString()}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Cost</span>
              <span className="stat-value">
                {summary.pricing_configured ? usd(summary.totals.estimated_cost_usd) : 'not priced'}
              </span>
            </div>
          </div>

          {summary.budget && (
            <p className="steps-hint">
              {summary.budget.monthly_budget_usd > 0
                ? `Monthly budget: ${usd(summary.budget.month_to_date_usd)} of ${usd(summary.budget.monthly_budget_usd)} spent this month` +
                  (summary.budget.month_to_date_usd >= summary.budget.monthly_budget_usd
                    ? ' — exhausted: requests and new tasks are refused until the month rolls or an admin raises it'
                    : '')
                : `No monthly budget set (${usd(summary.budget.month_to_date_usd)} spent this month). An admin can set one with keystone tenants set-limits.`}
            </p>
          )}
          {!summary.pricing_configured && (
            <p className="steps-hint">
              Set <code>MODEL_PRICES_PER_MILLION</code> (USD per 1M tokens per model role, from your own GPU
              economics) to see dollars. Tokens are recorded regardless.
            </p>
          )}

          <div className="spend-table-wrap">
            <table className="spend-table">
              <thead>
                <tr>
                  <th>Day</th>
                  <th>Role</th>
                  <th>Prompt</th>
                  <th>Completion</th>
                  <th>Total</th>
                  <th>Requests</th>
                  <th>Cost</th>
                </tr>
              </thead>
              <tbody>
                {summary.rows.length === 0 && (
                  <tr>
                    <td colSpan={7} className="steps-empty">
                      No usage in the last {days} days.
                    </td>
                  </tr>
                )}
                {summary.rows.map((r) => (
                  <tr key={`${r.day}-${r.model_role}`}>
                    <td>{r.day.slice(0, 10)}</td>
                    <td>
                      <code>{r.model_role}</code>
                    </td>
                    <td>{r.prompt_tokens.toLocaleString()}</td>
                    <td>{r.completion_tokens.toLocaleString()}</td>
                    <td>{r.total_tokens.toLocaleString()}</td>
                    <td>{r.request_count.toLocaleString()}</td>
                    <td>{summary.pricing_configured ? usd(r.estimated_cost_usd) : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}
