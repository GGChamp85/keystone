import { useState, type ReactNode } from 'react'

/**
 * Minimal, dependency-free rendering for model output: fenced code blocks (with a copy button
 * and language label) and inline code spans. No remark/rehype/highlighter — this project keeps
 * its web dependency footprint deliberately small (see package.json), and every added package is
 * one more thing the air-gap bundle has to carry. Anything outside a code span renders as plain
 * text with line breaks preserved by the caller's `white-space: pre-wrap`.
 */

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <button
      type="button"
      className="code-copy-btn"
      onClick={() => {
        void navigator.clipboard.writeText(text).then(() => {
          setCopied(true)
          setTimeout(() => setCopied(false), 1500)
        })
      }}
    >
      {copied ? 'Copied' : 'Copy'}
    </button>
  )
}

function renderInline(text: string, keyPrefix: string): ReactNode[] {
  const parts = text.split(/(`[^`]+`)/g)
  return parts.map((part, i) =>
    part.startsWith('`') && part.endsWith('`') && part.length >= 2 ? (
      <code key={`${keyPrefix}-${i}`} className="inline-code">
        {part.slice(1, -1)}
      </code>
    ) : (
      part
    ),
  )
}

export function RenderedText({ text }: { text: string }) {
  const segments = text.split(/```(\w*)\n([\s\S]*?)```/g)
  // split() with a 2-group regex yields [plain, lang, code, plain, lang, code, ..., plain]
  const nodes: ReactNode[] = []
  for (let i = 0; i < segments.length; i += 3) {
    const plain = segments[i]
    if (plain) nodes.push(<span key={`t-${i}`}>{renderInline(plain, `t-${i}`)}</span>)
    const lang = segments[i + 1]
    const code = segments[i + 2]
    if (code !== undefined) {
      nodes.push(
        <div key={`c-${i}`} className="code-block">
          <div className="code-block-head">
            <span>{lang || 'text'}</span>
            <CopyButton text={code} />
          </div>
          <pre>
            <code>{code.replace(/\n$/, '')}</code>
          </pre>
        </div>,
      )
    }
  }
  return <>{nodes}</>
}
