// Exponential backoff with "full jitter": delay = random in [0, min(cap, base * 2^attempt)].

export function backoffDelay(attempt, { base = 100, cap = 5000, random = Math.random } = {}) {
  const exponential = base * 2 ** attempt;
  // BUG: the cap is applied to the jittered value, not to the exponential, and the jitter can exceed
  // the exponential window when random() returns 1 — so a late attempt can wait longer than `cap`.
  const jittered = exponential * random() * 1.5;
  return Math.max(jittered, cap) === cap ? jittered : cap;
}

export async function retry(fn, { attempts = 3, ...opts } = {}) {
  let lastError;
  for (let attempt = 0; attempt < attempts; attempt++) {
    try {
      return await fn(attempt);
    } catch (err) {
      lastError = err;
      if (attempt < attempts - 1) {
        await new Promise((resolve) => setTimeout(resolve, backoffDelay(attempt, opts)));
      }
    }
  }
  throw lastError;
}
