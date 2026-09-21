## What does this change?

<!-- One or two sentences: what changed and why. -->

## How was this tested?

<!-- Real infrastructure this was run against (Postgres/Redis/Qdrant, a real sandbox, a real cluster, etc.) — not just "unit tests pass." -->

## Checklist

- [ ] Tests added/updated, exercising real infrastructure where the existing pattern does
- [ ] `ruff check` / `ruff format --check` pass locally
- [ ] `docs/` updated if behavior, config, or a deployment step changed
- [ ] No new telemetry/phone-home introduced without updating `docs/TELEMETRY_AUDIT.md`
- [ ] No new dependency without a license check (`docs/LICENSES_AND_COMPLIANCE.md`)
- [ ] Air-gap impact considered (any new external call in the production path?)

## Related issue

<!-- Closes #... -->
