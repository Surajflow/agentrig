## What this changes

## Which agent failure it prevents

## Checklist

- [ ] No new runtime dependencies
- [ ] Nothing in `src/agentguard/` imports a provider SDK
- [ ] A test covers the new behaviour (`python -m unittest discover -s tests`)
- [ ] `ruff check src tests examples` passes
- [ ] README / docs updated where the behaviour is user-visible
