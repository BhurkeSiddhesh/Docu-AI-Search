# Execution Checklist - Git Branch Consolidation

- [/] Merge target feature/bugfix branches into `main`
    - [x] Merge cumulative `fix/issue-81` branch
    - [x] Merge compatibility `fix/slowapi-middleware-compat` branch
    - [/] Merge pagination `fix/api-files-pagination` branch
    - [ ] Merge config preservation `claude/sleepy-shaw-33a61c` branch
    - [ ] Merge test fix `claude/interesting-mcclintock-4e358d` branch
    - [ ] Merge a11y and exceptions `claude/xenodochial-noether-9e832f` branch
- [ ] Run full test suites and code validation
    - [ ] Run quick backend test suite (`npm run test`)
    - [ ] Run full frontend Vitest test suite (`cd frontend && npm run test`)
    - [ ] Run structure validation (`npm run validate`)
- [ ] Resolve any test failures or bugs discovered during merge
- [ ] Repository cleanup & stale branch deletion
    - [ ] Delete merged local branches
    - [ ] Delete stale remote branches
    - [ ] Discard sandboxing `claude/determined-hoover-341185` branch
- [ ] Update `AGENTS.md` Change Log
