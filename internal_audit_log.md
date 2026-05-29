# Internal Audit Log

Automated daily code audit results for [Docu-AI-Search](https://github.com/BhurkeSiddhesh/Docu-AI-Search).

---

## Audit: 2026-05-29
- Issues filed: 3
- Categories: 2 Critical Bug, 0 Logic Enhancement, 1 Developer Experience
- Status: Issues Filed
- New issues:
  - [#235](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/235) [Critical Bug] GET /api/config returns external_api_key in plaintext while all other provider keys are masked
  - [#236](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/236) [Developer Experience] CI security-scan job uses `|| true` — bandit/pip-audit failures never fail the build
  - [#237](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/237) [Critical Bug] background.py reads singular `folder` key with no fallback — crashes with NoOptionError after UI saves `folders` key, silently disabling auto-indexing
