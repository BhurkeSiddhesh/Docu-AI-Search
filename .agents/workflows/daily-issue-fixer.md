---
description: 
---

You are a senior engineer performing a daily automated fix pass on the Docu-AI-Search repository
(https://github.com/BhurkeSiddhesh/Docu-AI-Search).

Your job has two phases:
  Phase A — Close existing open PRs that are ready to merge
  Phase B — Pick up open issues and fix them (new PRs)

Run Phase A first, always.

### A1 — Fetch all open PRs

gh pr list \
  --repo BhurkeSiddhesh/Docu-AI-Search \
  --state open \
  --json number,title,headRefName,body,labels,mergeable \
  --limit 50

Only process PRs whose branch name starts with fix/ or p3-batch/.
Skip any PR labeled needs-human-review.

### A2 — For each eligible PR, evaluate CI status

Fetch check results:

HEAD_SHA=$(gh pr view <PR_NUMBER> \
  --repo BhurkeSiddhesh/Docu-AI-Search \
  --json headRefOid -q .headRefOid)

gh pr checks <PR_NUMBER> \
  --repo BhurkeSiddhesh/Docu-AI-Search \
  --json name,state,description

Separate checks into two buckets:
  - CodeRabbit checks: any check whose name contains "coderabbit" (case-insensitive)
  - All other checks: everything else

### A3 — Evaluate non-CodeRabbit checks first

If ANY non-CodeRabbit check is failing or errored:
  → Do not merge. Leave PR open. Post comment:
    gh pr comment <PR_NUMBER> --repo BhurkeSiddhesh/Docu-AI-Search \
      --body "Auto-merge blocked: non-CodeRabbit CI checks failing.
      Requires human review before merge."
  → Log as "CI failing — left open"
  → Move to next PR

If non-CodeRabbit checks are still running:
  → Wait up to 5 minutes:
    gh pr checks <PR_NUMBER> --watch --timeout 300
  → Re-evaluate after timeout

If ALL non-CodeRabbit checks pass → proceed to A4

### A4 — Evaluate CodeRabbit check

If CodeRabbit check status is passing or neutral:
  → CodeRabbit reviewed successfully, no issues blocking merge
  → Proceed to A5

If CodeRabbit check status is failing or errored:
  → Fetch the check output to determine why:

  gh api "/repos/BhurkeSiddhesh/Docu-AI-Search/commits/${HEAD_SHA}/check-runs" \
    --jq '.check_runs[]
          | select(.name | ascii_downcase | contains("coderabbit"))
          | {conclusion: .conclusion, summary: .output.summary, title: .output.title}'

  Inspect the summary field:

  RATE LIMITED (skip CodeRabbit, proceed to merge):
    Summary contains any of: "rate limit", "quota", "limit reached",
    "limit exceeded", "too many requests", "monthly limit", "usage limit"
    → CodeRabbit has hit its limit — this is not a code quality signal
    → Log: "CodeRabbit rate-limited — bypassed for this PR"
    → Proceed to A5

  ACTUAL REVIEW ISSUES (CodeRabbit reviewed and found problems):
    Summary contains review content, code suggestions, or findings
    → Post a comment on the PR:
      gh pr comment <PR_NUMBER> --repo BhurkeSiddhesh/Docu-AI-Search \
        --body "CodeRabbit has flagged issues in this PR.
        Review the CodeRabbit comments before merging.
        Auto-merge paused — awaiting human review."
    → Add label needs-human-review
    → Log: "CodeRabbit review found issues — left for human"
    → Do not merge. Move to next PR.

  INCONCLUSIVE (check errored, timed out, or empty output):
    → Treat as neutral, proceed to A5
    → Log: "CodeRabbit inconclusive — treated as neutral"

### A5 — Merge decision

Read the PR body for the line:
  Merge Disposition: auto-merge

If auto-merge AND the PR is not a P1 Critical Bug fix:
  gh pr merge <PR_NUMBER> \
    --repo BhurkeSiddhesh/Docu-AI-Search \
    --squash \
    --delete-branch
  → Comment on the linked issue(s):
    gh issue comment <ISSUE_NUMBER> --repo BhurkeSiddhesh/Docu-AI-Search \
      --body "Fix merged via PR #<PR_NUMBER>. Closing."
  → Close the issue:
    gh issue close <ISSUE_NUMBER> --repo BhurkeSiddhesh/Docu-AI-Search
  → Log as "Auto-merged ✓"

If P1 Critical Bug OR "Merge Disposition: awaits human review":
  → Do not merge. Leave open.
  → Log as "Left for human review"

If PR body has no Merge Disposition line (manually created PR):
  → Do not merge. Leave open.
  → Log as "No disposition — skipped"


### B1 — Fetch and triage open issues

gh issue list \
  --repo BhurkeSiddhesh/Docu-AI-Search \
  --state open \
  --limit 100 \
  --json number,title,body,labels,createdAt \
  | jq 'sort_by(.labels[0].name) | reverse'

Sort into priority tiers:
  P1: label = Critical Bug
  P2: label = Logic Enhancement
  P3: label = Developer Experience

Within each tier, process issues labeled auto-merge-ok first.

Skip any issue that:
  - Already has a linked open PR:
    gh pr list --repo BhurkeSiddhesh/Docu-AI-Search --search "closes #<N>"
  - Is labeled needs-human-review or wont-fix
  - Has body containing <!-- skip-autofix -->

### B2 — Parse agent metadata from issue body

For each candidate issue, find the block:

  <!-- MACHINE READABLE — DO NOT EDIT -->
  fix-confidence: HIGH | MEDIUM | LOW
  auto-fix-eligible: yes | no
  estimated-loc-change: N
  affected-files: file1.py, file2.py
  <!-- END METADATA -->

If metadata block is absent (manually filed issue), treat as:
  fix-confidence: MEDIUM
  auto-fix-eligible: no

Determine merge disposition before touching any code:

  P1 | any confidence | any eligibility  → PR only, never auto-merge
  P2 | HIGH           | yes              → PR → auto-merge if CI passes
  P2 | MEDIUM         | yes              → PR only, awaits human review
  P2 | LOW            | no               → Escalate, no PR
  P3 | HIGH or MEDIUM | yes              → Batch PR → auto-merge if CI passes
  P3 | LOW            | no               → Escalate, no PR

### B3 — Validate each issue before writing code

For each candidate issue:

1. Read the referenced file and line number from the issue body.
   Confirm the bug still exists in current main.

2. If already fixed in main:
   gh issue close <N> \
     --repo BhurkeSiddhesh/Docu-AI-Search \
     --comment "Verified fixed in main as of $(date +%Y-%m-%d). Closing."
   Log as "Auto-closed — already fixed"
   Move to next issue.

3. If fix-confidence is LOW: skip to B8 (escalate). Do not write code.

### B4 — Group issues into work queues

Individual queue (one branch, one PR per issue):
  - All P1 issues
  - All P2 issues

Batch queue (one branch, one PR for all):
  - All P3 issues with auto-fix-eligible: yes

Process max 5 issues from the individual queue per run.
If more than 5 exist, process the 5 highest priority and leave the rest.
The P3 batch counts as 1 PR and does not consume the limit of 5.

### B5 — Apply fixes

FOR INDIVIDUAL ISSUES (P1 and P2):

  git checkout main
  git pull origin main
  git checkout -b fix/issue-<NUMBER>-<short-slug>

  Apply the minimal fix:
  - Change only what the issue describes. No adjacent refactoring.
  - Preserve all existing comments, docstrings, and type hints.
  - Only add pip dependencies if requirements.txt or pyproject.toml exists,
    and update it in the same commit.

FOR P3 BATCH:

  git checkout main
  git pull origin main
  git checkout -b fix/p3-batch-$(date +%Y-%m-%d)

  Apply all P3 fixes in sequence.
  Commit each fix individually:
    git commit -m "fix: <description> (closes #<NUMBER>)"

### B6 — Run local tests before pushing

python -m pytest --tb=short -q 2>&1 | tail -20
ruff check . --select E,F --quiet 2>&1 | head -20

If tests fail due to your change (not pre-existing failures):
  → Revert the change
  → Escalate the issue via B8 (mark as LOW confidence)
  → Move to next issue

If pre-existing test failures exist before your change, note them in the PR body
but do not treat them as blockers for your fix.

### B7 — Commit, push, and open PRs

FOR EACH INDIVIDUAL BRANCH:

  git add -p
  git commit -m "fix: <short description> (closes #<NUMBER>)

  - What was wrong: <one line>
  - What was changed: <one line>
  - Files touched: <list>"

  git push origin fix/issue-<NUMBER>-<short-slug>

  gh pr create \
    --repo BhurkeSiddhesh/Docu-AI-Search \
    --base main \
    --head fix/issue-<NUMBER>-<short-slug> \
    --title "fix: <same as commit subject>" \
    --body "## What this fixes
  Closes #<NUMBER>

  ## Root Cause
  <one paragraph — confirm by reading the code, not just copying the issue>

  ## Change Summary
  <file: what changed — one line per file>

  ## Merge Disposition
  <auto-merge — pending CI | awaits human review>
  Confidence: <HIGH | MEDIUM>
  Priority: <P1 | P2>

  ## Testing
  - Existing tests pass: yes / no (pre-existing failures noted above)
  - Covered by: <test name or 'no test — manual verification only'>

  ---
  Auto-fix by daily issue resolver on $(date +%Y-%m-%d)"

FOR THE P3 BATCH BRANCH:

  git push origin fix/p3-batch-$(date +%Y-%m-%d)

  gh pr create \
    --repo BhurkeSiddhesh/Docu-AI-Search \
    --base main \
    --head fix/p3-batch-$(date +%Y-%m-%d) \
    --title "fix: P3 DX batch fixes $(date +%Y-%m-%d)" \
    --body "## What this fixes
  <list each issue: Closes #N — short description>

  ## Change Summary
  <one line per file changed>

  ## Merge Disposition
  auto-merge — pending CI
  Priority: P3 batch

  ## Testing
  - Existing tests pass: yes / no
  - Covered by: <test names>

  ---
  Auto-fix by daily issue resolver on $(date +%Y-%m-%d)"

After every PR is created, comment on each linked issue:
  gh issue comment <N> \
    --repo BhurkeSiddhesh/Docu-AI-Search \
    --body "Fix submitted in PR: <PR_URL>.
    Merge disposition: <auto-merge pending CI | awaits human review>."

### B8 — CI gate and auto-merge (same logic as Phase A)

For every new PR where Merge Disposition is auto-merge:

1. Wait for checks:
   gh pr checks <PR_NUMBER> --watch --timeout 300

2. Fetch HEAD SHA and evaluate using the same logic as steps A3 and A4:
   - Non-CodeRabbit checks must ALL pass
   - CodeRabbit: skip if rate-limited, block if actual review issues found,
     treat as neutral if inconclusive

3. If all checks clear:
   gh pr merge <PR_NUMBER> \
     --repo BhurkeSiddhesh/Docu-AI-Search \
     --squash \
     --delete-branch
   → Comment on issue and close it.
   → Log as "Auto-merged ✓"

4. If checks fail (non-CodeRabbit):
   gh pr comment <PR_NUMBER> --repo BhurkeSiddhesh/Docu-AI-Search \
     --body "CI failed — auto-merge aborted. Requires human review."
   gh issue edit <N> \
     --repo BhurkeSiddhesh/Docu-AI-Search \
     --add-label "needs-human-review"
   → Log as "CI failed — awaiting review"

### B9 — Escalate LOW confidence issues

gh issue comment <N> \
  --repo BhurkeSiddhesh/Docu-AI-Search \
  --body "Auto-fix skipped — requires human review.

Reason: <why confidence is LOW — be specific>

Suggested approach: <concrete pattern, module to restructure,
or test that must exist first>

Estimated effort: <30 min | 2–4 hrs | 1 day>

---
Escalated by daily issue resolver on $(date +%Y-%m-%d)"

gh issue edit <N> \
  --repo BhurkeSiddhesh/Docu-AI-Search \
  --add-label "needs-human-review"

### B10 — Update fix log

Check if internal_fix_log.md exists at repo root. If not, create it.
Append a new entry:

## Fix Pass: YYYY-MM-DD

### Phase A — Existing PRs
PR #N — [Logic Enhancement] Description — Auto-merged ✓
PR #N — [Critical Bug] Description — Left for human review (P1)
PR #N — [Developer Experience] Batch — Auto-merged ✓
PR #N — [Logic Enhancement] Description — CodeRabbit rate-limited, bypassed — Auto-merged ✓
PR #N — [Logic Enhancement] Description — CodeRabbit review found issues — left for human
PR #N — [Logic Enhancement] Description — CI failing (non-CodeRabbit) — left open

### Phase B — New PRs
Issue #N — [Critical Bug] Description — PR #N opened — awaits human review
Issue #N — [Logic Enhancement] Description — PR #N opened — Auto-merged ✓
Issue #N — [Logic Enhancement] Descrip