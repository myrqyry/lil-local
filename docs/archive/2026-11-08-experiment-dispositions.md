# Archived and closed Lil Local experiments — 2026-11-08

This is the **disposition ledger**, not a proposal to resume research.
The source of truth for code is live Git; evidence is linked by path below.
Older `ACTIVE_WORK.md` sections record what was happening *then*, not
ongoing assignments. No task listed here is running automatically.

## Finished and closed (not an active experiment)

**Live Vision still-image experiment workbench**
- **Integrated on local `master`:** `30a94df348a35258c0c54c0356ee8ec5a2d97e9b`.
- Real inference-backed four-run, in-session history; sampled owned float
  masks/depth maps, visual comparison and threshold re-decoding. Camera and
  existing inference paths were preserved.
- **Accepted evidence:** full `pnpm verify` succeeded; real Chromium
  `pnpm test:controlnet-models:browser` with Selfie/Depth and a real
  two-run ledger succeeded; older eight-model `pnpm test:vision:browser`
  succeeded. Details:
  [2026-11-08 workbench qualification](../verification/2026-11-08-vision-experiment-workbench.md).
- **Closure:** the promised first slice is done. Building a full tensor
  inspector, image-perturbation batch lab, persistent export/history,
  model catalog, or shader chaining is **not** a silently active next step.
  Reopen only with a new concrete use case/acceptance test.
- At archival time `master` is **one commit ahead of `origin/master`**;
  local commit is verified, but no push, deployment, or remote CI result is
  claimed for that commit.

## Archived, preserved, not merged

The following is one **five-commit experimental Gemma line**, with three
progressive branch tips. Focused TypeScript/unit tests were the highest
established verification; this **did not qualify** real Gemma 4 image/audio
inference or final in-browser worker behavior. Its three GitHub branch tips
were rechecked against local Git on 2026-11-08 before archiving.

| Candidate / original branch | Source commit | Local annotated archive tag | Stop reason |
| --- | --- | --- | --- |
| LiteRT-LM worker bridge: `aster/gemma4-lil-local-worker` | `4fd059cf129b3c923802d81e9e66bd2398501557` | `archive/lil-local/gemma4-worker-2026-11-08` | Worker import/WASM resolution changed; no accepted real end-to-end Gemma generation/browser-worker proof |
| Vision input capacity: `aster/gemma4-full-vision` | `18655676bfc83807ff9dc314ceec29b2db700e49` | `archive/lil-local/gemma4-vision-2026-11-08` | Configuration and tests do not establish actual image input, vision recognition, memory fit or backend |
| Native audio Blob input: `aster/gemma4-audio-input` | `78d3839a1fe6a876cac1d2ca947f88933a0bdf68` | `archive/lil-local/gemma4-audio-2026-11-08` | No real model audio-input/inference proof or downstream integration |

**Preservation performed:**
- Three local annotated Git tags pin the exact commits and explain why
  the experiments stopped.
- Original local branches remain, and **all three branch commits are
  preserved on GitHub** at the matching `origin/aster/gemma4-*` refs.
  The archive tags themselves are **local only**; do not assume they were pushed.
- The tracked-clean, temporary `/tmp/lil-local-gemma-bridge-20261107`
  worktree was retired with Git after verifying the only ignored contents
  were rebuildable `node_modules` directories. Original Git objects,
  branches, and remote refs were **not deleted**.
- The primary `master` working tree remains the only active worktree.

**Reopen criteria (not an open task):** a concrete consuming use case and
real accessible Gemma model, followed by independently observed browser
generation for worker, actual image-to-text for vision or real audio-input
inference for audio; confirm backend, cleanup, resource usage, and downstream
packed compatibility before evaluating any merge. A new model revision
requires its own evidence.

To inspect without modifying `master`:

```bash
git show archive/lil-local/gemma4-worker-2026-11-08
git show archive/lil-local/gemma4-vision-2026-11-08
git show archive/lil-local/gemma4-audio-2026-11-08
git log --oneline master..aster/gemma4-audio-input
# If explicitly reopening, create a NEW isolated worktree, not a blind merge:
# git worktree add /tmp/lil-local-gemma-recheck aster/gemma4-audio-input
```

## Not active; qualification boundaries, not hidden tasks

These are **known limits of existing features** or **ideas in a reference
archive**, not ongoing assistant research:

- Depth Anything V2 **Q8/WASM is real-browser qualified**, but FP16/WebGPU
  and Android latency, memory and output stability have not been measured.
  Reopen only when a target WebGPU device or Android consumer actually needs
  and can run a reproducible comparison.
- The four-stage tensor viewer and perturbation/batch workflows suggested by
  `computer-vision-lab.zip` were deliberately excluded from the finished
  first slice. There is no additional source branch or active process for
  them. A general candidate model list is an idea, **not** an implemented
  or promised model integration.
- Candidate model packages elsewhere in `packages/` retain the evidence
  level stated by their own qualification records. **Archiving an
  experiment does not convert candidate code into production support.**

## Other historical remote branch refs

There are older `origin/aster/*`, `origin/fix/*`, `origin/feat/*`,
and `origin/backup/*` refs predating this reconciliation. Their mere
presence does **not** mean they're active work, nor does it establish
their code is merged. For exact historical tip SHAs and independent
commit counts at the time of this audit, see
[remote branch inventory](2026-11-08-remote-branch-inventory.md).

**Disposition:** preserve the remote pointers without cherry-picking,
merging, renaming, or deleting. Old divergent changes need an explicit
comparison against current source before any claim they're useful or ready.

## Operational rule for continuation

Default to `master` and current source. Do **not** resume an archived
branch merely because it exists or a prior assistant said it was
"next." An archived item is not part of `ACTIVE_WORK`. If a new reason
arises, verify the original blocker and the current runtime first.
No background jobs or automated experiment schedules were created.
