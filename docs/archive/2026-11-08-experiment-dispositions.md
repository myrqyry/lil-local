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
- **Published and CI-verified:** the workbench and archival documentation are on GitHub `master`. The `Verify` CI run for the status-sync commit `06b7acb` completed successfully on 2026-11-08. CI is not a new backend/model-runtime qualification.

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

**Preservation and clean branch unification:**
- Three individually published annotated Gemma archive tags pin the exact worker, vision and audio commits. Each peeled GitHub SHA was checked.
- One additional published annotated tag, `archive/lil-local/legacy-branch-snapshot-2026-11-08`, points to the metadata-only archival commit `7606a25f6782e574526de5a001c961844156bb70`. Its tree equals the then-current `master` tree; its 27 parents (mainline plus 26 former branch-tip commits) preserve the old histories **without merging code into `master`**.
- The published archive was independently verified to reach **all 26 GitHub topic tips before deletion**. The two open pull requests (#16 and #17) were closed as superseded after confirming their implementations and tests were already on mainline (19 image-segmentation tests and 15 moderation tests passed).
- All 26 historical GitHub topic branches were retired after this check. The three local Gemma branch pointers were also retired after verifying their archive tags pointed to the exact tips. `git fetch --prune` left only `master` and `origin/master` as branch refs.
- The tracked-clean temporary Gemma `/tmp` worktree was retired earlier; only rebuildable `node_modules` were removed. There is **one worktree and one active branch**, with recoverable source history and no forced reset or forced push.

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
git log --oneline master..archive/lil-local/gemma4-audio-2026-11-08
# To reopen only for an explicit, qualifying experiment:
# git worktree add -b review/gemma4-audio /tmp/lil-local-gemma-recheck \
#   archive/lil-local/gemma4-audio-2026-11-08^{}
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

The former `origin/aster/*`, `origin/fix/*`, `origin/feat/*` and `origin/backup/*` branch names and exact tip SHAs are preserved in the [historical remote branch inventory](2026-11-08-remote-branch-inventory.md). They are no longer visible branches. All 26 tips remain reachable through the published `archive/lil-local/legacy-branch-snapshot-2026-11-08` tag. Recover a pointer only if subsequent source review warrants it:

```bash
git merge-base --is-ancestor <original-sha> archive/lil-local/legacy-branch-snapshot-2026-11-08^{}
git branch review/recovered <original-sha>
```

The archive **does not imply** that old divergent code was integrated, verified, or currently usable. It is recoverable history, not a hidden work queue.

## Operational rule for continuation

Default to `master` and current source. Do **not** resume an archived
branch merely because it exists or a prior assistant said it was
"next." An archived item is not part of `ACTIVE_WORK`. If a new reason
arises, verify the original blocker and the current runtime first.
No background jobs or automated experiment schedules were created.
