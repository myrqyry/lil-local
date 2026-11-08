# Lil Local — mainline reconciliation, 2026-11-07

## Why this checkpoint exists

A working MediaPipe / Audio8 / SlimSAM / ControlNet Live sequence was present on
`aster/slimsam77-browser-qualification`, while the local `master` and
GitHub `origin/master` lagged. This looked like many lost features because earlier
commit-specific local branch names remained visible. Before acting, inspected
`git worktree list`, all local branch ancestry, remote refs (after
`git fetch --no-tags origin`), history, package tests, and browser receipts.
This audit distinguishes work **reachable from master**, equivalent-looking
but independently committed code, and genuinely still-outstanding experiments.

## Verified mainline integration

- Prior local `master`: `18bb741`; qualified development tip: `5ce3865`.
  `git merge-base --is-ancestor master 5ce3865` succeeded. There were
  **19 descendant commits** and zero commits exclusive to prior local `master`.
- Executed `git switch master && git merge --ff-only
  aster/slimsam77-browser-qualification`. The result was `master=5ce3865`
  with a clean working tree. The 19 original commits, authorship and history
  were preserved unchanged; this was not a squash, cherry-pick, or rebase.
- Previously completed real evidence at that exact code tip:
  `pnpm verify` **exit 0** (workspace typechecks/tests, packed external consumer,
  boundaries, runtime qualification, conversion checks, production build);
  `pnpm test:vision:browser` **exit 0** (all eight preexisting Vision tasks and
  fake-camera cleanup); `pnpm test:controlnet-models:browser` **exit 0**
  (real SHA-pinned Selfie Segmenter and immutable-revision Depth Anything V2
  Q8 inference through Lil Local's panel, local ONNX runtime and camera stop).
  CPU depth was **2,636 ms** for one frame, not real time. Real WebGPU/Android
  depth has **not** been qualified.
- On fetch, `origin/master=cebfe18` was an ancestor of local `master`, then
  **22 commits behind** `5ce3865`. No push, remote deletion, merge to GitHub,
  deployment or release was performed. The follow-on documentation checkpoint
  adds another local commit.
- The historical source branch had no other checkout/worktree. The separate
  `/tmp/lil-local-gemma-bridge-20261107` worktree remained untouched.

## Redundant local branch pointers safely covered by master

The following local topic refs pointed to ancestors of `5ce3865`. Each commit
is retained in the now-consolidated `master` history. After recording their
names and old tips below, all **15 merged local branch refs were deleted**
using `git branch -d`, with an ancestor check before and after. The final
source branch initially failed `-d` because its **upstream remote branch**
was behind local master; after verifying its tip was included in master, its
upstream setting was unset and `git branch -d` succeeded. No forced deletion
or history rewrite was needed. Local branches went from **19 to 4**. Their
remote counterparts are **not** removed.

| Former local branch | Recorded tip |
| --- | --- |
| `aster/mediapipe-parakeet-asr` | `88607d5` |
| `aster/mediapipe-vision-substrate` | `5a83858` |
| `aster/mediapipe-audio-substrate` | `c67ad15` |
| `aster/mediapipe-retrieval-substrate` | `ec0122b` |
| `aster/mediapipe-pose-substrate` | `8b5db73` |
| `aster/mediapipe-object-detection` | `4f8fbdc` |
| `aster/mediapipe-image-classifier` | `01b2dbb` |
| `aster/mediapipe-holistic-substrate` | `3862000` |
| `aster/mediapipe-semantic-segmentation` | `b0996cc` |
| `aster/mediapipe-magic-touch` | `9a9d5f5` |
| `aster/mediapipe-blazeface-detector` | `e80946c` |
| `aster/mediapipe-language-detection` | `09f1186` |
| `aster/audio8-litert-candidate` | `866edb8` |
| `aster/mediapipe-tts-qualification` | `5b436ae` |
| `aster/slimsam77-browser-qualification` | `5ce3865` |

## Real outstanding local development — not folded in

Three named Gemma branches are nested tips of one separate line from `5b436ae`:

- `aster/gemma4-lil-local-worker` — `4fd059c`
- `aster/gemma4-full-vision` — `1865567`
- `aster/gemma4-audio-input` — `78d3839`

They contain the five commits `dd0134b`, `bd251f8`, `4fd059c`,
`1865567`, and `78d3839`. The existing audio-input worktree is clean.
Read-only `git merge-tree --write-tree master aster/gemma4-audio-input`
completed with exit 0 (no textual merge conflicts). In that separate
worktree, `pnpm --filter @lil-local/text-gen typecheck` and `test` passed
(109 unit tests). Those checks do **not** establish native Gemma image/audio
inference, browser loader success, real device compatibility, or cross-branch
full verification. Leave it on its existing branch and worktree until its
opt-in integration is reviewed and qualified against current master.

## Older remote-only refs

Fetched remote branches were **not** deleted, rewritten or merged blindly.

- Many October 20 remote topic refs add the same historical
  `chore(repo): stop tracking Superpowers planning files` change at different
  branch bases. Their patches are not all commit-identical. `docs/superpowers`
  is no longer tracked on local master; preserve those historical source refs
  rather than importing their obsolete cleanup commits.
- `origin/aster/forma-image-segmentation`, `origin/aster/phonon-live-onnx`,
  `origin/aster/privacy-filter-smallest`, and
  `origin/aster/sharp-onnx-webgpu` have unique historical commits.
  Corresponding current implementations and verification records already
  exist on master under `cebfe18`, `1c8bcd6`, `503d3d2` and `a94bfe0`
  respectively. **Same capability does not prove exact patch equivalence**;
  these original refs remain available for any line-level semantic audit.
  SHARP model inference is not thereby claimed production-qualified.
- `origin/backup/litert-upstream-hardening-pre-sync` retains four old,
  non-ancestral commits. Keep the backup intact; its semantics were not
  proven equivalent in this reconciliation.
- Other remote refs have not been promoted merely because they have a
  nonzero `git cherry` result. The remote audit is an inventory, not proof
  that every legacy implementation can safely be replayed.

## Recovery and follow-up

`git log --oneline --graph master` exposes the consolidated sequence.
The prior local mainline tip remains addressable as `18bb741`, and the
verified integration tip remains addressable as `5ce3865`.
No history rewriting, forced reset, local/remote publishing, or model cache
cleanup was done. The decision still open is whether and when to qualify
Gemma's worker/vision/audio line and whether to synchronize the verified
master branch to GitHub; neither is an implied side effect of this pass.
