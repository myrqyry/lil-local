# Improvement Review: Lil Tart
Session 1 | 2026-10-07

## Current value

### Observed
- Lil Tart combines a browser model lab with reusable LiteRT inference packages.
- Its differentiator is evidence: compile, inference, output and manual verification are distinct claims; session telemetry does not silently promote durable adapter verification.
- The repo has a CI `pnpm verify` gate, a separate browser qualification workflow, a packed-consumer compatibility suite, and a real runtime-path proof in the playground.
- The working tree already contains a user edit to `ACTIVE_WORK.md`; this review preserves it.

### Inferred
- The highest-leverage product loop is making observed proof easy to inspect, export, reproduce, and consume downstream.
- Adapter breadth is less valuable than having a smaller set of models with trustworthy, portable evidence.

### Assumptions to validate
- Downstream app authors want machine-readable receipts/recipes, not only a visual runtime readout.
- Users benefit from understanding verification provenance before downloading a potentially large model.
- No usage analytics, customer interviews, or open feature requests were present to quantify demand.

## Thirty candidate ideas

1. Export current-session runtime proof as a machine-readable JSON receipt.
2. Show the complete durable verification provenance on the selected model.
3. Turn a verified runtime path into a copyable integration recipe for a consumer app.
4. Add a per-model/browser/backend qualification matrix with explicit evidence links.
5. Run a tiny browser smoke qualification on every supported browser in scheduled CI.
6. Add a reproducible numeric-output comparison tool for candidate model conversions.
7. Let users compare two backends against the same model/input with latency and output-difference reporting.
8. Add bounded memory estimates and pre-load warnings based on model assets and known browser limits.
9. Persist a local qualification history so regressions can be compared across sessions.
10. Offer an offline package for self-hosting runtime WASM and model assets.
11. Add automatic qualification-case generation from adapter input/output contracts.
12. Provide a model compatibility recommender based on device/browser capability probes.
13. Add resumable downloads for large model assets.
14. Add a download queue with pause/cancel and storage-budget controls.
15. Add one-click downloadable model bundles for disconnected deployment.
16. Expose adapter input/output schemas as generated JSON Schema for consumer tooling.
17. Add a consumer starter template that pins all Lil Tart packages to one SHA.
18. Make model search filter by backend, evidence level, size, task, and required assets.
19. Add guided preflight diagnostics that translate runtime failures into specific next steps.
20. Add structured output validators for more adapter families.
21. Keep all inference locally in-browser and add a visible network-activity boundary report.
22. Add a benchmark suite with representative devices and durable performance baselines.
23. Add model provenance/SBOM records for every external model artifact.
24. Support multiple named model profiles and restore a user's preferred backend/model.
25. Add multi-model pipelines that chain embedding, retrieval, and generation.
26. Add collaboration/sharing of verified model recipes between users.
27. Publish the packages to a registry and version them independently.
28. Add a hosted model catalog API with live availability and download mirrors.
29. Replace per-adapter wiring with a generated adapter/plugin framework.
30. Add a model-quality ranking leaderboard across browsers and devices.

## Critical evaluation

Ratings use High/Medium/Low against impact, reach, frequency, differentiation, defensibility, and feasibility. Feasibility reflects repository evidence, not a delivery-time estimate.

| # | Evaluation | Decision |
|---|---|---|
| 1 | H impact / H reach among people reusing proof / recurring / H differentiation / M defensibility / H feasibility. Directly extends existing telemetry proof without changing qualification semantics. | **Keep — Do now.** |
| 2 | M impact / H reach / recurring when evaluating models / M differentiation / M defensibility / H feasibility. Verification fields already exist but the UI mostly collapses them to a status label, hiding the evidence users need to trust that label. | **Keep — Do now.** |
| 3 | H impact / H reach for downstream adopters / recurring / H differentiation / H compounding potential / M feasibility. Strong strategic direction, but a correct install recipe needs package revision, asset-base, backend, and worker semantics that a session proof alone cannot guarantee. | **Explore next; do not auto-generate yet.** |
| 4 | H impact / H reach / recurring / H differentiation / M defensibility / M feasibility. Would make gaps legible, but current evidence is mostly adapter metadata and lacks an authoritative environment-by-environment matrix. Avoid inventing coverage. | **Pass for now; build after evidence becomes structured.** |
| 5 | H impact / M reach / periodic / M differentiation / M defensibility / L feasibility. Browser downloads and model asset availability make this brittle and expensive; the existing workflow is explicitly manual and evidence-producing. | **Pass for now.** |
| 6 | H impact / M reach / occasional / H differentiation / H defensibility / M feasibility. Numeric parity is already an explicit qualification concern; a general-purpose evaluator needs model-specific reference data and tolerances not available for all candidates. | **Explore after selecting target models.** |
| 7 | M impact / M reach / occasional / M differentiation / L defensibility / M feasibility. It risks comparing different correctness paths and resource states; benchmark validity needs controlled repeated runs and model-specific output checks. | **Pass for now.** |
| 8 | H impact for constrained devices / M reach / before load / M differentiation / M defensibility / L feasibility. Asset bytes are not peak memory; reliable estimates need runtime/model-specific evidence, and false precision could cause bad advice. | **Pass until measured memory data exists.** |
| 9 | M impact / M reach / occasional / L differentiation / M defensibility / M feasibility. Useful, but local persistence adds retention and schema-migration costs before users have asked for it; a downloaded receipt solves the immediate portability need. | **Pass for now.** |
| 10 | M impact / M reach / occasional / M differentiation / M defensibility / M feasibility. Self-hosting WASM is already documented; packaging all model assets would be large and license/provenance-specific. | **Pass for now.** |
| 11 | M impact / M reach / occasional / M differentiation / M defensibility / M feasibility. Contracts describe tensor shapes but do not reliably encode semantic expected outputs, so generated cases would look more complete than they are. | **Pass; keep cases explicit and falsifiable.** |
| 12 | H impact / H reach / recurring / M differentiation / L defensibility / L feasibility. Capability probes cannot predict memory, driver correctness, host policy, or model-specific behavior; actual load/qualification remains the authority. | **Pass; avoid implying prediction is proof.** |
| 13 | M impact / M reach / occasional / L differentiation / L defensibility / M feasibility. Valuable for large assets, but current resolver/caching and streaming behavior need a scoped design; no direct demand evidence. | **Pass for now.** |
| 14 | M impact / M reach / recurring for multi-model users / L differentiation / L defensibility / M feasibility. Useful UX, but queue semantics and cancellation across concurrent runtime work are nontrivial and not the core differentiator. | **Pass for now.** |
| 15 | M impact / L-M reach / occasional / L differentiation / M defensibility / L feasibility. Bundling large models, licenses, updates, and browser-specific runtime assets would create a separate distribution product. | **Pass.** |
| 16 | M impact / M reach / recurring for integrators / M differentiation / M defensibility / M feasibility. Adapter specs are present, but many runtime input conventions are encoded in custom adapter code; generated schema would be incomplete. | **Pass until schemas become authoritative.** |
| 17 | M impact / M reach / one-time / L differentiation / L defensibility / H feasibility. Existing examples and compatibility harness already cover key consumption paths; a starter may be useful later but is not a top differentiator. | **Pass for now.** |
| 18 | M impact / H reach / recurring / L differentiation / L defensibility / H feasibility. Search already handles name/tags/id and family/download filters; additional filters have marginal value versus evidence work. | **Pass.** |
| 19 | H impact / H reach / recurring on failures / M differentiation / M defensibility / M feasibility. Strong candidate, but accurate guidance requires a catalog of runtime failure modes and tested remediations; generic hints risk misdiagnosing upstream errors. | **Explore after accumulating real failure evidence.** |
| 20 | H impact / M reach / recurring / M differentiation / M defensibility / M feasibility. Improves claim quality, but each task needs a meaningful semantic oracle; avoid expanding validators without representative fixtures. | **Explore per selected adapter.** |
| 21 | M impact / M reach / recurring / M differentiation / M defensibility / M feasibility. Local-first is already the premise; network reporting must account for model fetches, WASM, and optional third-party engines without implying zero network use. | **Pass until privacy/network confusion is evidenced.** |
| 22 | M-H impact / M reach / recurring / M differentiation / H defensibility / L feasibility. Device coverage and stable hardware baselines are unavailable; CI timing would be noisy and potentially misleading. | **Explore only with device lab access.** |
| 23 | M impact / M reach / per model / M differentiation / H trust value / M feasibility. Provenance matters, but universal licensing and source metadata require artifact-by-artifact data currently absent. | **Explore as metadata is curated.** |
| 24 | L-M impact / M reach / recurring / L differentiation / L defensibility / H feasibility. Preference persistence is ordinary convenience and could conflict with the explicit-load/backend control principle. | **Pass.** |
| 25 | H impact for app builders / M reach / recurring / M differentiation / H defensibility / L feasibility. Product-specific orchestration belongs in consuming apps by project invariant; package-level generic composition may be considered only after repeated consumer evidence. | **Pass in this repository for now.** |
| 26 | M impact / L reach presently / occasional / M differentiation / H network effects / L feasibility. No user identity, hosted service, or collaboration workflow exists; this changes the product's operating model. | **Pass.** |
| 27 | M impact / M reach / one-time migration / L differentiation / L defensibility / L feasibility. Current policy intentionally pins a single Git SHA and packages are private; registry release/version policy is a major strategic decision, not a quick improvement. | **Pass unless distribution needs change.** |
| 28 | M impact / M reach / recurring / L differentiation / L defensibility / L feasibility. Hosting/catalog operations and service dependency contradict local-first simplicity absent clear demand. | **Pass.** |
| 29 | M impact / M reach / recurring developer benefit / L differentiation / M defensibility / L feasibility. Current adapters have custom semantic behavior; a plugin framework risks abstracting prematurely and obscuring correctness. | **Pass.** |
| 30 | M impact / L reach / recurring / M differentiation / M defensibility / L feasibility. Leaderboards require standardized tasks, devices, licenses, and anti-gaming controls; current evidence is intentionally scoped to environment and backend. | **Pass.** |

## Kept ideas: concrete plans

### 1. Export current-session runtime proof as JSON

**What:** When real inference telemetry has produced a `RuntimePathProof`, let the user download a versioned JSON receipt containing that proof, an explicit `current-session-runtime-path` evidence kind, and wording that it does not promote durable verification. Do not create a download before real inference; do not mutate adapter metadata.

**Why:** The app already computes the most useful downstream facts—selected/requested/resolved backend, graph paths, fallbacks, timings, outputs, and timestamps—but they currently live only in the UI. Export makes this evidence portable and inspectable and creates a clean input for the future “add to my app” recipe without pretending it is a qualification result.

**Plan / implementation shape:**

```ts
interface RuntimePathReceipt {
  schemaVersion: 1
  evidenceKind: 'current-session-runtime-path'
  proof: RuntimePathProof
  interpretation: string
}

function serializeRuntimePathReceipt(proof: RuntimePathProof): string {
  return JSON.stringify({
    schemaVersion: 1,
    evidenceKind: 'current-session-runtime-path',
    proof,
    interpretation: 'This receipt records one browser session. It does not promote durable model verification.',
  }, null, 2) + '\n'
}
```

The implemented flow calls the serializer from a button rendered only alongside the existing non-null proof, packages the JSON as `application/json`, and downloads `<modelId>-runtime-session.json`. A focused unit test checks the version, evidence kind, observed backend, and non-promotion wording.

**Downsides:** The schema becomes a consumer-facing contract if other tools start depending on it; browser/runtime version and model artifact digest are not currently in `RuntimePathProof`, so this is not yet a reproducibility or qualification artifact. Keep the evidence kind explicitly session-scoped and version future changes.

**Confidence:** 91/100. Directly reuses trusted telemetry and has a narrow, reversible UI/data path; downstream adoption is inferred rather than measured.

### 2. Show durable verification provenance in the model details

**What:** Expand the selected adapter's durable verification record to show backend(s), verification date, note, and evidence reference, while preserving the existing status badge and keeping current-session proof visually separate. Render web URLs as safe new-tab links; render repository paths as text.

**Why:** An evidence status label alone cannot answer what was verified, where its evidence lives, or when it was recorded. This is especially important in a catalog where registered adapters must not be mistaken for working browser paths.

**Plan / implementation shape:**

```tsx
{verification && <details>
  <summary>Durable verification evidence</summary>
  {verification.backends?.length && <p>Verified backends: …</p>}
  {verification.verifiedAt && <p>Verified: {verification.verifiedAt}</p>}
  {verification.note && <p>{verification.note}</p>}
  {verification.evidence && <EvidenceReference value={verification.evidence} />}
</details>}
```

Keep this in the selected-model inspector rather than every card to avoid turning the library into a wall of metadata. Use `target="_blank"` with `rel="noreferrer"` only for `http(s)` values; do not treat arbitrary evidence strings as URLs.

**Downsides:** Some existing records have only a status or lack evidence links, so the disclosure can be sparse. It exposes metadata but does not validate that a referenced document still exists; the status remains a repository-maintained claim.

**Confidence:** 88/100. The data model and UI already carry these fields, making the change low-risk and directly aligned with the project's evidence-first value.

## Priority

### Do now
1. Export current-session proof as a versioned JSON receipt. **Implemented in this session.**
2. Show durable verification provenance in the selected model inspector. **Implemented in this session.**

### Do next
1. Design an “add to my app” recipe as a separate, explicitly qualified artifact after receipts include enough environment and artifact identity to be reproducible.
2. Add model-specific semantic/numeric validators only where an independent expected-output oracle exists.

### Explore
1. Structured per-browser/backend qualification matrix after environment coverage is recorded as durable data.
2. Diagnostics derived from real, catalogued failure signatures and tested remediations.
3. Provenance/SBOM metadata as source/license information is curated for actual model assets.

### Backlog / Pass for now
Candidates 5–30 are deferred or rejected in the evaluation table for concrete evidence, feasibility, policy, or product-fit reasons. Revisit only if demand or project constraints change; do not infer that the list represents user requests.

## Continuity and limits

- No previous Lil Tart strategy session existed; this is session 1.
- No usage analytics or customer research were found in the inspected project context.
- The implementation exports observed telemetry but does not record a new real-model browser run, promote durable verification, or establish consumer adoption.
- The pre-existing user modification to `ACTIVE_WORK.md` is preserved.
