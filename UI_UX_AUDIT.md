# Neurofluxion product-wide UI/UX audit

**Audit date:** 2026-09-10
**Scope:** `frontend/src` as it exists before the redesign implementation. The backend/model registry was inspected only where it shapes a visible state or input contract.

This is the implementation audit required before changing UI code. It is based on the actual component graph, styles, stores, hooks, and API calls—not on a generic dashboard template.

## Executive summary

Neurofluxion already has unusually strong neural-network teaching and diagnostic capability. The main problem is not missing features; it is that five workspaces, two comparison systems, three levels of simulator modes, and several overlay systems are presented with nearly the same visual weight. The current shell also makes status and actions compete with the work surface.

The redesign will keep the product's warm barley/orange identity and the neural visual language, but establish a quieter product shell:

- one persistent workspace navigation system with the same vocabulary on desktop and mobile;
- one page-header grammar;
- one primary action per workspace;
- contextual status instead of a global command/status strip that behaves like a second nav;
- progressive disclosure for Lab diagnostics and Simulator research tools;
- metadata-driven Prediction and Models experiences;
- reusable accessible primitives for tabs, dialogs, toggles, progress, tooltips, and actions.

The existing architecture is serviceable and does not need a backend rewrite. The main refactor is composition: `App.tsx` becomes a product shell, the large pages become clearer workspace compositions, and existing domain visualizations remain the implementation source of truth.

---

## 1. Global shell findings

### Current implementation

- `frontend/src/App.tsx` owns routing-like mode state (`predict`, `train`, `models`, `lab`, `simulator`), backend bootstrap, global keyboard handling, command palette, rail, mobile nav, status strip, and a few Lab/Simulator actions.
- `frontend/src/index.css` owns the shell under `.ncc-*`, plus a second set of simulator and design-system rules. It is 2,838 lines and mixes semantic tokens, legacy `--neural-*` aliases, page rules, simulator rules, and component rules.
- The command strip (`.ncc-command-strip`) combines system status, a workspace action, and command-palette entry. On Lab and Simulator it duplicates page/workspace controls.
- The shell starts in Lab without an explicit onboarding state. `activeModel ?? "Dense_v3"` is displayed even when no model is active.
- The global `statusText` is derived from `/models/available`, so it communicates model availability as if it were whole-application readiness.
- Simulator adds its own header and status bar through `UnifiedLayout`, creating a second shell hierarchy.
- The shell does not expose a product wordmark/title in the collapsed rail, only a Brain icon.

### Priority findings

- **P0:** system status, navigation, contextual actions, and command palette are merged in one bottom strip; users cannot tell whether it is status or an action bar.
- **P0:** Simulator has a nested global-looking header/status system inside the global shell.
- **P1:** default `Dense_v3` is fictional/stale when no model is loaded.
- **P1:** startup bootstrap is tied to legacy trained-model availability and does not represent registry/backend state accurately.
- **P1:** no explicit product-level landmark structure (`header`/`nav`/`main`) that remains stable across workspaces.
- **P2:** route transitions animate the entire page but do not announce workspace changes.
- **P2:** shell loading/error state is a banner without a persistent recovery context.

### Decision

Keep one global shell. Replace the hover-expand rail with a persistent desktop sidebar (expanded at >= 1024px) and a compact mobile header + bottom workspace navigation using the same five labels: **Prediction, Lab, Simulator, Training, Models**. Move reload to a utility menu/button in the sidebar footer and command palette. Keep system state in a small, non-competing status area in the sidebar/header. Remove workspace-specific quick actions from the global strip; the page owns its primary action.

---

## 2. Navigation findings

### Actual surfaces

- Desktop rail: `App.tsx` `.ncc-rail` / `.ncc-link`.
- Mobile nav: `App.tsx` `.ncc-mobile-tabs`.
- Simulator sections: `UnifiedLayout.tsx` `NeuralTabs` with `BUILD`, `RUN`, `ANALYZE`, `ADVANCED`.
- Prediction family tabs: `PredictionMode.tsx` custom `role="tablist"`.
- Command palette: `App.tsx`, keyboard shortcut `Ctrl/Cmd+K`.

### Problems

- Desktop rail depends on hover/focus to reveal labels; its collapsed state is not self-explanatory.
- Mobile vocabulary is inconsistent: `Predict`, `Sim`, and `Train` instead of the global labels.
- `Reload` is styled like a primary rail item and lives in the navigation stack.
- `NeuralTabs` supplies selected state but not arrow-key navigation, `tabIndex` roving behavior, or `aria-controls`/tabpanel wiring.
- Command palette has focus trapping logic but no focus restoration, no command descriptions, and only a small set of actual commands.

### Decision

Use a sidebar with visible labels and active workspace description. Keep mobile bottom nav as the same information architecture, not a shortened vocabulary. Use a shared `WorkspaceNavItem` visual language. Make Simulator tabs a workspace-local stepper with explicit descriptions and a single active tab panel. Keep keyboard navigation in both the palette and tabs.

---

## 3. Prediction findings

**Responsible files:** `components/prediction/PredictionMode.tsx` (1,139 lines), `pages/ArchitectureComparisonPage.tsx`, `components/probability/ProbabilityLandscape.tsx`, `components/comparison/*`, `store/predictionStore.ts`, model catalog types in `types/index.ts`.

### Current strengths

- Registry-fed family selection and per-family model dropdown already exist.
- ANN canvas has undo, redo, clear, grid, samples, auto-predict, and preprocessing crop/resize logic.
- CNN upload/examples and RNN text presets use different input controls appropriately.
- Result includes confidence ring, probability visualizations, trace disclosure, history, and latency.
- The architecture comparison button is preserved and opens a separate comparison experience.

### Problems

- The 1,139-line component owns catalog loading, three input experiences, inference, history, visualizations, keyboard shortcuts, and comparison routing. This makes it difficult to maintain consistent states.
- The page header currently contains result latency, readiness dot, comparison action, family tabs, and model selector; it competes with the input/result work surface.
- Empty, loading, unavailable, and error states are mostly embedded in the result ring; the primary next step is not always obvious.
- `autoPredict` makes inference happen while drawing but the primary action is then not always visibly `Predict`; this is useful, but it must be an opt-in/supporting behavior.
- ANN controls and CNN/RNN controls use different raw button/select patterns instead of shared primitives.
- Metadata is shown as a dense inline row and then again in the expanded trace.
- The probability view offers bars/radial/terrain at the same level as the result, before the user knows the prediction; this is advanced analysis and should be subordinate.
- `ArchitectureComparisonPage` runs the legacy `model_type` path against the same digit pixels for ANN/CNN/RNN. It should not be presented as a comparison of the unrelated registry models/tasks. This is a distinct legacy architecture exploration workflow and needs an explicit scope note.
- The comparison page has mouse-only canvas handlers and no touch/pointer drawing support.

### Priority

- **P0:** clarify registry model/task/input compatibility and prevent the comparison screen from implying a benchmark across unrelated pretrained models.
- **P1:** make `Predict` the single primary action; demote auto-predict, probability views, trace, and history.
- **P1:** extract input, model context, result summary, analysis disclosure, and history into workspace primitives.
- **P2:** pointer/touch canvas and keyboard shortcut guard for text/select fields.

### Page specification

- **Purpose:** Use a selected model to turn an input into an interpretable inference.
- **Primary user:** learner or practitioner testing one model.
- **Primary object:** current inference.
- **Primary action:** Predict.
- **Information order:** Task/input contract → model readiness → input → prediction/confidence/latency → top predictions → optional analysis → history.
- **Keep:** all real input modalities, samples, undo/redo, preprocessing, history, architecture comparison entry, trace data.
- **Move/collapse:** detailed metadata, probability modes, activations, explanation, and history into labeled disclosures after the result.
- **Mobile:** stack Input then Result; sticky actions are limited to Predict and a compact model context, never a three-column layout.

---

## 4. Lab findings

**Responsible files:** `pages/LabPage.tsx`, `components/lab/LabHeader.tsx`, `InputStage.tsx`, `StagePipeline.tsx`, `PlaybackControls.tsx`, `PassDirectionIndicator.tsx`, `components/lab/backward/*`, `comparison/*`, `inspection/*`, `saliency/*`, `counterfactual/*`, `flow/*`, `neuron/*`, `profiler/*`, all Lab stores and `store/labStore.ts`.

### Current strengths

- Lab has the richest educational model in the application: input preprocessing, stage pipeline, forward pass, loss, backward pass, trained/untrained activations, saliency, weights, neuron biography, counterfactuals, profiler, and architecture comparison.
- The store has coherent domain state for forward/backward progression and advanced experiments.
- Stage cards include explanation, equations, shapes, and activation metadata.
- Playback keyboard shortcuts exist: Space, arrows, Home, End, speed keys, architecture/dataset keys, and advanced-tool keys.

### Problems

- `LabPage.tsx` renders almost every capability at page level and mounts several overlays regardless of whether the user is in a relevant state.
- `DataFlowRibbon` is a persistent fixed left rail, `PlaybackControls` is a persistent fixed bottom rail, and the global command strip is also fixed at the bottom. These three surfaces compete for the same viewport and can overlap on smaller screens.
- Lab header puts architecture and dataset switches in the page header, while the actual input and pipeline are below; the action hierarchy is unclear.
- `StageCardV2` renders explanation/math but the richer `StageVisualization`, `StageExplanation`, and section components are not part of the active StageCard path. This is a capability discoverability/unused-path issue, not a reason to delete them without checking tutorial use.
- Advanced actions are primarily keyboard discoverable: `F` flow, `P` profiler, `W` counterfactual, `C` comparison, `N` neuron. There is no visible shortcut reference or consistent action launcher.
- Keyboard handler in `LabPage` fires while typing/selecting because it does not guard editable targets. Space always prevents default.
- Several overlays are full-screen/fixed but have no consistent dialog semantics or focus management. `ArchitectureComparison` is a full-screen workspace; `WeightInspector`, `CounterfactualExplorer`, `NeuronBiographyPanel`, and `CostProfiler` need task-appropriate panel treatment.
- The current `Esc` handler resets the pipeline as well as closing tools, which is surprising and potentially destructive.
- `StageExplanation` contains corrupted/placeholder icon text (`??`) and is not on the active card path.

### Priority

- **P0:** remove overlapping global/fixed control surfaces and prevent `Esc` from unexpectedly resetting an experiment.
- **P1:** progressive disclosure: Core (Input, Pipeline, Playback), Understand (loss/backward/trained toggle), Explore (saliency/weights/neuron/counterfactual), Advanced (profiler/comparison).
- **P1:** make current stage/next stage/completion and playback state visible without requiring keyboard knowledge.
- **P1:** add shortcut help and editable-target guards.
- **P2:** unify overlays around side panel/full-screen workspace/modal semantics.

### Page specification

- **Purpose:** Understand how a signal changes through a network.
- **Primary user:** learner inspecting one forward/backward computation.
- **Primary object:** current pipeline stage.
- **Primary action:** run/step the pass.
- **Information order:** Input → current stage → current computation/visualization → next stage → playback → explanation.
- **Keep:** all educational math and advanced diagnostic stores/capabilities.
- **Collapse:** saliency, weights, neuron biography, counterfactual, profiler, architecture comparison until requested.
- **Mobile:** playback becomes a compact bottom sheet/toolbar with an explicit “More tools” disclosure; flow ribbon becomes inline stage progress.

---

## 5. Simulator findings

**Responsible files:** `pages/SimulatorPage.tsx`, `components/layout/UnifiedLayout.tsx`, `components/tabs/BuildTab.tsx`, `RunInspectTab.tsx`, `AnalyzeTab.tsx`, `AdvancedTab.tsx`, `components/simulator/*`, `store/sessionStore.ts`, `store/simulatorStore.ts`, `store/computationStore.ts`, `store/architectureStore.ts`, `hooks/useSimulatorApi.ts`.

### Current strengths

- Simulator already has a meaningful `BUILD → RUN → ANALYZE → ADVANCED` information architecture.
- Architecture builder supports many layer types, presets, templates, validation, parameter/FLOP summaries, import/export, datasets, and hyperparameters.
- Run/Inspect integrates real build/forward/backward APIs, layer selection, equations, network canvas, and execution state.
- Research tools exist behind the user mode (`beginner`, `standard`, `research`).

### Problems

- `SimulatorPage.tsx` automatically creates demo architecture, builds it, generates random input, and performs a forward pass. The user gets no explicit demo label or explanation of what happened.
- It hard-codes `http://127.0.0.1:8000/api`, bypassing the shared API client/proxy; it is a preview/production portability bug.
- `UnifiedLayout` adds its own title, mode selector, tabs, status bar, and next-action hint under the global shell. This is the nested-header problem.
- `sessionStore.getNextAction()` says “Click Build Model” but `ArchitectureBuilder` button says “Build”; wording is inconsistent. It includes emoji in a system hint.
- Status bar permanently shows Model, Dataset, Device, Status even when not actionable.
- `BuildTab` presents architecture, dataset, hyperparameters, and import/export with similar panel weight; import/export is advanced but visible in standard/research modes.
- `RunInspectTab` is large and mixes build, forward, backward, execution controls, device, network, context, and equations. It also uses raw/legacy CSS in places.
- `AnalyzeTab` contains empty placeholders for metrics/profiling/comparison. `hasProfiling` is hardcoded false and comparison is a placeholder, so the page feels incomplete.
- `AdvancedTab` is a vertical dump of Interpretability, Adversarial, Compression, Embeddings, and Generative; in Research mode all five are equally prominent.
- `NetworkCanvas` intentionally falls back to stable pseudo-random weights when real weights are absent. This is a visualization fallback, not model inference, but it must be labeled as illustrative to avoid false confidence.
- `LayerCard` is a clickable `div role="button"` containing inputs and a remove button; nested interactive semantics are problematic. The remove button renders `?` instead of a semantic icon/label.
- Simulator controls use the legacy `http://127.0.0.1` endpoints while other paths use `apiClient`.

### Priority

- **P0:** remove hard-coded backend base in `SimulatorPage.tsx` and `RunInspectTab.tsx`; use shared client.
- **P1:** make demo-first state explicit (“Demo network” + what ran + next action) rather than silently initialized.
- **P1:** collapse simulator status to actionable status, and keep the global shell hierarchy singular.
- **P1:** make each tab's single purpose and primary action explicit; move research tools behind sub-navigation/disclosure.
- **P2:** fix nested interactive semantics and misleading fallback labels.

### Page specification

- **Purpose:** Build and execute an educational neural network.
- **Primary user:** experimenter constructing a network.
- **Primary object:** current network/session.
- **Primary action by tab:** Build Model / Run Forward / Inspect Result / Open Advanced Tool.
- **Keep:** all layer types, datasets, import/export, network visualization, forward/backward, equations, analysis, research tools.
- **Remove:** none of the capabilities; remove only redundant chrome and placeholder framing.
- **Mobile:** tabs remain horizontal scroll/segmented with clear active step; builder is a single-column flow; network canvas gets a minimum usable viewport; advanced tools use one-at-a-time disclosure.

---

## 6. Training findings

**Responsible files:** `components/training/TrainingMode.tsx`, `hooks/useTrainingSocket.ts`, `types/index.ts`.

### Current strengths

- WebSocket status/reconnect/queueing is implemented.
- Start, pause, resume, stop, save, live batch data, epoch history, logs, and charts are present.
- Save errors are surfaced.

### Problems

- Training page displays eight metrics plus three aggregate metrics before the configuration/run story; no clear “Training Run” primary object.
- Configuration uses raw `select`/`input` and does not validate epochs/batch size/learning rate or show explanatory constraints.
- Start/stop uses raw buttons while reconnect/save uses `NeuralButton`.
- Batch loss, accuracy, and gradient norm share one chart despite different scales/units; it is difficult to interpret.
- Three charts compete equally. Logs are visible in the main configuration column instead of Advanced/Diagnostics.
- No clear empty/loading/paused/completed narrative beyond metric placeholders and a status subtitle.
- `TrainingStatus` type does not include `stopping` even though the socket normalizes to it.

### Priority

- **P1:** lifecycle layout: Configure → Run → Monitor → Evaluate → Save.
- **P1:** primary metrics first; secondary metrics in “More metrics”.
- **P1:** move logs into a diagnostics disclosure and split charts by meaningful units.
- **P2:** validate config and standardize button/input primitives.

### Page specification

- **Purpose:** Train and save a model.
- **Primary user:** user running an experiment.
- **Primary object:** training run.
- **Primary action:** Start Training; during run Pause/Stop; after completion Evaluate/Save.
- **Keep:** WebSocket behavior, reconnect, all metrics/log data, save workflow.
- **Collapse:** precision/recall/F1, gradient norm, batch stream, raw logs, optimizer details.
- **Mobile:** sticky run action at bottom, configuration sections stacked, charts horizontally scrollable but not squeezed.

---

## 7. Models findings

**Responsible files:** `components/models/ModelsMode.tsx`, `types/index.ts`, `api/client.ts`.

### Current strengths

- Pretrained registry catalog is real and metadata-driven.
- Family grouping, status reasons, load/unload, skeleton, error/retry, and trained model management are present.
- Trained model reload/use/delete behavior is preserved.

### Problems

- Registry cards show nearly every property simultaneously, including source/weights/license and preprocessing; technical metadata dominates the “Use model” decision.
- Pretrained cards have only Load/Unload, not an obvious `Use Model` action. The user must infer that loaded means usable.
- “Runnable” is derived from statuses but availability and input contract are not summarized as a single decision.
- Filesystem path is still the most prominent property on trained model cards.
- Status vocabulary omits explicit cached/active/missing distinctions in UI even though the product needs them.
- No filter/search or family/status controls for a catalog that already has multiple families.

### Priority

- **P1:** model cards lead with identity, family/task/input, readiness, and Use Model; move source/weights/path into details.
- **P1:** make unavailable reason and load state explicit.
- **P2:** add search/filter and consistent technical-details disclosure.

### Page specification

- **Purpose:** Discover, inspect, load, and select models.
- **Primary user:** someone choosing a model for Prediction or managing a saved experiment.
- **Primary object:** model.
- **Primary action:** Use Model; secondary Load/Unload, Inspect, Reload, Remove.
- **Keep:** every catalog entry and trained management operation.
- **Collapse:** source URLs, paths, weights, licenses, low-level preprocessing details.
- **Mobile:** one card per row, details disclosure, full-width Use/Load action.

---

## 8. Design-system findings

### Existing primitives

`design-system/components`: `NeuralButton`, `NeuralPanel`, `NeuralInput`, `NeuralSelect`, `NeuralTabs`, `NeuralBadge`, `NeuralModal`, `NeuralNotification`, `NeuralNumber`, `NeuralProgress`, `NeuralScrollArea`, `NeuralSkeleton`, `NeuralSlider`, `NeuralToggle`, `NeuralTooltip`, `PageHeader`.

### Problems

- `NeuralButton` has variants but no loading state, no explicit `type="button"`, and pages frequently use raw buttons beside it.
- `NeuralTabs` has no keyboard arrow behavior, roving tab index, or panel relationship.
- `NeuralModal` focuses the first control and traps Tab, but does not restore focus and can be unlabeled when `label` is absent. It treats every overlay as a modal even when inspection needs a side panel.
- `NeuralToggle` implements a switch on a `span` inside a label rather than a native input/button; accessible name/state behavior is fragile.
- `NeuralProgress` lacks progressbar semantics and value text.
- `NeuralTooltip` is hover-only and not keyboard/screen-reader complete.
- `NeuralBadge` is visual-only; status changes are not announced.
- `PageHeader` is a good base but has no eyebrow, status slot, or explicit primary-action convention.
- `index.css` retains legacy `--neural-*` aliases while newer semantic tokens coexist. Usage should be measured before removal.
- Typography token file says Inter while global CSS uses a system stack; this is harmless but inconsistent.

### Decisions

Keep the primitives and harden them before page polish: focus, type, semantics, loading, disabled, responsive sizes, and reduced motion. Add small product primitives (`WorkspaceShell`, `WorkspaceNav`, `StatusPill`, `Disclosure`, `ShortcutHelp`, `EmptyState`) instead of one-off markup. Do not rewrite the canvas/visualization primitives.

---

## 9. Responsive findings

Current responsive CSS handles basic stacking at 768/900/1200px, but it is mostly desktop-first collapse:

- The hover rail disappears at 767px and becomes mobile nav; vocabulary changes.
- Prediction uses fixed 280–360px input/result regions and a three-column grid at xl; it will fit poorly at 320px without intentional ordering.
- Lab has two fixed surfaces (`DataFlowRibbon`, `PlaybackControls`) plus global mobile nav/command strip; overlap risk is high.
- Simulator hides left/right columns at breakpoints instead of moving their content into deliberate disclosures.
- Training charts are responsive containers but the surrounding metric grid is dense at 320–375px.
- Full-screen comparison assumes desktop columns and has no touch canvas.

Target checks: 320, 375, 430, 768, 1024, 1280, 1440, 1920+. Implement mobile as a deliberate input→result flow, not only `flex-direction: column`.

---

## 10. Accessibility findings

### Existing positives

- Some active/current semantics, labels, roles, and `aria-live` surfaces already exist.
- Global `:focus-visible` styling is present.
- Coarse pointer touch-target rule exists.
- Reduced-motion and reduced-transparency media queries exist.

### Gaps

- Global shortcuts fire in text inputs/selects/textareas and contenteditable contexts (`LabPage`, `PredictionMode`).
- Mobile and desktop nav use different visible labels.
- Command palette focus restoration and screen-reader active option behavior need hardening.
- Custom tabs/toggles/canvas use partial semantics.
- Canvas `role="img"` is not keyboard-operable; users can not create input without a pointer.
- Many raw buttons with icons have title but no consistent accessible name; some are text placeholders (`?`).
- Modal/overlay focus hierarchy is inconsistent.
- Status changes are often visual text without `role=status`/live announcements.
- Color is used heavily for architecture/status; labels and shape/text must carry meaning too.

Priority: P1 for keyboard/editable guards, nav/tabs/dialog primitives, button names, status semantics, and overlay focus. P2 for canvas keyboard drawing support and full screen-reader narration of visualizations.

---

## 11. State-management/UI architecture findings

- App-level mode state is local to `App.tsx`; this is acceptable for the current no-router architecture but should be isolated behind navigation helpers.
- `labStore` is intentionally rich and is the source of truth for Lab execution and advanced state. Do not duplicate it in UI state.
- `predictionStore.ts` and `comparisonStore.ts` both export a hook named `useComparisonStore` for different domains. This is confusing and increases accidental imports; names should become `usePredictionComparisonStore` and `useLabComparisonStore` in a safe compatibility refactor.
- `sessionStore` owns Simulator shell status and history, while `simulatorStore`/`computationStore` own execution. The UI should read both through workspace selectors rather than inventing new status state.
- Several stores are real capability backends but have no current visible surface in the main tabs (assistant, experiments, augmentation, spatial, etc.). They should not be deleted in this redesign; they are advanced capability candidates and need explicit ownership/disclosure if surfaced.
- `SimulatorPage` and `RunInspectTab` bypass `apiClient` with localhost/127.0.0.1 URLs. This is a technical blocker for the live preview and should be fixed before visual polish.

---

## 12. Redundant, dead, or misleading UI inventory

| Finding | File/component | Classification | Decision |
|---|---|---:|---|
| Hover-only collapsed rail | `App.tsx`, `.ncc-rail` | P1 | Replace with persistent desktop nav |
| Mobile abbreviated labels | `App.tsx` | P1 | Use Prediction/Lab/Simulator/Training/Models |
| Reload as rail destination | `App.tsx` | P1 | Move to utility action/command |
| Global status strip + Lab playback + Lab flow ribbon | `App.tsx`, `LabPage.tsx`, `DataFlowRibbon.tsx`, `PlaybackControls.tsx` | P0 | Keep one contextual playback surface and inline/optional flow |
| Simulator nested title/status | `UnifiedLayout.tsx` | P0 | Make it the Simulator workspace header/stepper, not second global shell |
| Fictional fallback model label | `App.tsx` | P1 | Show actual active model or “No model selected” |
| Prediction dense header metadata | `PredictionMode.tsx` | P1 | Move into model context/result details |
| Probability views before result | `PredictionMode.tsx` | P2 | Keep behind Analysis disclosure |
| Architecture comparison scope ambiguity | `ArchitectureComparisonPage.tsx` | P0 | Label as legacy same-input architecture exploration; do not imply registry benchmark |
| Lab advanced overlays always mounted | `LabPage.tsx` | P1 | Keep state/capabilities but render only when open/relevant |
| Escape resets Lab pipeline | `LabPage.tsx` | P0 | Escape closes topmost surface; reset requires explicit action |
| `StageCardV2` bypasses rich visualizations | `components/lab/stages/*` | P1 | Add explicit Deep Dive disclosure; do not remove visual components |
| `StageExplanation` placeholder icons | `StageExplanation.tsx` | P2 | Replace with semantic Lucide icons or remove unused wrapper |
| Simulator auto-run without demo label | `SimulatorPage.tsx` | P1 | Explicit Demo state and “what happened” summary |
| Hard-coded simulator API URLs | `SimulatorPage.tsx`, `RunInspectTab.tsx` | P0 | Shared `apiClient` paths |
| `hasProfiling = false` and comparison placeholder | `AnalyzeTab.tsx` | P1 | Honest empty state with action/ownership, not fake live capability |
| `LayerCard` nested role-button controls | `LayerCard.tsx` | P1 | Use selectable article/button header; inputs remain independent |
| Layer remove `?` | `LayerCard.tsx` | P2 | Trash icon + accessible label |
| Registry technical details always expanded | `ModelsMode.tsx` | P1 | Details disclosure and Use Model hierarchy |
| Raw controls mixed with Neural primitives | Training, Models, Prediction, Lab | P1 | Migrate high-frequency actions to primitives |

---

## 13. Implementation order

1. Global information architecture and shell.
2. Navigation, mobile model, command palette, status utility.
3. Design-system hardening and semantic tokens.
4. Consistent page header/action hierarchy.
5. Models registry hierarchy and use flow.
6. Prediction input/model/result/analysis composition.
7. Lab progressive disclosure/playback/overlays/shortcut safety.
8. Simulator demo state, tabs, API client, contextual status.
9. Training lifecycle hierarchy and chart/log disclosure.
10. Responsive/accessibility pass.
11. Reduced-motion and restrained motion polish.
12. Typecheck/build and final OCD audit.

## Acceptance checklist

- [ ] A user can identify the current workspace in one second.
- [ ] Every workspace has one primary object and one primary action.
- [ ] Desktop and mobile share workspace vocabulary.
- [ ] No global strip duplicates page actions.
- [ ] Prediction never presents incompatible registry models as one benchmark.
- [ ] Lab advanced capability remains available but is not all visible at once.
- [ ] Simulator demo is explicit and API calls work through the shared client.
- [ ] Training primary metrics and lifecycle dominate raw diagnostics.
- [ ] Models lead with Use/Load decisions and disclose technical metadata.
- [ ] Keyboard navigation works without firing shortcuts in editable controls.
- [ ] Dialogs/tabs/toggles/progress have complete semantics.
- [ ] 320px remains usable.
- [ ] TypeScript/build pass and backend behavior remains untouched.

---

## 14. Post-implementation OCD audit — 2026-09-10

The sections above are intentionally preserved as the pre-code audit and decision log. The implementation pass now covers the requested order without removing the underlying Lab or Simulator capability.

### Implemented

- **Shell / IA:** a persistent desktop sidebar and mobile header/bottom navigation use the same five labels: Prediction, Lab, Simulator, Training, Models. Reload remains a utility action, not a workspace.
- **Header grammar:** `PageHeader` now supports eyebrow, subtitle, status, actions, and contextual children. Simulator has one local workflow header rather than competing global chrome.
- **Design system:** buttons have explicit types and loading semantics; progress uses progressbar semantics; tabs have roving focus and arrow/Home/End behavior; toggles/tooltips/modals include keyboard and reduced-motion support; semantic tokens and reduced transparency/motion rules are present.
- **Prediction:** family tabs remain registry-fed; each family has one model selector; the architecture comparison entry is preserved; Input → model → Result is primary; probability/trace/history analysis is disclosed; latency and class-count feedback is immediate; selection from Models is deferred until Prediction mounts.
- **Lab:** the page now states Input → Computation → Internal behavior → Explanation; flow is inline/optional; profiler, saliency, counterfactual, comparison, and neuron inspection remain available through Explore; Escape closes the topmost surface and no longer resets the pipeline; editable controls are excluded from shortcuts.
- **Simulator:** demo initialization is explicitly labeled only after a real build/forward response; unavailable backend state is not presented as a successful demo; API calls and WebSocket URLs are host-safe and use the shared client/proxy; Build, Run, Analyze, and Advanced have distinct hierarchy; research tools are disclosure-based; simulator training uses the existing WebSocket path rather than fabricated progress.
- **Training:** Configure → Train → Monitor → Evaluate → Save is explicit; run lifecycle states and connection/retry state are visible; primary metrics and accuracy/loss charts lead; secondary metrics and raw diagnostics are disclosed; validation prevents invalid numeric settings.
- **Models:** registry search/family filtering, unavailable reasons, technical-details disclosure, lazy load/unload, Use model, trained-model reload/delete, and Models → Prediction handoff are preserved.
- **Responsive/accessibility:** skip link, focus-visible styling, status/live regions, explicit alerts, coarse pointer targets, 320px-oriented stacking rules, and mobile-safe simulator/Lab surfaces are implemented. Page-level shortcuts and global command shortcuts guard text-entry contexts.

### Verification

- `frontend/npm run lint` — **passes** with zero errors and warnings.
- `frontend/npm run build` — **passes** (`tsc` + `vite build`).
- `npm ci --dry-run --ignore-scripts --no-audit --no-fund` — **passes**, confirming the package manifest and lockfile are consistent.
- Vite dev server — **serves HTTP 200** on port 5173; the shared proxy is configured for `/backend/*`, including WebSocket forwarding.
- Backend pure smoke coverage — **12 tests pass** (`tests/test_simulator_smoke.py` and `tests/test_utils.py`).
- Backend live/runtime coverage — **not executable in the base workspace** because FastAPI/Uvicorn/TensorFlow are not installed; the attempted isolated TensorFlow install was removed after the large wheel download failed. No model training or Simulator WebSocket `start` action was invoked.
- `git diff --check` — **passes**.

### Remaining non-blocking verification

- The legacy simulator CSS block still coexists with newer semantic shell rules; it is isolated from the new shell and can be removed in a later visual-diff pass after browser screenshots are available.
- Vite reports the stale Browserslist database and a large application chunk; the prior `three-mesh-bvh`/Three.js export warning is resolved by aligning Three.js and its type package to `0.159.0`.
- Full WCAG testing with a screen reader, browser zoom, and real backend data still needs to be performed in a browser environment with the Python dependencies installed. Static focus, keyboard-guard, reduced-motion, live-region, and responsive CSS review is complete; runtime browser verification is environment-dependent.

### Final acceptance state

- [x] Workspace vocabulary and global information architecture are coherent.
- [x] Every major workspace has a primary object, primary action, and next step.
- [x] Existing advanced capabilities remain available through progressive disclosure.
- [x] Registry-driven model selection and trained-model management are preserved.
- [x] Simulator remains unmodified under `backend/simulator/**`, uses lazy/CPU-safe paths, and has no hard-coded browser API base.
- [x] Keyboard shortcuts are retained with editable-control guards and command-layer discovery.
- [x] Responsive and reduced-motion/transparency foundations are implemented.
- [x] Pointer/touch drawing is supported for Prediction and the retained architecture-comparison canvas.
- [x] TypeScript, lint, dependency-lock, pure backend smoke, and diff verification pass.
- [ ] Full live-backend visual/accessibility audit remains environment-dependent.

## 15. Significant-component decision matrix

| Area / actual component | Decision | Rationale / implementation result |
|---|---|---|
| `App.tsx` shell | REFINE | Retains local workspace routing and bootstrap, but delegates visible chrome to `WorkspaceNav`, mobile nav, `PageHeader`, and command layer. |
| `WorkspaceNav.tsx` / `MobileWorkspaceNav` | REPLACE | Replaces the hover-dependent legacy rail with persistent, labeled desktop/mobile navigation using identical workspace vocabulary. |
| `CommandPalette.tsx` | KEEP / REFINE | Keeps the real command-layer concept; adds grouped results, active-option semantics, focus return, Escape, keyboard traversal, and empty results. |
| `PageHeader.tsx` | KEEP / REFINE | Becomes the shared header grammar with eyebrow, purpose, state, actions, and contextual child slot. |
| `NeuralButton`, `NeuralProgress`, `NeuralTabs`, `NeuralToggle`, `NeuralTooltip`, `NeuralModal` | KEEP / REFINE | Preserves existing primitive API direction while adding explicit button types, loading, progressbar semantics, tab keyboard behavior, switch semantics, tooltip focus, modal focus trapping/restoration, and reduced-motion compatibility. |
| Prediction family/model selector | KEEP / REFINE | Preserves registry-driven family tabs and one selector per family; filters out incompatible/unavailable models from runnable choices and explains unavailable entries in Models. |
| Prediction canvas / CNN upload / RNN text | KEEP / REFINE | Real task-specific input modes remain primary; controls are ordered around Predict and touch/mobile sizing is preserved through responsive layout. |
| Prediction result ring and probability views | KEEP / MOVE | Confidence/result remains immediate; probability bars/radial/terrain and trace are behind Analysis. |
| `ArchitectureComparisonPage` | KEEP / REFINE | Existing Compare Architectures entry remains; it is not replaced by an unrelated registry benchmark. |
| `LabHeader` / `LabPage` core pipeline | KEEP / REFINE | Keeps architecture, dataset, forward/backward pass, truth/loss, stages, and playback; reframes the page around the current computation and next step. |
| `DataFlowRibbon` | MOVE / REFINE | Moves from a competing fixed rail into optional inline flow context. |
| `PlaybackControls` | KEEP / MOVE | Keeps all play/pause/step/reset/speed operations, but relocates the surface to avoid mobile-nav and sidebar competition. |
| Lab profiler, saliency, counterfactual, comparison, neuron, weight surfaces | KEEP / MOVE | Capabilities remain; Explore actions and existing overlays provide progressive disclosure rather than deleting diagnostics. |
| `UnifiedLayout` | REPLACE | Converts the Simulator's nested shell into a focused workspace header, Build/Run/Analyze/Advanced stepper, honest status area, and demo note. |
| `BuildTab` / `RunInspectTab` | KEEP / REFINE | Existing architecture, device, execution, network, equations, forward/backward, and training capabilities remain with explicit tab purpose and shared API client. |
| `AnalyzeTab` | REPLACE | Reorganizes actual metrics/profiler/comparison surfaces into primary run health plus honest empty/disclosure states; removes the permanent fake profiling flag. |
| `AdvancedTab` | REFINE / MOVE | Keeps all research tools but places each in a native disclosure and makes Research mode gating explicit. |
| `TrainingMode` | REPLACE composition, KEEP capability | Preserves WebSocket training, pause/resume/stop, charts, metrics, logs, and save; changes emphasis to Training Run lifecycle and diagnostic disclosures. |
| `ModelsMode` catalog cards | REFINE | Keeps metadata/status/load/unload but makes Use model / Load and use the decision hierarchy and moves paths/source/license into technical details. |
| Saved trained-model management | KEEP / REFINE | Reload, use, delete, disk/memory state remain available and are visually separated from pretrained registry discovery. |
| Legacy `.ncc-*` shell CSS | REMOVE | No live JSX references remained after the shell migration; removed to prevent obsolete command-strip/rail styles from competing with semantic shell rules. |
| `backend/simulator/**` | KEEP | Protected implementation was not modified. Frontend Simulator communication now uses `apiClient` and existing WebSocket endpoints. |
