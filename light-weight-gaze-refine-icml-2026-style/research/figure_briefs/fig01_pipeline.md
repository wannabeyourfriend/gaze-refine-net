# Figure 1 — Pipeline overview

## Replaces
Old ICML Figure 1 (three-tier comparison: end-to-end, eye-tracking vector
field fitting, multi-baseline neural refinement).

## Why we are revising it
The old figure portrayed our method as just adding a neural network on top of
classical baselines. Our honest evaluation shows that the network does not
help in the way the figure implied. The new figure must:

1. Still show the broader landscape (end-to-end vs tracker-based calibration).
2. Position our **online shrinkage** as a third tier that augments classical
   per-trial calibration with a lightweight learned per-trial bias correction.
3. Make clear that the **classical per-trial baseline (Stage I) remains the
   workhorse** — our contribution is an additive correction, not a
   replacement.

## Layout (single-column, 1.0 × \linewidth, ~5.5 in × 4 in)

Three horizontally stacked panels labelled (a), (b), (c). Same vertical axis
height; slim separators between rows.

### Panel (a) — End-to-end gaze estimation
- Input: cartoon of a webcam-captured face with eye crops highlighted.
- Pipeline icon: stylized CNN block (3 stacked convolution layers).
- Output: target dot vs predicted dot ~3-4° apart, with a red dashed arrow
  indicating the residual error.
- Caption strip below panel: *"End-to-end (e.g. iTracker, EFE): no per-user
  calibration; ≈ 2-4° error."*

### Panel (b) — Classical tracker-based calibration
- Input: cartoon of a head-mounted Pupil Labs Core eye tracker on a side-view
  human head.
- A short pipeline showing: raw gaze coordinates → per-trial classical
  calibrator (similarity, polynomial, RBF, sim+RBF — show as four small
  parallel boxes that converge into a single output).
- Output: target vs predicted, ~1° apart.
- Caption strip: *"Per-trial classical calibration (Stage I, existing): ≈ 1.0°
  residual; per-trial bias remains."*

### Panel (c) — Our pipeline: classical Stage I + online shrinkage Stage II
- Same head-mounted tracker icon as (b).
- Pipeline:
  1. Stage I block: per-trial classical calibrators producing anchor
     prediction (use the same four-box motif as (b), but with one box
     highlighted as the **anchor**).
  2. Stage II block: a small **"online recalibration"** widget — visualize
     K=12 extra fixation points being shown to the user and the resulting
     per-trial bias residuals (small arrows from anchor predictions to
     targets, mostly pointing the same direction to convey "bias structure").
  3. Stage III block: a tiny MLP icon (one hidden layer is enough — emphasize
     that it has ≈ 1k parameters; label it `λ-MLP`).
  4. Combine block: anchor prediction + λ ⊙ bias → refined prediction.
- Output: target vs predicted, ~0.7-0.8° apart, with the green arrow much
  shorter than panel (b).
- Caption strip: *"Ours: per-trial classical calibration + online K-point
  bias estimate, shrunk by a tiny learned λ. ~17% reduction at K=12."*

## Mandatory annotations
- Each pipeline block has a small "params" badge: end-to-end ~10M, classical 0
  (closed form fits), our shrinkage MLP ~1k.
- Highlight the `λ` symbol clearly in panel (c).
- Use consistent arrow styles between panels.

## Color & style
- Panel (a): grey/blue palette (cool, generic deep-learning).
- Panel (b): warm orange/yellow (classical methods).
- Panel (c): emerald green for the new shrinkage component, with the
  classical block in panel (b)'s warm color so the reader sees that we
  build *on top* of classical calibration, not replace it.
- Soft drop shadows behind each panel, not heavy.

## Caption text (for the LaTeX file)
> Three approaches to monocular gaze calibration. (a) End-to-end estimation
> from face/eye images requires no per-user calibration but suffers from
> 2–4° error. (b) Classical tracker-based calibration fits a parametric
> model (similarity, polynomial, RBF, sim+RBF) to ≈18 calibration fixations
> per session, reducing error to ≈1°. (c) Our pipeline retains the classical
> per-trial fit as an *anchor*, then appends a brief online recalibration of
> K extra fixations whose mean residual is multiplicatively shrunk by a
> ≈1 000-parameter MLP. The shrinkage policy is trained leave-one-subject-
> out, never seeing test-subject data, and recovers an additional 17% at
> K=12.
