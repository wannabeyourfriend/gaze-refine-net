# Figure A2 — Fixation CCDF (kept, lightly cleaned from old appendix4)

## Why this figure
Justifies the 2-second target-acquisition window in the data-collection
protocol. Useful for the reproducibility appendix.

## Layout
Single panel, single curve. \linewidth × 2.5 in.

## Axes
- X-axis: waiting time (s), 0 to 2.4.
- Y-axis: complementary cumulative distribution function P(Twait > t).

## Curve
- Solid blue curve (computed once, no updates needed).
- Vertical red dashed line at t = 1.85 s (the 95th-percentile point).
- Annotate "1.85 s" near the line and a horizontal red dashed line at
  CCDF = 0.05.

## Color & style
- Blue curve, 1.6pt.
- Red guides at the 5 % crossing.

## Caption text
> Empirical complementary cumulative distribution of the time required
> for participants to enter and remain inside a 3-pixel target region
> after a calibration point appears. Over 95 % of points are reached
> within 1.85 s, supporting our 2-second pre-acquisition window for
> recording stable fixation data.
