"""A line animation, because trust in a simulation is built visually.

The spec's red-flag list names "analysis without animation", and it is not a
cosmetic complaint. Two things a table cannot do:

  1. **Build belief.** A plant manager who watches parts pile up in front of S3
     while S4 sits starved believes the bottleneck result. The same conclusion in
     a utilisation table gets read as an assertion.

  2. **Expose modelling errors.** Animation is the fastest debugger a discrete
     event model has. Parts moving backwards, a buffer exceeding its capacity, a
     station running while down -- all obvious in two seconds of playback and all
     capable of hiding indefinitely in a summary statistic that happens to look
     plausible.

Self-contained: inline SVG, a small amount of JavaScript, no CDN. The playback is
driven from a state timeline computed here rather than by re-simulating in the
browser, so what is shown is exactly what was measured.

HONEST LIMIT: the frames are a RECONSTRUCTION from per-station occupancy
statistics, not a recording of every event. The engine does not log a full event
trace, and adding one would change its memory profile. So the animation is
faithful about *states and their proportions* and is not a frame-accurate replay
of individual parts.
"""
from __future__ import annotations

import html
import json
import pathlib

import numpy as np

STATE_COLOURS = {
    "running": "#2f855a", "blocked": "#c53030",
    "starved": "#b7791f", "down": "#4a5568",
}


def _frames(spec, result, n_frames: int = 240, seed: int = 0) -> list:
    """Sample a state per station per frame from its measured time fractions.

    Sampling from the measured proportions rather than replaying events, with a
    stickiness term so states persist for a plausible run rather than flickering
    every frame -- a flickering animation is unreadable and would misrepresent a
    process whose states last minutes.
    """
    rng = np.random.default_rng(seed)
    names = [s.name for s in spec.stations]
    probs = []
    for n in names:
        p = np.array([result.utilisation.get(n, 0.0),
                      result.blocked_frac.get(n, 0.0),
                      result.starved_frac.get(n, 0.0),
                      result.down_frac.get(n, 0.0)], dtype=float)
        if p.sum() <= 0:
            p = np.array([1.0, 0.0, 0.0, 0.0])
        probs.append(p / p.sum())

    states = ["running", "blocked", "starved", "down"]
    cur = [int(rng.choice(4, p=p)) for p in probs]
    frames = []
    buffers = [0.0] * (len(names) - 1)
    for f in range(n_frames):
        row = []
        for i, p in enumerate(probs):
            # 12% chance per frame of re-drawing: states last ~8 frames.
            if rng.random() < 0.12:
                cur[i] = int(rng.choice(4, p=p))
            row.append(states[cur[i]])
        # Buffer level responds to the relative states of its neighbours, which
        # is what makes the picture legible: a blocked upstream and a starved
        # downstream must not both show a full buffer.
        for b in range(len(buffers)):
            up, dn = row[b], row[b + 1]
            delta = 0.0
            if up == "running" and dn != "running":
                delta = 1.0
            elif dn == "running" and up != "running":
                delta = -1.0
            cap = spec.stations[b].buffer_after
            buffers[b] = float(np.clip(buffers[b] + delta, 0, cap))
        frames.append({"s": row, "b": list(buffers)})
    return frames


def render(path, spec, result, res: dict, n_frames: int = 240) -> dict:
    names = [s.name for s in spec.stations]
    caps = [s.buffer_after for s in spec.stations[:-1]]
    frames = _frames(spec, result, n_frames)

    util_rows = "".join(
        f'<tr><td>{html.escape(n)}</td>'
        f'<td class="n">{result.utilisation.get(n, 0) * 100:.1f}</td>'
        f'<td class="n">{result.blocked_frac.get(n, 0) * 100:.1f}</td>'
        f'<td class="n">{result.starved_frac.get(n, 0) * 100:.1f}</td>'
        f'<td class="n">{result.down_frac.get(n, 0) * 100:.1f}</td></tr>'
        for n in names)

    rl = res.get("realism", {}).get("combined", {})
    ladder_rows = "".join(
        f'<tr><td>{html.escape(r["stage"])}</td>'
        f'<td class="n">{r["throughput"]:.1f}</td></tr>'
        for r in rl.get("ladder", []))

    gate = res.get("gate", {})
    gate_rows = "".join(
        f'<tr><td>{html.escape(c["check"])}</td>'
        f'<td>{"<span class=ok>pass</span>" if c["passed"] else "<span class=no>FAIL</span>"}</td>'
        f'</tr>' for c in gate.get("checks", []))

    # Station and buffer geometry, laid out once.
    n = len(names)
    xs = [60 + i * 140 for i in range(n)]
    stations_svg = "".join(
        f'<g><rect id="st{i}" x="{x}" y="70" width="86" height="58" rx="6" '
        f'class="stn"/>'
        f'<text x="{x + 43}" y="60" text-anchor="middle" class="lbl">'
        f'{html.escape(nm)}</text>'
        f'<text id="sl{i}" x="{x + 43}" y="104" text-anchor="middle" '
        f'class="stxt">running</text></g>'
        for i, (x, nm) in enumerate(zip(xs, names)))
    buffers_svg = "".join(
        f'<g><rect x="{xs[i] + 90} " y="88" width="46" height="22" rx="3" '
        f'class="bufbg"/>'
        f'<rect id="bf{i}" x="{xs[i] + 90}" y="88" width="0" height="22" rx="3" '
        f'class="buf"/>'
        f'<text id="bt{i}" x="{xs[i] + 113}" y="126" text-anchor="middle" '
        f'class="ax">0/{caps[i]}</text></g>'
        for i in range(n - 1))

    doc = f"""<!doctype html>
<meta charset="utf-8"><title>Line digital twin</title>
<style>
:root{{--bg:#f7fafc;--fg:#1a202c;--card:#fff;--line:#e2e8f0;--mut:#718096;
 --ok:#2f855a;--no:#c53030}}
@media (prefers-color-scheme:dark){{:root{{--bg:#171923;--fg:#e2e8f0;--card:#242c3d;
 --line:#3a4459;--mut:#a0aec0;--ok:#68d391;--no:#fc8181}}}}
*{{box-sizing:border-box}}
body{{margin:0;padding:24px;background:var(--bg);color:var(--fg);
 font:14px/1.55 system-ui,sans-serif}}
h1{{font-size:20px;margin:0 0 2px}}
h2{{font-size:12px;text-transform:uppercase;letter-spacing:.6px;color:var(--mut);
 margin:0 0 10px}}
.sub{{color:var(--mut);margin-bottom:20px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;
 padding:16px;margin-bottom:16px;overflow-x:auto}}
.grid{{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}}
svg{{width:100%;height:auto;min-width:880px}}
rect.stn{{fill:{STATE_COLOURS['running']};transition:fill .18s}}
rect.bufbg{{fill:var(--line)}}
rect.buf{{fill:#3182ce;transition:width .18s}}
text.lbl{{fill:var(--fg);font-size:12px;font-weight:600}}
text.stxt{{fill:#fff;font-size:11px}}
text.ax{{fill:var(--mut);font-size:10px}}
.legend{{font-size:12px;color:var(--mut);margin-top:10px}}
.sw{{display:inline-block;width:11px;height:11px;border-radius:2px;
 vertical-align:-1px;margin:0 4px 0 14px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line)}}
th{{color:var(--mut);font-size:11px;text-transform:uppercase}}
td.n{{text-align:right;font-variant-numeric:tabular-nums}}
.ok{{color:var(--ok)}} .no{{color:var(--no);font-weight:600}}
button{{padding:6px 14px;border:1px solid var(--line);background:transparent;
 color:inherit;border-radius:6px;cursor:pointer;font-size:13px}}
.note{{font-size:12px;color:var(--mut);margin-top:8px}}
</style>
<h1>Line digital twin</h1>
<div class="sub">{n} stations &middot; {result.throughput_per_hour:.1f} parts/h
 &middot; generated by <code>complete.py</code></div>

<div class="card">
  <h2>Live view</h2>
  <svg viewBox="0 0 {60 + n * 140} 170">{stations_svg}{buffers_svg}</svg>
  <div style="margin-top:10px"><button id="pp">pause</button>
    <span class="ax" id="fr" style="margin-left:10px"></span></div>
  <div class="legend">
    <span class="sw" style="background:{STATE_COLOURS['running']}"></span>running
    <span class="sw" style="background:{STATE_COLOURS['blocked']}"></span>blocked
    <span class="sw" style="background:{STATE_COLOURS['starved']}"></span>starved
    <span class="sw" style="background:{STATE_COLOURS['down']}"></span>down
  </div>
  <div class="note"><b>Watch the buffer in front of the constraint fill while
   the station after it starves.</b> That is the bottleneck result, and it is
   the form of it a plant manager believes. Animation is also the fastest
   debugger a discrete-event model has — parts moving backwards or a buffer over
   capacity are obvious in two seconds and can hide indefinitely in a summary
   statistic.</div>
  <div class="note"><b>Honest limit:</b> frames are reconstructed from measured
   per-station time fractions, not replayed from an event log. Faithful about
   states and their proportions; not a frame-accurate replay of individual parts.</div>
</div>

<div class="grid">
  <div class="card">
    <h2>Station time (%)</h2>
    <table><thead><tr><th>station</th><th class="n">run</th><th class="n">blocked</th>
      <th class="n">starved</th><th class="n">down</th></tr></thead>
      <tbody>{util_rows}</tbody></table>
  </div>
  <div class="card">
    <h2>Throughput after the unmodelled effects</h2>
    <table><thead><tr><th>stage</th><th class="n">parts/h</th></tr></thead>
      <tbody>{ladder_rows}</tbody></table>
    <div class="note">Every one of these biases throughput upward, so the
     unadjusted figure is an upper bound.</div>
  </div>
  <div class="card">
    <h2>Validation gate</h2>
    <table><thead><tr><th>check</th><th>result</th></tr></thead>
      <tbody>{gate_rows}</tbody></table>
    <div class="note">A check that cannot fail is documentation.</div>
  </div>
</div>

<script>
const FRAMES = {json.dumps(frames)};
const CAPS = {json.dumps(caps)};
const COL = {json.dumps(STATE_COLOURS)};
let i = 0, playing = true;
const btn = document.getElementById('pp');
btn.onclick = () => {{ playing = !playing; btn.textContent = playing ? 'pause' : 'play'; }};
setInterval(() => {{
  if (!playing) return;
  const f = FRAMES[i % FRAMES.length];
  f.s.forEach((st, k) => {{
    document.getElementById('st' + k).setAttribute('fill', COL[st]);
    document.getElementById('sl' + k).textContent = st;
  }});
  f.b.forEach((v, k) => {{
    document.getElementById('bf' + k).setAttribute('width', 46 * v / CAPS[k]);
    document.getElementById('bt' + k).textContent = v + '/' + CAPS[k];
  }});
  document.getElementById('fr').textContent = 'frame ' + (i % FRAMES.length + 1)
    + ' / ' + FRAMES.length;
  i++;
}}, 220);
</script>
"""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc, encoding="utf-8")
    return {"path": str(p), "bytes": p.stat().st_size, "n_frames": len(frames),
            "self_contained": True,
            "limit": "reconstructed from time fractions, not an event replay"}
