---
title: Vitrines for Bach's Musical Offering
date: 2026-05-07 09:00:00 +0800
categories: [art]
tags: [music, visualization, bach]
pin: false
math: false
---
Render each canon in J. S. Bach's *Musical Offering* (BWV 1079) as a small interactive instrument inside a glass vitrine. The structural rule that defines each canon — palindrome, inversion, augmentation, modulation, perpetual cycle — becomes a literal geometric constraint you can rotate, scrub, and listen to.

The "musical offering" was Bach's 1747 reply to a theme dictated to him by Frederick the Great. Among its pieces are ten canons, most written down only as a single line with a Latin instruction telling the other voices how to derive their parts. Each instruction is a transformation: read the line backwards, read it upside-down, read it twice as slowly, and so on. The transformation *is* the score.

These visualizations try to make that transformation visible at a glance — every voice is a track of points in 3D space, and the canon's rule shows up as the geometric relationship between tracks (mirror plane, helical pitch, time compression). The audio is generated live by Tone.js from the same data the geometry is drawn from, so what you see and what you hear are two views of one source. (Bach left these canons unscored; the instrument labels in each vitrine are mine, chosen to make the geometry audible.)

---

## I. Crab Canon — palindrome

> *Two voices share one line. One plays it forwards, the other backwards. The piece is exactly its own retrograde.*

The simplest of the five visualized here in concept, and the strangest in effect: a single melody that, when reversed, harmonizes with itself.

<iframe
  src="/vitrine/crab_canon_vitrine_pbr.html"
  style="width:100%; aspect-ratio: 4/5; border:0; border-radius:12px; background:#f1eee6;"
  loading="lazy"
  allow="autoplay"
  title="Crab Canon · BWV 1079"></iframe>

<p style="text-align:right; font-size:0.85em; margin-top:-0.5em;">
<a href="/vitrine/crab_canon_vitrine_pbr.html" target="_blank" rel="noopener">↗ open standalone</a>
</p>

---

## II. Canon a 2 per Motum contrarium — mirror inversion

> *Violin and viola move in contrary motion as exact y-mirrors across the E♭/E axis. Every paired pitch sums to MIDI 127, while flute carries the Royal Theme.*

When the violin goes up by an interval, the viola goes down by the same interval. The two voices live in literal mirror symmetry — every paired note pitch-class summing to a constant. Above them, the flute carries the *Royal Theme* itself, the melody Frederick the Great gave Bach.

<iframe
  src="/vitrine/canon_per_motum_contrarium_vitrine_pbr.html"
  style="width:100%; aspect-ratio: 4/5; border:0; border-radius:12px; background:#f1eee6;"
  loading="lazy"
  allow="autoplay"
  title="Canon per Motum contrarium · BWV 1079"></iframe>

<p style="text-align:right; font-size:0.85em; margin-top:-0.5em;">
<a href="/vitrine/canon_per_motum_contrarium_vitrine_pbr.html" target="_blank" rel="noopener">↗ open standalone</a>
</p>

---

## III. Canon a 2 per Augmentationem, contrario Motu — mirror + slow-down

> *Violin plays the cello's line inverted and twice as slow. Every violin dash sits directly above its cello dot across the y-mirror. Viola weaves the Royal Theme on top.*

Two transformations stacked: the second voice plays the first inverted and at half speed. In the vitrine the cello's notes appear as compact dots below the mirror plane; the violin's appear as long dashes above — same x-position, opposite y. The augmentation reads as the difference between a dot and a dash.

<iframe
  src="/vitrine/canon_per_augmentationem_vitrine_pbr.html"
  style="width:100%; aspect-ratio: 4/5; border:0; border-radius:12px; background:#f1eee6;"
  loading="lazy"
  allow="autoplay"
  title="Canon per Augmentationem · BWV 1079"></iframe>

<p style="text-align:right; font-size:0.85em; margin-top:-0.5em;">
<a href="/vitrine/canon_per_augmentationem_vitrine_pbr.html" target="_blank" rel="noopener">↗ open standalone</a>
</p>

---

## IV. Canon a 2 per Tonos — endlessly rising

> *Each repetition modulates up a whole tone. Six turns of the helix close the spiral, returning home an octave higher.*

Bach inscribed this one with the motto *Ascendenteque Modulatione ascendat Gloria Regis* — "as the modulation rises, so may the King's glory rise." Six iterations later you've ascended a whole octave and arrived back at the home pitch class — the canon's tail splices straight into its head, a closed loop that nonetheless rises forever. Hofstadter opens *Gödel, Escher, Bach* with this canon as his first illustration of a strange loop.

<iframe
  src="/vitrine/canon_per_tonos_vitrine_pbr.html"
  style="width:100%; aspect-ratio: 4/5; border:0; border-radius:12px; background:#f1eee6;"
  loading="lazy"
  allow="autoplay"
  title="Canon per Tonos · BWV 1079"></iframe>

<p style="text-align:right; font-size:0.85em; margin-top:-0.5em;">
<a href="/vitrine/canon_per_tonos_vitrine_pbr.html" target="_blank" rel="noopener">↗ open standalone</a>
</p>

---

## V. Canon perpetuus a 3 — three-voice infinite wheel

> *Three voices lock at fixed phase on an endless wheel. Flute and violin are exact y-mirrors across the bass continuo, and the canon never resolves.*

The most architecturally complete of the five: three voices in fixed phase relationships on a closed orbit, with the bass continuo acting as the symmetry axis between flute and violin. The piece has no ending — it's a *canon perpetuus* that the performer is meant to fade out at will.

<iframe
  src="/vitrine/canon_a_3_perpetuus_vitrine_pbr.html"
  style="width:100%; aspect-ratio: 4/5; border:0; border-radius:12px; background:#f1eee6;"
  loading="lazy"
  allow="autoplay"
  title="Canon perpetuus a 3 · BWV 1079"></iframe>

<p style="text-align:right; font-size:0.85em; margin-top:-0.5em;">
<a href="/vitrine/canon_a_3_perpetuus_vitrine_pbr.html" target="_blank" rel="noopener">↗ open standalone</a>
</p>

---

## Notes
- inspired by Hofstadter's [*Gödel, Escher, Bach*](https://en.wikipedia.org/wiki/G%C3%B6del,_Escher,_Bach), Daniel Linssen's [print-gallery-of-an-artist](https://managore.itch.io/print-gallery-of-an-artist), and Netherlands Bach Society's [BWV 1079 playlist](https://www.youtube.com/watch?v=Lv5A1gy2oys)
- If your laptop fan starts spinning, that's the bloom postprocessing — you can drop to "front" view and the GPU load roughly halves.
