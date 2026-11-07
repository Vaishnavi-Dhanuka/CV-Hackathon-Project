"""Self-contained cinematic introduction for the LUMINA Streamlit app."""

import streamlit as st


_EXPERIENCE_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {
  color-scheme: dark;
  --night: #070b18;
  --navy: #0b1020;
  --panel: #111a2e;
  --blue: #38bdf8;
  --violet: #818cf8;
  --white: #f8fafc;
  --muted: #94a3b8;
  --line: rgba(148, 163, 184, .17);
  --page-height: 820px;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
* { box-sizing: border-box; }
html { background: var(--night); scroll-behavior: smooth; }
body {
  margin: 0;
  background: var(--night);
  color: var(--white);
  overflow: hidden;
}
a { color: inherit; }
button, a { -webkit-tap-highlight-color: transparent; }
.experience {
  position: relative;
  width: 100%;
  overflow: hidden;
  background: var(--night);
}
.hero {
  position: relative;
  display: grid;
  min-height: 620px;
  height: clamp(620px, calc(var(--page-height) * .9), 860px);
  align-items: center;
  isolation: isolate;
  overflow: hidden;
  padding: clamp(36px, 7vw, 104px) clamp(24px, 8vw, 120px);
  background:
    radial-gradient(ellipse at 74% 44%, rgba(56, 189, 248, .11), transparent 30%),
    radial-gradient(ellipse at 85% 72%, rgba(129, 140, 248, .13), transparent 34%),
    linear-gradient(120deg, #070b18 4%, #0b1020 61%, #0d1428 100%);
}
.hero::before, .hero::after {
  position: absolute;
  z-index: -1;
  border: 1px solid rgba(129, 140, 248, .11);
  border-radius: 50%;
  content: "";
  pointer-events: none;
}
.hero::before { width: 68vw; height: 68vw; right: -24vw; top: -34vw; }
.hero::after { width: 52vw; height: 52vw; right: -16vw; bottom: -34vw; border-color: rgba(56, 189, 248, .09); }
.hero-grid {
  position: absolute;
  z-index: -1;
  inset: 0;
  background-image: linear-gradient(rgba(148, 163, 184, .025) 1px, transparent 1px), linear-gradient(90deg, rgba(148, 163, 184, .025) 1px, transparent 1px);
  background-size: 64px 64px;
  mask-image: linear-gradient(90deg, transparent, black 80%);
}
.topline {
  position: absolute;
  top: 28px;
  right: clamp(24px, 6vw, 80px);
  left: clamp(24px, 6vw, 80px);
  display: flex;
  align-items: center;
  justify-content: space-between;
  color: var(--muted);
  font-size: 10px;
  font-weight: 650;
  letter-spacing: .2em;
  text-transform: uppercase;
}
.topline-mark { display: inline-flex; align-items: center; gap: 10px; color: var(--white); }
.topline-mark i {
  width: 9px;
  height: 9px;
  border: 1px solid var(--blue);
  border-radius: 50%;
  background: rgba(56, 189, 248, .28);
  box-shadow: 0 0 16px rgba(56, 189, 248, .8);
}
.hero-copy { position: relative; z-index: 2; width: min(650px, 62%); }
.eyebrow {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 0 0 25px;
  color: var(--blue);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: .22em;
  text-transform: uppercase;
}
.eyebrow::before { width: 30px; height: 1px; background: var(--blue); content: ""; }
.wordmark {
  margin: 0 0 21px;
  color: var(--white);
  font-size: clamp(64px, 10.5vw, 144px);
  font-weight: 580;
  letter-spacing: .105em;
  line-height: .9;
  text-shadow: 0 0 50px rgba(129, 140, 248, .17);
}
.hero h1 {
  max-width: 620px;
  margin: 0;
  color: var(--white);
  font-size: clamp(34px, 5vw, 66px);
  font-weight: 500;
  letter-spacing: -.055em;
  line-height: 1.04;
}
.hero h1 span {
  color: #b6c8ff;
  background: linear-gradient(100deg, #f8fafc 10%, #a5b4fc 58%, #7dd3fc);
  background-clip: text;
  -webkit-text-fill-color: transparent;
}
.hero-description {
  max-width: 470px;
  margin: 24px 0 0;
  color: #a4b0c5;
  font-size: clamp(14px, 1.5vw, 17px);
  line-height: 1.8;
}
.hero-actions { display: flex; flex-wrap: wrap; align-items: center; gap: 22px; margin-top: 37px; }
.explore {
  position: relative;
  display: inline-flex;
  min-height: 50px;
  align-items: center;
  justify-content: center;
  gap: 13px;
  overflow: hidden;
  border: 1px solid rgba(125, 211, 252, .48);
  border-radius: 5px;
  background: linear-gradient(110deg, rgba(56, 189, 248, .15), rgba(129, 140, 248, .19));
  padding: 0 22px;
  color: var(--white);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: .09em;
  text-decoration: none;
  text-transform: uppercase;
  box-shadow: 0 10px 38px rgba(56, 189, 248, .09), inset 0 1px rgba(255,255,255,.1);
  transition: background .25s ease, box-shadow .25s ease, border-color .25s ease, transform .2s ease;
}
.explore:hover, .explore:focus-visible {
  border-color: var(--blue);
  background: linear-gradient(110deg, rgba(56, 189, 248, .25), rgba(129, 140, 248, .29));
  box-shadow: 0 12px 42px rgba(56, 189, 248, .2), inset 0 1px rgba(255,255,255,.14);
}
.explore:focus-visible, .text-link:focus-visible, .tech-card:focus-visible { outline: 2px solid var(--blue); outline-offset: 4px; }
.explore-arrow { color: #7dd3fc; font-size: 18px; transition: transform .2s ease; }
.explore:hover .explore-arrow { transform: translateX(4px); }
.hero-note {
  color: #718098;
  font-size: 10px;
  letter-spacing: .14em;
  text-transform: uppercase;
}
.restoration-visual {
  position: absolute;
  top: 50%;
  right: clamp(2%, 6vw, 92px);
  width: clamp(300px, 41vw, 590px);
  aspect-ratio: 1.12;
  transform: perspective(1200px) translateY(-50%) rotateX(var(--tilt-y, 0deg)) rotateY(var(--tilt-x, 0deg)) scale(var(--scroll-scale, 1));
  transform-style: preserve-3d;
  transition: transform .38s cubic-bezier(.2,.7,.2,1), opacity .35s ease;
  pointer-events: none;
}
.visual-aura {
  position: absolute;
  inset: 13% 8%;
  border-radius: 24px;
  background: radial-gradient(ellipse, rgba(56,189,248,.19), rgba(129,140,248,.09) 42%, transparent 72%);
  filter: blur(34px);
}
.visual-frame {
  position: absolute;
  inset: 8% 5% 13%;
  overflow: hidden;
  border: 1px solid rgba(125,211,252,.42);
  border-radius: 13px;
  background: #101a2b;
  box-shadow: 0 25px 80px rgba(0,0,0,.42), 0 0 50px rgba(56,189,248,.12), inset 0 1px rgba(255,255,255,.08);
}
.visual-topline {
  position: absolute;
  z-index: 4;
  top: 12px;
  right: 13px;
  left: 13px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  color: #d6e5f7;
  font-size: 8px;
  font-weight: 700;
  letter-spacing: .14em;
  text-transform: uppercase;
  text-shadow: 0 1px 5px #07101f;
}
.visual-live { display: inline-flex; align-items: center; gap: 6px; color: #9de8ff; }
.visual-live::before {
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: #38bdf8;
  box-shadow: 0 0 10px #38bdf8;
  content: "";
}
.scene-split {
  position: absolute;
  inset: 0;
  display: grid;
  grid-template-columns: 1fr 1fr;
  overflow: hidden;
  border-radius: inherit;
}
.scene-half {
  position: relative;
  overflow: hidden;
  background: linear-gradient(180deg, #10243a 0%, #23394a 49%, #111a28 100%);
}
.scene-half::before {
  position: absolute;
  z-index: 1;
  inset: 0;
  background: linear-gradient(180deg, rgba(5,12,25,.58), transparent 34%, rgba(6,12,22,.24));
  content: "";
}
.scene-half.underexposed { background: linear-gradient(180deg, #080d1b 0%, #131725 48%, #080d16 100%); }
.scene-half.underexposed::after {
  position: absolute;
  z-index: 2;
  inset: 0;
  background-image: radial-gradient(rgba(200,218,244,.33) .55px, transparent .9px);
  background-size: 4px 4px;
  opacity: .31;
  mix-blend-mode: screen;
  content: "";
}
.scene-art { position: absolute; inset: 0; width: 100%; height: 100%; }
.underexposed .scene-art { filter: brightness(.31) saturate(.43) contrast(1.2); }
.restored::before { background: linear-gradient(180deg, rgba(5,12,25,.24), transparent 34%, rgba(6,12,22,.12)); }
.restored .scene-art { filter: brightness(1.16) saturate(1.03) contrast(1); }
.scene-tag {
  position: absolute;
  z-index: 3;
  bottom: 12px;
  left: 12px;
  border: 1px solid rgba(203,213,225,.19);
  border-radius: 3px;
  background: rgba(5,11,22,.68);
  padding: 5px 7px;
  color: #d9e8f4;
  font-size: 7px;
  font-weight: 700;
  letter-spacing: .14em;
}
.scene-tag.dim { color: #99a6ba; }
.scan-boundary {
  position: absolute;
  z-index: 5;
  top: 0;
  bottom: 0;
  left: 50%;
  width: 2px;
  background: linear-gradient(180deg, transparent, #bff5ff 12%, #38bdf8 48%, #a5b4fc 83%, transparent);
  box-shadow: 0 0 10px 2px rgba(56,189,248,.65), 0 0 28px rgba(56,189,248,.32);
  transform: translateX(-50%);
}
.scan-boundary::before, .scan-boundary::after {
  position: absolute;
  left: 50%;
  width: 16px;
  height: 16px;
  border: 1px solid #a5f3fc;
  border-radius: 50%;
  background: rgba(7,15,29,.92);
  box-shadow: 0 0 15px rgba(56,189,248,.5);
  content: "";
  transform: translateX(-50%);
}
.scan-boundary::before { top: 31%; }
.scan-boundary::after { top: 65%; width: 7px; height: 7px; }
.scan-sweep {
  position: absolute;
  z-index: 3;
  top: -25%;
  left: 0;
  width: 100%;
  height: 24%;
  background: linear-gradient(180deg, transparent, rgba(56,189,248,.08), rgba(125,211,252,.19), transparent);
  animation: scene-scan 6s ease-in-out infinite;
  animation-play-state: paused;
}
.hero.in-view .scan-sweep { animation-play-state: running; }
.hud-chip {
  position: absolute;
  z-index: 6;
  border: 1px solid rgba(125,211,252,.25);
  border-radius: 5px;
  background: rgba(8,15,29,.88);
  padding: 8px 10px;
  color: #d9e8f4;
  box-shadow: 0 10px 25px rgba(0,0,0,.3), 0 0 18px rgba(56,189,248,.06);
  backdrop-filter: blur(8px);
}
.hud-chip strong {
  display: block;
  margin-bottom: 5px;
  color: #8bdffc;
  font-size: 7px;
  font-weight: 700;
  letter-spacing: .14em;
}
.hud-chip span { display: block; color: #8293aa; font-size: 7px; letter-spacing: .08em; }
.histogram-chip { top: 0; left: 0; width: 112px; }
.histogram-bars { display: flex; height: 25px; align-items: end; gap: 3px; margin: 4px 0 6px; }
.histogram-bars i { flex: 1; border-radius: 2px 2px 0 0; background: linear-gradient(180deg,#7dd3fc,#6366f1); opacity: .86; }
.histogram-bars i:nth-child(1) { height: 23%; }
.histogram-bars i:nth-child(2) { height: 37%; }
.histogram-bars i:nth-child(3) { height: 50%; }
.histogram-bars i:nth-child(4) { height: 71%; }
.histogram-bars i:nth-child(5) { height: 92%; }
.histogram-bars i:nth-child(6) { height: 66%; }
.histogram-bars i:nth-child(7) { height: 43%; }
.histogram-bars i:nth-child(8) { height: 61%; }
.edge-chip { right: 0; bottom: 18%; width: 94px; }
.edge-preview {
  position: relative;
  height: 37px;
  margin: 4px 0 5px;
  overflow: hidden;
  border: 1px solid rgba(148,163,184,.13);
  background:
    linear-gradient(130deg, transparent 39%, rgba(103,232,249,.85) 40% 41%, transparent 42%),
    linear-gradient(90deg, transparent 22%, rgba(129,140,248,.8) 23% 24%, transparent 25% 63%, rgba(103,232,249,.6) 64% 65%, transparent 66%),
    linear-gradient(0deg, #0b1425, #101c30);
}
.edge-preview::after {
  position: absolute;
  inset: 0;
  background: repeating-linear-gradient(0deg, transparent 0 5px, rgba(148,163,184,.12) 5px 6px);
  content: "";
}
.fusion-chip { right: 4%; top: 0; display: flex; align-items: center; gap: 7px; padding: 7px 9px; }
.fusion-orbit {
  display: grid;
  width: 22px;
  height: 22px;
  place-items: center;
  border: 1px solid rgba(56,189,248,.55);
  border-radius: 50%;
  color: #a5f3fc;
  font-size: 10px;
  box-shadow: 0 0 12px rgba(56,189,248,.22), inset 0 0 9px rgba(56,189,248,.12);
}
.fusion-chip strong { margin: 0; }
.visual-footer {
  position: absolute;
  right: 5%;
  bottom: 2%;
  left: 5%;
  display: flex;
  justify-content: space-between;
  color: #718198;
  font-size: 7px;
  letter-spacing: .16em;
  text-transform: uppercase;
}
.visual-footer b { color: #91dff6; font-weight: 600; }
.restoration-visual::before, .restoration-visual::after {
  position: absolute;
  z-index: 7;
  width: 22px;
  height: 22px;
  border-color: rgba(103,232,249,.8);
  content: "";
  pointer-events: none;
}
.restoration-visual::before { top: 5%; left: 3%; border-top: 1px solid; border-left: 1px solid; }
.restoration-visual::after { right: 3%; bottom: 10%; border-right: 1px solid; border-bottom: 1px solid; }
.visual-label {
  position: absolute;
  right: 7%;
  bottom: -1%;
  color: #64748b;
  font-size: 8px;
  letter-spacing: .18em;
  text-transform: uppercase;
}
.hero-index {
  position: absolute;
  right: clamp(24px, 6vw, 80px);
  bottom: 33px;
  display: flex;
  align-items: center;
  gap: 12px;
  color: #75839b;
  font-size: 10px;
  letter-spacing: .17em;
  text-transform: uppercase;
}
.hero-index::before { width: 40px; height: 1px; background: linear-gradient(90deg, var(--blue), transparent); content: ""; }
.technology {
  position: relative;
  padding: clamp(74px, 9vw, 124px) clamp(24px, 8vw, 120px) clamp(80px, 10vw, 140px);
  overflow: hidden;
  background:
    radial-gradient(ellipse at 12% 30%, rgba(129,140,248,.08), transparent 36%),
    linear-gradient(180deg, #0b1020, #080d1b);
}
.tech-heading {
  display: grid;
  grid-template-columns: minmax(0, .85fr) minmax(240px, .55fr);
  align-items: end;
  gap: 32px;
  margin: 0 auto 54px;
  max-width: 1160px;
}
.section-kicker {
  margin: 0 0 16px;
  color: var(--blue);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: .22em;
  text-transform: uppercase;
}
.tech-heading h2 {
  max-width: 630px;
  margin: 0;
  color: var(--white);
  font-size: clamp(34px, 5vw, 58px);
  font-weight: 500;
  letter-spacing: -.05em;
  line-height: 1.05;
}
.tech-intro { margin: 0 0 4px; color: var(--muted); font-size: 14px; line-height: 1.8; }
.tech-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 18px;
  max-width: 1160px;
  margin: 0 auto;
  perspective: 1200px;
}
.tech-card {
  position: relative;
  min-height: 278px;
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: linear-gradient(145deg, rgba(17,26,46,.94), rgba(10,16,32,.98));
  padding: 28px 27px 26px;
  transform: perspective(900px) rotateX(var(--card-y, 0deg)) rotateY(var(--card-x, 0deg)) translateY(var(--card-lift, 0px));
  transform-style: preserve-3d;
  transition: transform .28s ease, border-color .28s ease, box-shadow .28s ease, background .28s ease;
}
.tech-card::before {
  position: absolute;
  inset: 0;
  background: radial-gradient(circle at var(--glow-x, 50%) var(--glow-y, 50%), rgba(56,189,248,.14), transparent 47%);
  opacity: 0;
  content: "";
  transition: opacity .3s ease;
  pointer-events: none;
}
.tech-card:hover, .tech-card:focus-visible {
  --card-lift: -7px;
  border-color: rgba(56,189,248,.48);
  background: linear-gradient(145deg, rgba(19,34,58,.98), rgba(10,16,32,.98));
  box-shadow: 0 25px 55px rgba(0,0,0,.24), 0 0 32px rgba(56,189,248,.08);
}
.tech-card:hover::before, .tech-card:focus-visible::before { opacity: 1; }
.card-top { position: relative; display: flex; align-items: center; justify-content: space-between; }
.card-number { color: #6e7c96; font-size: 10px; letter-spacing: .18em; }
.card-icon {
  display: grid;
  width: 44px;
  height: 44px;
  place-items: center;
  border: 1px solid rgba(129,140,248,.24);
  border-radius: 50%;
  background: rgba(129,140,248,.08);
  color: #a5b4fc;
  font-size: 19px;
  box-shadow: 0 0 23px rgba(129,140,248,.07);
  transform: translateZ(24px);
}
.tech-card:nth-child(2) .card-icon { border-color: rgba(56,189,248,.25); background: rgba(56,189,248,.08); color: #7dd3fc; }
.tech-card:nth-child(3) .card-icon { border-color: rgba(167,139,250,.25); background: rgba(167,139,250,.08); color: #c4b5fd; }
.tech-card h3 {
  position: relative;
  margin: 35px 0 12px;
  color: var(--white);
  font-size: 20px;
  font-weight: 550;
  letter-spacing: -.025em;
  transform: translateZ(18px);
}
.tech-card p { position: relative; max-width: 300px; margin: 0; color: var(--muted); font-size: 13px; line-height: 1.75; transform: translateZ(12px); }
.card-rule { position: absolute; right: 27px; bottom: 25px; left: 27px; height: 1px; background: linear-gradient(90deg, rgba(56,189,248,.4), transparent 75%); }
.tech-footnote {
  max-width: 1160px;
  margin: 26px auto 0;
  color: #6e7c96;
  font-size: 11px;
  letter-spacing: .04em;
  line-height: 1.7;
}
.experience-footer {
  display: flex;
  max-width: 1160px;
  align-items: center;
  justify-content: space-between;
  gap: 22px;
  margin: 65px auto 0;
  border-top: 1px solid var(--line);
  padding-top: 24px;
}
.experience-footer p { margin: 0; color: var(--muted); font-size: 12px; }
.text-link { color: #a5b4fc; font-size: 11px; font-weight: 700; letter-spacing: .1em; text-decoration: none; text-transform: uppercase; }
.text-link:hover { color: #7dd3fc; }
.lumina-cursor-bridged, .lumina-cursor-bridged body, .lumina-cursor-bridged body * { cursor: none !important; }
.js-ready .reveal { opacity: 0; transform: translateY(24px); transition: opacity .8s ease, transform .8s cubic-bezier(.2,.7,.2,1); }
.js-ready .reveal.is-visible { opacity: 1; transform: translateY(0); }
.tech-card:nth-child(2) { transition-delay: .08s; }
.tech-card:nth-child(3) { transition-delay: .16s; }
@keyframes scene-scan { 0%, 10% { transform: translateY(-20%); opacity: 0; } 25%, 75% { opacity: 1; } 90%, 100% { transform: translateY(520%); opacity: 0; } }
@media (max-width: 760px) {
  .hero { min-height: 680px; height: clamp(680px, calc(var(--page-height) * .88), 800px); align-items: start; padding-top: 130px; }
  .hero-copy { width: min(100%, 590px); }
  .wordmark { font-size: clamp(58px, 16vw, 100px); }
  .hero h1 { max-width: 490px; font-size: clamp(36px, 9vw, 58px); }
  .hero-description { max-width: 420px; }
  .restoration-visual { top: auto; right: 2%; bottom: -22%; width: min(86%, 520px); transform: none; opacity: .88; }
  .hero-note { display: none; }
  .tech-heading { grid-template-columns: 1fr; gap: 18px; margin-bottom: 34px; }
  .tech-grid { grid-template-columns: 1fr; gap: 12px; }
  .tech-card { min-height: 220px; padding: 23px; }
  .tech-card h3 { margin-top: 24px; }
  .card-rule { right: 23px; left: 23px; }
  .experience-footer { align-items: flex-start; flex-direction: column; margin-top: 46px; }
}
@media (max-width: 420px) {
  .hero { padding-right: 20px; padding-left: 20px; }
  .topline { right: 20px; left: 20px; font-size: 8px; }
  .topline > span:last-child { max-width: 125px; text-align: right; line-height: 1.5; }
  .hero-actions { gap: 14px; }
  .explore { min-height: 47px; padding: 0 16px; font-size: 10px; }
  .technology { padding-right: 20px; padding-left: 20px; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior: auto !important; animation-duration: .01ms !important; animation-iteration-count: 1 !important; transition-duration: .01ms !important; }
  .js-ready .reveal { opacity: 1; transform: none; }
}
</style>
</head>
<body>
<main class="experience" id="lumina-experience">
  <section class="hero" id="lumina-hero" aria-labelledby="hero-title">
    <div class="hero-grid" aria-hidden="true"></div>
    <div class="topline">
      <span class="topline-mark"><i aria-hidden="true"></i> LUMINA / COMPUTER VISION</span>
      <span>Intelligent image restoration</span>
    </div>
    <div class="hero-copy reveal">
      <p class="eyebrow">An invitation to see differently</p>
      <p class="wordmark" aria-label="LUMINA">LUMINA</p>
      <h1 id="hero-title"><span>See Beyond</span><br>the Darkness.</h1>
      <p class="hero-description">
        An intelligent low-light restoration system combining classical computer
        vision, deep learning and hybrid intelligence.
      </p>
      <div class="hero-actions">
        <a class="explore magnetic" href="#restoration-workspace" target="_top">
          Explore the workspace <span class="explore-arrow" aria-hidden="true">→</span>
        </a>
        <span class="hero-note">Image intelligence / 01</span>
      </div>
    </div>
    <div class="restoration-visual" aria-hidden="true">
      <div class="visual-aura"></div>
      <div class="visual-frame">
        <div class="visual-topline"><span>LUMINA / VISION SYSTEMS</span><span class="visual-live">LIVE ANALYSIS</span></div>
        <div class="scene-split">
          <div class="scene-half underexposed">
            <svg class="scene-art" viewBox="0 0 300 220" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
              <defs><linearGradient id="road-dark" x2="0" y2="1"><stop stop-color="#273547"/><stop offset="1" stop-color="#101722"/></linearGradient></defs>
              <path fill="#192435" d="M0 0h91v133L0 169zm300 0h-92v133l92 36z"/>
              <path fill="#34485a" d="M0 25 63 10v94L0 130zm300 0-63-15v94l63 31z"/>
              <path fill="#101a29" d="M18 45h28v29H18zm0 42h28v25H18zm39-52h25v28H57zm0 41h25v27H57zm169-29h25v27h-25zm0 41h25v27h-25zm37-28h27v29h-27zm0 43h27v26h-27z"/>
              <path fill="url(#road-dark)" d="M91 117h117l92 103H0z"/>
              <path stroke="#64748b" stroke-opacity=".58" stroke-width="2" d="m150 119-5 25m-8 22-10 54m22-74 5 74"/>
              <path stroke="#fbbf24" stroke-opacity=".7" stroke-width="2" d="M14 137h17m-11 13h23m221-11h20m-28 15h25"/>
              <path fill="#111a28" d="M91 158q7-13 22-13h26q9 0 15 13l8 14v17H83v-17zm63 0q8-11 20-11h28q9 0 16 11l9 15v16h-79z"/>
              <path fill="#fbbf24" d="M88 166h10v4H88zm75 1h10v4h-10z"/>
              <circle cx="101" cy="188" r="6" fill="#080d16"/><circle cx="149" cy="188" r="6" fill="#080d16"/><circle cx="173" cy="188" r="6" fill="#080d16"/><circle cx="218" cy="188" r="6" fill="#080d16"/>
            </svg>
            <span class="scene-tag dim">INPUT / LOW LIGHT</span>
          </div>
          <div class="scene-half restored">
            <svg class="scene-art" viewBox="0 0 300 220" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
              <defs><linearGradient id="road-light" x2="0" y2="1"><stop stop-color="#425a68"/><stop offset="1" stop-color="#18232d"/></linearGradient></defs>
              <path fill="#25394b" d="M0 0h91v133L0 169zm300 0h-92v133l92 36z"/>
              <path fill="#607789" d="M0 25 63 10v94L0 130zm300 0-63-15v94l63 31z"/>
              <path fill="#1d3040" d="M18 45h28v29H18zm0 42h28v25H18zm39-52h25v28H57zm0 41h25v27H57zm169-29h25v27h-25zm0 41h25v27h-25zm37-28h27v29h-27zm0 43h27v26h-27z"/>
              <path fill="#eabf77" d="M21 49h22v18H21zm39-10h21v19H60zm169-1h21v18h-21zm40 11h20v19h-20z"/>
              <path fill="url(#road-light)" d="M91 117h117l92 103H0z"/>
              <path stroke="#bfccd0" stroke-opacity=".65" stroke-width="2" d="m150 119-5 25m-8 22-10 54m22-74 5 74"/>
              <path stroke="#ffd277" stroke-width="2" d="M14 137h17m-11 13h23m221-11h20m-28 15h25"/>
              <path fill="#263544" d="M91 158q7-13 22-13h26q9 0 15 13l8 14v17H83v-17zm63 0q8-11 20-11h28q9 0 16 11l9 15v16h-79z"/>
              <path fill="#ffe59a" d="M88 166h10v4H88zm75 1h10v4h-10z"/>
              <circle cx="101" cy="188" r="6" fill="#090e15"/><circle cx="149" cy="188" r="6" fill="#090e15"/><circle cx="173" cy="188" r="6" fill="#090e15"/><circle cx="218" cy="188" r="6" fill="#090e15"/>
            </svg>
            <span class="scene-tag">OUTPUT / RESTORED</span>
          </div>
          <div class="scan-sweep"></div>
          <div class="scan-boundary"></div>
        </div>
      </div>
      <div class="hud-chip histogram-chip"><strong>ANALYSIS / LUMA</strong><div class="histogram-bars"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div><span>TONAL DISTRIBUTION</span></div>
      <div class="hud-chip edge-chip"><strong>EDGE MAP</strong><div class="edge-preview"></div><span>STRUCTURAL DETAIL</span></div>
      <div class="hud-chip fusion-chip"><span class="fusion-orbit">✳</span><strong>FUSION<br><span>CV + NEURAL</span></strong></div>
      <div class="visual-footer"><span><b>ENHANCEMENT</b> &nbsp;·&nbsp; ANALYSIS</span><span>FUSION &nbsp;·&nbsp; <b>RESTORATION</b></span></div>
      <span class="visual-label">Low-light restoration / 01</span>
    </div>
    <div class="hero-index"><span>Scroll to explore</span><span aria-hidden="true">↓</span></div>
  </section>
  <section class="technology" id="lumina-technology" aria-labelledby="technology-title">
    <div class="tech-heading reveal">
      <div>
        <p class="section-kicker">Three perspectives / One image</p>
        <h2 id="technology-title">A clearer view, built in layers.</h2>
      </div>
      <p class="tech-intro">
        Each approach brings a different strength to low-light restoration.
        LUMINA brings them together in a workspace designed for careful analysis.
      </p>
    </div>
    <div class="tech-grid">
      <article class="tech-card reveal" tabindex="0">
        <div class="card-top"><span class="card-number">01 / CLASSICAL</span><span class="card-icon" aria-hidden="true">◉</span></div>
        <h3>Classical Computer Vision</h3>
        <p>Illumination-aware image analysis and foundational signal-processing techniques.</p>
        <div class="card-rule" aria-hidden="true"></div>
      </article>
      <article class="tech-card reveal" tabindex="0">
        <div class="card-top"><span class="card-number">02 / LEARNING</span><span class="card-icon" aria-hidden="true">✳</span></div>
        <h3>Deep Learning</h3>
        <p>Learned image-restoration approaches, explored with attention to detail and fidelity.</p>
        <div class="card-rule" aria-hidden="true"></div>
      </article>
      <article class="tech-card reveal" tabindex="0">
        <div class="card-top"><span class="card-number">03 / FUSION</span><span class="card-icon" aria-hidden="true">◇</span></div>
        <h3>Hybrid Intelligence</h3>
        <p>A considered combination of computer vision and learned restoration methods.</p>
        <div class="card-rule" aria-hidden="true"></div>
      </article>
    </div>
    <p class="tech-footnote">These are the project’s three approaches. Restoration outputs appear in the workspace when their modules are integrated.</p>
    <footer class="experience-footer">
      <p>Every image has more to reveal.</p>
      <a class="text-link magnetic" href="#restoration-workspace" target="_top">Enter restoration workspace&nbsp; →</a>
    </footer>
  </section>
</main>
<script>
(() => {
  const frame = window.frameElement;
  const page = window.parent;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const canHover = window.matchMedia("(hover: hover) and (pointer: fine)");
  const nativeCursorSelector = [
    'input:not([type="button"]):not([type="submit"]):not([type="reset"])',
    "textarea", "select", '[contenteditable="true"]', '[role="textbox"]',
    '[role="combobox"]', '[role="slider"]', '[role="checkbox"]',
    '[role="radio"]', '[role="switch"]', '[role="spinbutton"]'
  ].join(",");
  const interactiveSelector = [
    "a[href]", "button:not(:disabled)", '[role="button"]', ".tech-card",
    '[data-testid="stExpander"] summary'
  ].join(",");
  const createGlobalCursor = () => {
    try {
      const parentDocument = page.document;
      if (!parentDocument.body) return null;
      let controller = page.__luminaCursorController;
      if (controller) {
        parentDocument.getElementById("lumina-global-cursor-style") ||
          parentDocument.head.appendChild(Object.assign(parentDocument.createElement("style"), {
            id: "lumina-global-cursor-style",
            textContent: controller.styleText
          }));
        return controller;
      }

      const styleText = `
        @media (hover: hover) and (pointer: fine) {
          body[data-lumina-cursor-active="true"] .stApp,
          body[data-lumina-cursor-active="true"] .stApp * { cursor: none !important; }
          body[data-lumina-cursor-active="true"] .stApp input:not([type="button"]):not([type="submit"]):not([type="reset"]),
          body[data-lumina-cursor-active="true"] .stApp textarea,
          body[data-lumina-cursor-active="true"] .stApp select,
          body[data-lumina-cursor-active="true"] .stApp [contenteditable="true"],
          body[data-lumina-cursor-active="true"] .stApp [role="combobox"],
          body[data-lumina-cursor-active="true"] .stApp [role="slider"],
          body[data-lumina-cursor-active="true"] .stApp [role="checkbox"],
          body[data-lumina-cursor-active="true"] .stApp [role="radio"],
          body[data-lumina-cursor-active="true"] .stApp [role="switch"],
          body[data-lumina-cursor-active="true"] .stApp [role="spinbutton"] { cursor: auto !important; }
          body[data-lumina-cursor-active="true"] .stApp input:not([type="button"]):not([type="submit"]):not([type="reset"]),
          body[data-lumina-cursor-active="true"] .stApp textarea,
          body[data-lumina-cursor-active="true"] .stApp [contenteditable="true"],
          body[data-lumina-cursor-active="true"] .stApp [role="textbox"] { cursor: text !important; }
          #lumina-global-cursor-core, #lumina-global-cursor-ring {
            position: fixed; z-index: 2147483647; top: 0; left: 0; border-radius: 50%;
            opacity: 0; pointer-events: none; will-change: transform, opacity;
          }
          #lumina-global-cursor-core {
            width: 4px; height: 4px; background: #e0f2fe;
            box-shadow: 0 0 10px 3px rgba(56,189,248,.76);
            transform: translate3d(var(--cursor-x, -100px), var(--cursor-y, -100px), 0) translate(-50%, -50%);
          }
          #lumina-global-cursor-ring {
            width: 29px; height: 29px; border: 1px solid rgba(125,211,252,.56);
            background: rgba(56,189,248,.025);
            transform: translate3d(var(--cursor-x, -100px), var(--cursor-y, -100px), 0) translate(-50%, -50%);
            transition: width .22s ease, height .22s ease, border-color .22s ease,
              background .22s ease, opacity .16s ease, transform .11s cubic-bezier(.2,.7,.2,1);
          }
          #lumina-global-cursor-core.is-active, #lumina-global-cursor-ring.is-active { opacity: 1; }
          #lumina-global-cursor-ring.is-over {
            width: 47px; height: 47px; border-color: rgba(165,180,252,.85);
            background: rgba(129,140,248,.09);
          }
        }
        @media (prefers-reduced-motion: reduce) {
          #lumina-global-cursor-core, #lumina-global-cursor-ring { display: none !important; }
        }
      `;
      const style = parentDocument.createElement("style");
      style.id = "lumina-global-cursor-style";
      style.textContent = styleText;
      parentDocument.head.appendChild(style);

      const core = parentDocument.createElement("div");
      core.id = "lumina-global-cursor-core";
      core.setAttribute("aria-hidden", "true");
      const ring = parentDocument.createElement("div");
      ring.id = "lumina-global-cursor-ring";
      ring.setAttribute("aria-hidden", "true");
      parentDocument.body.append(core, ring);

      const canHover = page.matchMedia("(hover: hover) and (pointer: fine)");
      const reduceMotion = page.matchMedia("(prefers-reduced-motion: reduce)");
      const updateEnabled = () => {
        const enabled = canHover.matches && !reduceMotion.matches;
        if (enabled) {
          if (parentDocument.body.dataset.luminaCursorActive !== "true") {
            parentDocument.body.dataset.luminaCursorActive = "true";
          }
        } else if (parentDocument.body.hasAttribute("data-lumina-cursor-active")) {
          delete parentDocument.body.dataset.luminaCursorActive;
          core.classList.remove("is-active");
          ring.classList.remove("is-active", "is-over");
        }
        return enabled;
      };
      const move = (x, y, target, embedded = false) => {
        if (!updateEnabled()) return;
        const element = target?.nodeType === 1 ? target : target?.parentElement;
        if ((!embedded && !element?.closest(".stApp")) || element?.closest(nativeCursorSelector)) {
          core.classList.remove("is-active");
          ring.classList.remove("is-active", "is-over");
          return;
        }
        const position = `translate3d(${x}px, ${y}px, 0) translate(-50%, -50%)`;
        core.style.transform = position;
        ring.style.transform = position;
        core.classList.add("is-active");
        ring.classList.add("is-active");
        ring.classList.toggle("is-over", Boolean(element?.closest(interactiveSelector)));
      };
      const hide = () => {
        core.classList.remove("is-active");
        ring.classList.remove("is-active", "is-over");
      };
      controller = { move, hide, styleText };
      page.__luminaCursorController = controller;
      parentDocument.addEventListener("pointermove", event => move(event.clientX, event.clientY, event.target), { passive: true });
      parentDocument.addEventListener("pointerout", event => {
        if (!event.relatedTarget) hide();
      }, { passive: true });
      canHover.addEventListener?.("change", updateEnabled);
      reduceMotion.addEventListener?.("change", updateEnabled);
      updateEnabled();
      return controller;
    } catch (error) {
      console.warn(
        "[LUMINA] Parent-page access is unavailable; the global custom cursor is disabled.",
        error
      );
      return null;
    }
  };
  const globalCursor = createGlobalCursor();
  const syncIframeCursor = () => {
    document.documentElement.classList.toggle(
      "lumina-cursor-bridged",
      Boolean(globalCursor) && canHover.matches && !reduceMotion.matches
    );
  };
  syncIframeCursor();
  canHover.addEventListener?.("change", syncIframeCursor);
  reduceMotion.addEventListener?.("change", syncIframeCursor);

  const experience = document.getElementById("lumina-experience");
  const hero = document.getElementById("lumina-hero");
  const tech = document.getElementById("lumina-technology");
  const visual = document.querySelector(".restoration-visual");
  let scrollContainer = page;
  try {
    scrollContainer = page.document.querySelector('[data-testid="stMain"]') || page;
  } catch (_) {
    // Keep component-local interactions when the parent document is isolated.
  }
  let scheduled = false;

  const setFrameSize = () => {
    try {
      const viewport = page.innerHeight || 820;
      document.documentElement.style.setProperty("--page-height", `${viewport}px`);
      if (frame) frame.style.height = `${Math.ceil(experience.scrollHeight)}px`;
    } catch (_) {
      document.documentElement.style.setProperty("--page-height", "820px");
    }
  };

  const revealInView = () => {
    try {
      const frameRect = frame.getBoundingClientRect();
      const viewport = page.innerHeight || 820;
      for (const section of [hero, tech]) {
        const sectionTop = frameRect.top + section.offsetTop;
        const visible = sectionTop < viewport * .88 && sectionTop + section.offsetHeight > viewport * .08;
        section.classList.toggle("in-view", visible);
        if (visible) section.querySelectorAll(".reveal").forEach((item, index) => {
          window.setTimeout(() => item.classList.add("is-visible"), index * 90);
        });
      }
      const heroRect = hero.getBoundingClientRect();
      const progress = Math.max(0, Math.min(1, (viewport - heroRect.bottom) / Math.max(hero.offsetHeight, 1)));
      visual.style.setProperty("--scroll-scale", String(1 - progress * .035));
      visual.style.opacity = String(Math.max(.45, 1 - progress * .45));
      if (heroRect.bottom < 0) visual.style.opacity = ".4";
    } catch (_) {
      hero.querySelectorAll(".reveal").forEach(item => item.classList.add("is-visible"));
      tech.querySelectorAll(".reveal").forEach(item => item.classList.add("is-visible"));
    }
  };

  const onScroll = () => {
    if (scheduled) return;
    scheduled = true;
    page.requestAnimationFrame(() => { revealInView(); scheduled = false; });
  };

  const onPointerMove = event => {
    if (!canHover.matches || reduceMotion.matches) return;
    const frameRect = frame.getBoundingClientRect();
    globalCursor?.move(frameRect.left + event.clientX, frameRect.top + event.clientY, event.target, true);
    const bounds = hero.getBoundingClientRect();
    if (event.clientY < bounds.bottom && event.clientY > bounds.top) {
      const x = (event.clientX / Math.max(window.innerWidth, 1) - .5) * 8;
      const y = (.5 - event.clientY / Math.max(window.innerHeight, 1)) * 8;
      visual.style.setProperty("--tilt-x", `${x}deg`);
      visual.style.setProperty("--tilt-y", `${y}deg`);
    }
  };
  const onPointerLeave = () => {
    globalCursor?.hide();
    visual.style.setProperty("--tilt-x", "0deg");
    visual.style.setProperty("--tilt-y", "0deg");
  };

  if (!reduceMotion.matches) {
    document.documentElement.classList.add("js-ready");
    experience.addEventListener("pointermove", onPointerMove, { passive: true });
    experience.addEventListener("pointerleave", onPointerLeave, { passive: true });
    experience.querySelectorAll(".tech-card").forEach(card => {
      card.addEventListener("pointermove", event => {
        if (!canHover.matches) return;
        const rect = card.getBoundingClientRect();
        const x = (event.clientX - rect.left) / rect.width;
        const y = (event.clientY - rect.top) / rect.height;
        card.style.setProperty("--card-x", `${(x - .5) * 8}deg`);
        card.style.setProperty("--card-y", `${(.5 - y) * 8}deg`);
        card.style.setProperty("--glow-x", `${x * 100}%`);
        card.style.setProperty("--glow-y", `${y * 100}%`);
      }, { passive: true });
      card.addEventListener("pointerleave", () => {
        card.style.setProperty("--card-x", "0deg");
        card.style.setProperty("--card-y", "0deg");
      });
    });
  } else {
    experience.querySelectorAll(".reveal").forEach(item => item.classList.add("is-visible"));
  }

  experience.querySelectorAll('a[href="#restoration-workspace"]').forEach(link => {
    link.addEventListener("click", event => {
      try {
        const target = page.document.getElementById("restoration-workspace");
        if (target) {
          event.preventDefault();
          target.scrollIntoView({ behavior: reduceMotion.matches ? "auto" : "smooth", block: "start" });
        }
      } catch (_) {
        // The top-level hash link remains the fallback when the parent is isolated.
      }
    });
  });

  setFrameSize();
  revealInView();
  scrollContainer.addEventListener("scroll", onScroll, { passive: true });
  page.addEventListener("resize", () => { setFrameSize(); onScroll(); }, { passive: true });
  reduceMotion.addEventListener?.("change", event => {
    if (event.matches) onPointerLeave();
  });
})();
</script>
</body>
</html>
"""


def render_immersive_experience() -> None:
    """Render the visual introduction without replacing the Streamlit workspace."""
    st.components.v1.html(
        _EXPERIENCE_HTML,
        height=1600,
        scrolling=False,
    )
