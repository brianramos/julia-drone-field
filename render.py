#!/usr/bin/env python3
"""
Julia Drone Field — offline arrangement renderer.

Parses an arrangement exported from the app ("Export arrangement" -> *.json,
kind == "julia-drone-arrangement") and re-renders it to a lossless FLAC by
reproducing the exact same signal path the app uses:

  * the deterministic score derivation (mulberry32 / xmur3 RNG, buildScore,
    orbit features, scale voicing) — bit-for-bit the same note choices;
  * the beat-quantized scene MUTATION (single evolving system, bar-stepped
    field interpolation, midpoint key pivot) that the live engine plays;
  * the voice set (sine drone, triangle haze, sub-bass, click-free sine melody,
    pink-noise wash) with the app's envelopes;
  * the effect chain (LFO-swept low-pass, ping-pong delay, convolution reverb,
    dry/wet blend, master + limiter).

Everything runs in numpy/scipy, so the output has none of the browser
OfflineAudioContext artifacts. Deterministic: same JSON -> same audio.

Usage:
    python3 render_arrangement.py arrangement.json [output.flac] [--sr 48000]
"""

import sys, json, math, argparse
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, oaconvolve

# ---------------------------------------------------------------------------
# Deterministic RNG — exact ports of the app's xmur3 / mulberry32 (uint32 math)
# ---------------------------------------------------------------------------
M32 = 0xFFFFFFFF
def _imul(a, b):
    return ((a & M32) * (b & M32)) & M32

def xmur3(s):
    s = str(s)
    h = (1779033703 ^ len(s)) & M32
    for ch in s:
        h = _imul(h ^ ord(ch), 3432918353)
        h = ((h << 13) | (h >> 19)) & M32
    state = {'h': h}
    def gen():
        h = state['h']
        h = _imul(h ^ (h >> 16), 2246822507)
        h = _imul(h ^ (h >> 13), 3266489909)
        h = (h ^ (h >> 16)) & M32
        state['h'] = h
        return h
    return gen

def hash_seed(text):
    return xmur3(text)()

def mulberry32(a):
    state = {'a': a & M32}
    def gen():
        state['a'] = (state['a'] + 0x6D2B79F5) & M32
        t = state['a']
        t = _imul(t ^ (t >> 15), t | 1)
        t = (t ^ ((t + _imul(t ^ (t >> 7), t | 61)) & M32)) & M32
        return ((t ^ (t >> 14)) & M32) / 4294967296.0
    return gen

# ---------------------------------------------------------------------------
# Small helpers (match the JS semantics, incl. Math.round = floor(x+0.5))
# ---------------------------------------------------------------------------
def clamp(v, lo, hi): return max(lo, min(hi, v))
def lerp(a, b, t): return a + (b - a) * t
def jsmod(n, m): return ((n % m) + m) % m
def jsround(x): return math.floor(x + 0.5)
def pick(rng, lst): return lst[math.floor(rng() * len(lst))]
def rand(rng, lo, hi): return lo + (hi - lo) * rng()
def chance(rng, p): return rng() < p
def db2gain(db): return 10.0 ** (db / 20.0)

NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
SCALES = {
    "aeolian": [0,2,3,5,7,8,10], "dorian": [0,2,3,5,7,9,10], "lydian": [0,2,4,6,7,9,11],
    "phrygian": [0,1,3,5,7,8,10], "pentatonic": [0,3,5,7,10], "whole": [0,2,4,6,8,10],
    "harmonicMinor": [0,2,3,5,7,8,11],
}
TRACK_KEYS = ["drone","haze","bass","melody","noise"]
TRACK_DEFS = {
    "drone":  {"level":0,"duration":1,"fadeIn":5.5,"fadeOut":9},
    "haze":   {"level":0,"duration":1,"fadeIn":2.5,"fadeOut":10},
    "bass":   {"level":0,"duration":1,"fadeIn":1.5,"fadeOut":6},
    "melody": {"level":0,"duration":1,"fadeIn":0.06,"fadeOut":4.5},
    "noise":  {"level":0,"duration":1,"fadeIn":1.8,"fadeOut":3},
}
ENV_FLOORS = {
    "drone": (.04,.10), "haze": (.04,.10), "bass": (.035,.12),
    "melody": (.09,.12), "noise": (.12,.16),
}

def scale_midi(root_pc, scale, degree, octave):
    n = len(scale)
    oct_shift = math.floor(degree / n)
    idx = jsmod(degree, n)
    return 12 * (octave + 1 + oct_shift) + root_pc + scale[idx]

def note_to_midi(note):
    if isinstance(note, (int, float)):
        return int(note)
    s = str(note).strip()
    i = 1
    if len(s) > 1 and s[1] == '#':
        i = 2
    name = s[:i]
    try:
        octave = int(s[i:])
    except ValueError:
        return 60
    return NOTE_NAMES.index(name) + 12 * (octave + 1)

def midi_to_freq(m): return 440.0 * (2.0 ** ((m - 69) / 12.0))

def safe_melody_freq(midi):
    hz = midi_to_freq(midi)
    if not math.isfinite(hz) or hz <= 0: return 440.0
    while hz > 1400: hz *= 0.5
    while hz < 110: hz *= 2.0
    return clamp(hz, 110, 1400)

def notation_seconds(value, bpm):
    try:
        return max(.001, float(value))
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    quarter = 60.0 / clamp(float(bpm) if bpm else 72, 20, 240)
    import re
    m = re.match(r'^(\d+(?:\.\d+)?)m$', text, re.I)
    if m: return float(m.group(1)) * 4 * quarter
    m = re.match(r'^(\d+(?:\.\d+)?)n(\.)?$', text, re.I)
    if m: return (4.0 / float(m.group(1))) * quarter * (1.5 if m.group(2) else 1.0)
    m = re.match(r'^(\d+(?:\.\d+)?)t$', text, re.I)
    if m: return (4.0 / float(m.group(1))) * quarter * (2.0 / 3.0)
    try: return max(.001, float(text))
    except ValueError: return quarter

def safe_velocity(value, dyn_min):
    return clamp(max(float(value) if value else 0.0, db2gain(clamp(dyn_min, -60, -3))), .0001, 1.0)

def sanitize_mixer(payload):
    out = {}
    payload = payload or {}
    for k in TRACK_KEYS:
        d = TRACK_DEFS[k]; src = payload.get(k) or {}
        out[k] = {
            "level": clamp(float(src.get("level", d["level"])), -36, 6),
            "duration": clamp(float(src.get("duration", d["duration"])), .25, 3),
            "fadeIn": clamp(float(src.get("fadeIn", d["fadeIn"])), .01, 12),
            "fadeOut": clamp(float(src.get("fadeOut", d["fadeOut"])), .05, 16),
            "mute": bool(src.get("mute", False)),
            "solo": bool(src.get("solo", False)),
        }
    return out

def stabilized_env(track, mixer):
    fa, fr = ENV_FLOORS.get(track, (.025, .08))
    src = mixer.get(track) or TRACK_DEFS[track]
    return (max(fa, float(src.get("fadeIn", fa)) or fa),
            max(fr, float(src.get("fadeOut", fr)) or fr))

# ---------------------------------------------------------------------------
# Score derivation — exact port of buildScore / orbitFeatures / interpolation
# ---------------------------------------------------------------------------
def orbit_features(c, count, seed_phase=0.0):
    zr = math.cos(seed_phase) * .001
    zi = math.sin(seed_phase) * .001
    out = []
    for i in range(count):
        nr = zr * zr - zi * zi + c["re"]
        zi = 2 * zr * zi + c["im"]
        zr = nr
        mag = math.hypot(zr, zi)
        out.append({"angle": math.atan2(zi, zr), "mag": mag})
        if mag > 8 or not math.isfinite(mag):
            zr = math.sin(i * 1.73) * .01
            zi = math.cos(i * .91) * .01
    return out

def build_score(systems, root_pc, scale_name, density, rng):
    scale = SCALES.get(scale_name, SCALES["aeolian"])
    melody = []
    prior = math.floor(rng() * len(scale))
    for i in range(32):
        sys = systems[i % len(systems)]
        f = orbit_features(sys, 5, i * .31 + sys["angle"])[4]
        angular = (f["angle"] + math.pi) / (2 * math.pi)
        raw = math.floor(angular * len(scale) * 2) - math.floor(len(scale) / 2)
        cont = clamp(raw, prior - 2, prior + 2) if chance(rng, .72) else raw
        prior = cont
        register = 5 if (i % 8 == 7 and chance(rng, .35)) else 4
        midi = scale_midi(root_pc, scale, cont, register)
        accent = 1 if i % 8 == 0 else (.82 if i % 4 == 0 else .62)
        probability = clamp(density * (0.62 + sys["noteEnergy"] * .45) * (.86 if (i % 2) else 1.12), .04, .98)
        active = chance(rng, probability)
        velocity = clamp(accent * rand(rng, .45, .84), .18, .92)
        duration = pick(rng, (["4n","2n","2n."] if chance(rng, .22) else ["8n","4n","4n."]))
        melody.append({"midi": midi, "active": active, "velocity": velocity,
                       "duration": duration, "systemIndex": i % len(systems)})
    chord_count = int(clamp(3 + jsround(rng() * 2), 3, 4))
    chord_degrees = [0, 2, 4, 6][:chord_count]
    drone = [scale_midi(root_pc, scale, d, 1 + (1 if idx > 1 else 0)) for idx, d in enumerate(chord_degrees)]
    upper = [scale_midi(root_pc, scale, d, 3 + (1 if idx == 2 else 0)) for idx, d in enumerate([0, 4, 8])]
    bass = scale_midi(root_pc, scale, 0, 1)
    alt_bass = scale_midi(root_pc, scale, (3 if chance(rng, .5) else 4), 1)
    return {"melody": melody, "drone": drone, "upperDrone": upper, "bass": bass, "altBass": alt_bass}

def interpolate_deep(a, b, t):
    an = isinstance(a, (int, float)) and not isinstance(a, bool)
    bn = isinstance(b, (int, float)) and not isinstance(b, bool)
    if an and bn:
        return lerp(a, b, t)
    if isinstance(a, dict) and isinstance(b, dict):
        out = {}
        for k in set(list(a.keys()) + list(b.keys())):
            out[k] = interpolate_deep(a.get(k), b.get(k), t)
        return out
    return a if t < .5 else b

def interpolate_systems(from_sys, to_sys, t):
    if not from_sys: return [dict(s) for s in (to_sys or [])]
    if not to_sys: return [dict(s) for s in (from_sys or [])]
    count = max(len(from_sys), len(to_sys))
    return [interpolate_deep(from_sys[i % len(from_sys)], to_sys[i % len(to_sys)], t) for i in range(count)]

def normalize_comp(comp):
    if not comp:
        return {"drone": [], "upperDrone": [], "bass": None, "altBass": None, "melody": []}
    tm = note_to_midi
    return {
        "drone": [tm(x) for x in (comp.get("drone") or [])],
        "upperDrone": [tm(x) for x in (comp.get("upperDrone") or [])],
        "bass": tm(comp["bass"]) if comp.get("bass") is not None else None,
        "altBass": tm(comp["altBass"]) if comp.get("altBass") is not None else None,
        "melody": [{"midi": int(m["midi"]), "active": bool(m["active"]),
                    "velocity": float(m["velocity"]), "duration": m["duration"]}
                   for m in (comp.get("melody") or [])],
    }

# ---------------------------------------------------------------------------
# Bar timeline — mirrors buildMutationTimeline()
# ---------------------------------------------------------------------------
def build_timeline(arr):
    a = arr.get("arrangement") or {}
    scenes = a.get("scenes", [])
    loop = bool(a.get("loop"))
    if not scenes:
        # Fall back to a single-settings export ("Export JSON") -> one static scene.
        if arr.get("composition"):
            scenes = [{"name": arr.get("name", "Scene 1"), "duration": 120, "transition": 0,
                       "settings": {"controls": arr.get("controls", {}), "mixer": arr.get("mixer"),
                                    "systems": arr.get("systems", []), "composition": arr.get("composition")}}]
            loop = False
        else:
            return None

    def scene_data(i):
        sc = scenes[i]; st = sc.get("settings", {})
        return {
            "name": sc.get("name") or f"Scene {i+1}",
            "duration": float(sc.get("duration", 60)),
            "transition": float(sc.get("transition", 0)),
            "comp": st.get("composition"), "systems": st.get("systems") or [],
            "controls": st.get("controls") or {}, "mixer": sanitize_mixer(st.get("mixer")),
            "bpm": clamp(float(st.get("controls", {}).get("tempo", 72)), 20, 240),
            "dynMin": clamp(float(st.get("controls", {}).get("dynamicMin", -36)), -60, -3),
            "density": clamp(float(st.get("controls", {}).get("density", .44)), 0, 1),
        }

    def sec_per_bar(bpm): return 4 * 60 / clamp(bpm, 20, 240)
    bars = []

    def push_hold(d):
        hold_bars = int(clamp(jsround(d["duration"] / sec_per_bar(d["bpm"])), 1, 512))
        for _ in range(hold_bars):
            bars.append({"comp": normalize_comp(d["comp"]), "controls": d["controls"],
                         "mixer": d["mixer"], "bpm": d["bpm"], "dynMin": d["dynMin"]})

    def interp_controls(ca, cb, t):
        out = {}
        for k in set(list(ca.keys()) + list(cb.keys())):
            try:
                av = float(ca.get(k)); bv = float(cb.get(k)); out[k] = lerp(av, bv, t)
            except (TypeError, ValueError):
                out[k] = ca.get(k) if t < .5 else cb.get(k)
        return out

    def interp_mixer(ma, mb, t):
        out = sanitize_mixer(None)
        for k in TRACK_KEYS:
            for f in ("level", "duration", "fadeIn", "fadeOut"):
                out[k][f] = lerp(ma[k][f], mb[k][f], t)
            out[k]["mute"] = ma[k]["mute"] if t < .5 else mb[k]["mute"]
            out[k]["solo"] = ma[k]["solo"] if t < .5 else mb[k]["solo"]
        return out

    def push_transition(ai, bi):
        A, B = scene_data(ai), scene_data(bi)
        tr = max(0.0, A["transition"])
        if tr <= .01: return
        steps = int(clamp(jsround(tr / sec_per_bar((A["bpm"] + B["bpm"]) / 2)), 1, 64))
        rng_seed = hash_seed(f'{A["name"]}:{B["name"]}:{ai}:{bi}:mutate')
        bars.append({"comp": normalize_comp(A["comp"]), "controls": A["controls"],
                     "mixer": A["mixer"], "bpm": A["bpm"], "dynMin": A["dynMin"]})  # bar 0 keeps A
        for s in range(1, steps):
            t = s / steps
            systems = interpolate_systems(A["systems"], B["systems"], t)
            ac = A["comp"] or {}; bc = B["comp"] or {}
            root_raw = ac.get("rootPc") if t < .5 else bc.get("rootPc")
            try: root_pc = jsround(float(root_raw))
            except (TypeError, ValueError): root_pc = 0
            scale_name = (ac.get("scaleName") if t < .5 else bc.get("scaleName")) \
                or ac.get("scaleName") or bc.get("scaleName") or "aeolian"
            score = build_score(systems, root_pc, scale_name, lerp(A["density"], B["density"], t), mulberry32(rng_seed))
            bars.append({"comp": normalize_comp(score), "controls": interp_controls(A["controls"], B["controls"], t),
                         "mixer": interp_mixer(A["mixer"], B["mixer"], t),
                         "bpm": clamp(lerp(A["bpm"], B["bpm"], t), 20, 240), "dynMin": lerp(A["dynMin"], B["dynMin"], t)})

    N = len(scenes)
    for i in range(N):
        push_hold(scene_data(i))
        if i < N - 1: push_transition(i, i + 1)
    if loop and N >= 1: push_transition(N - 1, 0)

    total_sec = sum(8 * notation_seconds("8n", bar["bpm"]) for bar in bars)
    all_fade = [sanitize_mixer(sc.get("settings", {}).get("mixer"))[k]["fadeOut"] for sc in scenes for k in TRACK_KEYS]
    tail = clamp(max([6] + all_fade), 6, 16)
    return {"bars": bars, "total_sec": total_sec, "tail": tail}

# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------
def adsr_env(dur, attack, decay, sustain, release, sr):
    end = max(dur, attack + decay)
    tb = [0.0, attack, attack + decay, end, end + release]
    vb = [0.0, 1.0, sustain, sustain, 0.0]
    total = int(round((end + release) * sr)) + 1
    ts = np.arange(total) / sr
    return np.interp(ts, tb, vb).astype(np.float64)

def melody_env(hold, attack0, release0, peak, sr):
    attack = min(max(.12, attack0), max(.12, hold * .72))
    release = clamp(release0, .18, 6.0)
    attack_end = attack
    decay_end = min(hold, attack_end + max(.04, min(.45, hold * .2)))
    release_start = max(hold, decay_end + .02)
    end = release_start + release
    tb = [0.0, attack_end, decay_end, release_start, end]
    vb = [0.0, peak, peak * .14, peak * .14, 0.0]
    total = int(round(end * sr)) + 1
    ts = np.arange(total) / sr
    return np.interp(ts, tb, vb).astype(np.float64)

def osc_sine(freq, n, sr):
    return np.sin(2 * np.pi * freq * np.arange(n) / sr)

def osc_triangle(freq, n, sr):
    ph = 2 * np.pi * freq * np.arange(n) / sr
    return (2 / np.pi) * np.arcsin(np.sin(ph))

def add_note(buf, idx, wave, env, gain):
    n = len(env)
    if idx >= len(buf) or n <= 0: return
    end = min(len(buf), idx + n)
    buf[idx:end] += (wave[:end - idx] * env[:end - idx]) * gain

def make_pink(n, seed):
    rs = np.random.RandomState(seed & 0x7FFFFFFF)
    white = rs.standard_normal(n)
    # Paul Kellet economy pink filter
    b = np.zeros(7); out = np.empty(n)
    for i in range(n):
        w = white[i]
        b[0] = 0.99886 * b[0] + w * 0.0555179
        b[1] = 0.99332 * b[1] + w * 0.0750759
        b[2] = 0.96900 * b[2] + w * 0.1538520
        b[3] = 0.86650 * b[3] + w * 0.3104856
        b[4] = 0.55000 * b[4] + w * 0.5329522
        b[5] = -0.7616 * b[5] - w * 0.0168980
        out[i] = b[0]+b[1]+b[2]+b[3]+b[4]+b[5]+b[6]+w*0.5362
        b[6] = w * 0.115926
    return out * 0.11

def tv_lowpass(x, cutoff, sr, order=4, block=2048):
    """Time-varying low-pass: recompute the biquad per block, carry state."""
    out = np.empty_like(x); zi = None; last_fc = None; sos = None
    for start in range(0, len(x), block):
        end = min(len(x), start + block)
        fc = clamp(float(np.mean(cutoff[start:end])), 30.0, sr * 0.45)
        if last_fc is None or abs(fc - last_fc) / last_fc > 0.02:
            sos = butter(order, fc / (sr * 0.5), btype="low", output="sos")
            last_fc = fc
            if zi is None: zi = sosfilt_zi(sos) * x[start]
        y, zi = sosfilt(sos, x[start:end], zi=zi)
        out[start:end] = y
    return out

def fixed_lowpass(x, fc, sr, order=4):
    sos = butter(order, clamp(fc, 30, sr * 0.45) / (sr * 0.5), btype="low", output="sos")
    return sosfilt(sos, x)

def ping_pong(mono, delay_samps, feedback, sr):
    """Feed-forward ping-pong: echoes bounce R,L,R,... with geometric decay."""
    n = len(mono); L = np.zeros(n); R = np.zeros(n)
    k = 1
    while feedback ** k > 1e-4 and k < 64:
        shift = delay_samps * k
        if shift >= n: break
        amp = feedback ** k
        tgt = R if (k % 2 == 1) else L
        tgt[shift:] += amp * mono[:n - shift]
        k += 1
    return L, R

def make_reverb_ir(decay, predelay, sr, seed):
    n = int(decay * sr)
    rs = np.random.RandomState(seed & 0x7FFFFFFF)
    t = np.arange(n) / sr
    envelope = np.exp(-t * (6.9 / max(0.5, decay)))  # ~ -60 dB at `decay` seconds
    ir = rs.standard_normal(n) * envelope
    pre = int(predelay * sr)
    ir = np.concatenate([np.zeros(pre), ir])
    energy = np.sqrt(np.sum(ir ** 2)) + 1e-9
    return (ir / energy).astype(np.float64)

def bar_param_curve(bars, bar_starts, total_n, sr, fn, ramp=0.06):
    """Piecewise curve: hold each bar's value, quick linear ramp at bar starts."""
    tb = [0.0]; vb = [fn(bars[0])]
    for i, bar in enumerate(bars):
        t0 = bar_starts[i]
        tb.append(t0); vb.append(vb[-1])          # hold previous until bar start
        tb.append(t0 + ramp); vb.append(fn(bar))  # ramp to this bar's value
    tb.append(total_n / sr); vb.append(vb[-1])
    ts = np.arange(total_n) / sr
    return np.interp(ts, tb, vb).astype(np.float64)

def render(arr, sr=48000, verbose=True):
    tl = build_timeline(arr)
    if not tl or not tl["bars"]:
        raise SystemExit("No scenes found in arrangement JSON.")
    bars = tl["bars"]
    total_n = int(round((tl["total_sec"] + tl["tail"] + 0.35) * sr))
    if verbose:
        print(f"  {len(bars)} bars, {tl['total_sec']:.1f}s + {tl['tail']:.1f}s tail = {total_n/sr:.1f}s @ {sr} Hz")

    tracks = {k: np.zeros(total_n) for k in TRACK_KEYS}
    bar_starts = []
    # scene-level tempo for the (fixed) delay time, taken from the first bar
    delay_time = notation_seconds("8n.", bars[0]["bpm"])

    # ---- schedule notes (mirrors scheduleOfflineTimeline) ----
    time = 0.0; global_step = 0; drone_cycle = 0
    noise_rng = mulberry32(hash_seed("wav-noise"))
    for bar in bars:
        bar_starts.append(time)
        bpm = bar["bpm"]; eighth = notation_seconds("8n", bpm)
        mixer = bar["mixer"]; comp = bar["comp"]; dyn = bar["dynMin"]
        bright = clamp(float(bar["controls"].get("brightness", .45)), 0, 1)
        d_a, d_r = stabilized_env("drone", mixer); h_a, h_r = stabilized_env("haze", mixer)
        b_a, b_r = stabilized_env("bass", mixer); m_a, m_r = stabilized_env("melody", mixer)
        n_a, n_r = stabilized_env("noise", mixer)
        def dur_of(v, tr): return max(.02, notation_seconds(v, bpm) * mixer[tr]["duration"])
        for _ in range(8):
            idx = int(round(time * sr))
            if global_step % 32 == 0 and comp["drone"]:
                drone_cycle += 1
                midis = list(reversed(comp["drone"])) if (drone_cycle % 2) else comp["drone"]
                d = dur_of("6m", "drone"); env = adsr_env(d, d_a, 3, .78, d_r, sr)
                g = safe_velocity(.45, dyn) * db2gain(-11)
                for mi in midis:
                    add_note(tracks["drone"], idx, osc_sine(midi_to_freq(mi), len(env), sr), env, g)
            if global_step % 32 == 16 and comp["upperDrone"]:
                d = dur_of("3m", "haze"); env = adsr_env(d, h_a, 4, .3, h_r, sr)
                g = safe_velocity(.2, dyn) * db2gain(-19)
                for mi in comp["upperDrone"]:
                    add_note(tracks["haze"], idx, osc_triangle(midi_to_freq(mi), len(env), sr), env, g)
            if global_step % 16 == 0:
                mi = comp["bass"] if ((global_step / 16) % 2 < 1) else comp["altBass"]
                if mi is not None:
                    d = dur_of("2m", "bass"); env = adsr_env(d, b_a, 2, .55, b_r, sr)
                    g = safe_velocity(.45, dyn) * db2gain(-9)
                    add_note(tracks["bass"], idx, osc_sine(midi_to_freq(mi), len(env), sr), env, g)
            mel = comp["melody"]
            if mel:
                ms = mel[global_step % len(mel)]
                if ms["active"]:
                    vel = min(.72, safe_velocity(ms["velocity"] * (.58 + bright * .34), dyn))
                    peak = clamp(vel, 0, .72) * db2gain(-15)
                    hold = max(.18, dur_of(ms["duration"], "melody"))
                    env = melody_env(hold, m_a, m_r, peak, sr)
                    add_note(tracks["melody"], idx, osc_sine(safe_melody_freq(ms["midi"]), len(env), sr), env, 1.0)
            if global_step % 16 == 12 and noise_rng() < .34:
                d = dur_of("2n", "noise"); env = adsr_env(d, n_a, 3.5, 0.0, n_r, sr)
                g = safe_velocity(.09, dyn) * db2gain(-27)
                add_note(tracks["noise"], idx, make_pink(len(env), 1234 + global_step), env, g)
            global_step += 1
            time += eighth
    if verbose: print("  voices synthesized; applying mix + effects…")

    # ---- per-track gain automation (level + mute/solo, ramped per bar) ----
    for k in TRACK_KEYS:
        def gfn(bar, kk=k):
            mx = bar["mixer"]; anysolo = any(mx[t]["solo"] for t in TRACK_KEYS)
            silent = mx[kk]["mute"] or (anysolo and not mx[kk]["solo"])
            return 0.0 if silent else db2gain(mx[kk]["level"])
        tracks[k] *= bar_param_curve(bars, bar_starts, total_n, sr, gfn)

    # ---- control automation curves ----
    bright_c = bar_param_curve(bars, bar_starts, total_n, sr, lambda b: clamp(float(b["controls"].get("brightness", .45)), 0, 1))
    gravity_c = bar_param_curve(bars, bar_starts, total_n, sr, lambda b: clamp(float(b["controls"].get("gravity", .3)), 0, 1))
    reverb_c = bar_param_curve(bars, bar_starts, total_n, sr, lambda b: clamp(float(b["controls"].get("reverb", .4)), 0, .95))
    avg_gravity = float(np.mean(gravity_c)); avg_bright = float(np.mean(bright_c))

    # ---- main LFO-swept low-pass on drone+haze+bass ----
    tt = np.arange(total_n) / sr
    lfo = 0.5 - 0.5 * np.cos(2 * np.pi * 0.035 * tt)  # phase 180 -> starts at min
    lo_min = 150 + bright_c * 260; lo_max = 700 + bright_c * 4200
    cutoff = lo_min + (lo_max - lo_min) * lfo
    bus_low = tracks["drone"] + tracks["haze"] + tracks["bass"]
    bus_low = tv_lowpass(bus_low, cutoff, sr)

    # ---- melody: soft ceiling (-8 dB) then guard low-pass ----
    mel = tracks["melody"]; thr = db2gain(-8)
    mel = thr * np.tanh(mel / thr)
    bus_mel = fixed_lowpass(mel, 2800 + avg_bright * 1000, sr)

    # ---- noise guard low-pass ----
    bus_noise = fixed_lowpass(tracks["noise"], 5200, sr)

    # ---- ping-pong delay (input = filtered low bus + melody) ----
    delay_in = bus_low + bus_mel
    dsamp = max(1, int(round(delay_time * sr)))
    feedback = .16 + avg_gravity * .28
    ppL, ppR = ping_pong(delay_in, dsamp, feedback, sr)
    dwet = .08 + gravity_c * .27
    delay_L = (1 - dwet) * delay_in + dwet * ppL
    delay_R = (1 - dwet) * delay_in + dwet * ppR

    # ---- convolution reverb (input = delay + noise) ----
    rin_L = delay_L + bus_noise; rin_R = delay_R + bus_noise
    ir_L = make_reverb_ir(6.0, 0.08, sr, 101)
    ir_R = make_reverb_ir(6.0, 0.08, sr, 202)
    REV_GAIN = 0.9
    wet_L = oaconvolve(rin_L, ir_L)[:total_n] * REV_GAIN
    wet_R = oaconvolve(rin_R, ir_R)[:total_n] * REV_GAIN
    rev_L = (1 - reverb_c) * rin_L + reverb_c * wet_L
    rev_R = (1 - reverb_c) * rin_R + reverb_c * wet_R

    # ---- master: reverb + dry(0.48*filtered low bus), master gain, limiter ----
    dry = 0.48 * bus_low
    master_L = 0.7 * (rev_L + dry)
    master_R = 0.7 * (rev_R + dry)

    dyn_max = clamp(float(arr.get("controls", {}).get("dynamicMax", -6)), -18, 0)
    ceil = db2gain(dyn_max)
    master_L = ceil * np.tanh(master_L / ceil)
    master_R = ceil * np.tanh(master_R / ceil)

    stereo = np.stack([master_L, master_R], axis=1)
    peak = float(np.max(np.abs(stereo))) or 1.0
    if peak > 0.999:
        stereo *= 0.999 / peak
    return stereo


def main():
    ap = argparse.ArgumentParser(description="Render a Julia Drone Field arrangement JSON to lossless FLAC.")
    ap.add_argument("input", help="arrangement .json exported from the app")
    ap.add_argument("output", nargs="?", help="output .flac (default: <input>.flac)")
    ap.add_argument("--sr", type=int, default=48000, help="sample rate (default 48000)")
    args = ap.parse_args()

    with open(args.input) as f:
        arr = json.load(f)
    if arr.get("kind") not in (None, "julia-drone-arrangement") and "arrangement" not in arr:
        print("Warning: this does not look like an exported arrangement.", file=sys.stderr)

    out = args.output or (args.input.rsplit(".", 1)[0] + ".flac")
    name = (arr.get("name") or "arrangement")
    print(f'Rendering "{name}" -> {out}')
    audio = render(arr, sr=args.sr)

    import soundfile as sf
    sf.write(out, audio, args.sr, format="FLAC", subtype="PCM_24")
    dur = len(audio) / args.sr
    print(f"Done: {dur:.1f}s, {audio.shape[1]}ch, 24-bit FLAC, peak {np.max(np.abs(audio)):.3f}")


if __name__ == "__main__":
    main()
