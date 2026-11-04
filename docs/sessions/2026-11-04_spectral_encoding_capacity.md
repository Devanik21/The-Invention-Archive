--
session_id: IA-2026-308-T6
date: 2026-11-04
topic: Spectral Encoding Capacity
seed: 20261104
N_signal: 1024
fs_hz: 1000.0
n_components: 5
---

# Invention Archive — Daily Session 2026-11-04

**Session ID:** `IA-2026-308-T6`
**Topic:** Spectral Decomposition and Encoding Capacity: FFT Analysis, Per-Component SNR, and Shannon-Hartley Bounds

---

## 1. Constructs — Spectral Domain

- **SpectraNova**: Advanced spectral decomposition of complex signals
- **FRAE**: Frequency-Resonance Adaptive Encoder
- **AetherSPARC**: Signal Processing and Resonance Coding

---

## 2. Experimental Signal

A synthetic $N = 1024$-sample signal ($f_s = 1000$ Hz,
$\Delta t = 0.0010$ s) comprising 5 frequency components
plus Gaussian noise ($\sigma_n = 0.0657$):

$$x(t) = \sum_{k=1}^{5} A_k \cos(2\pi f_k t + \phi_k) + \eta(t)$$

True components: $f \in \{86.77, 107.21, 108.17, 128.26, 144.74\}$ Hz,
$A \in \{0.815, 0.783, 0.744, 0.374, 0.168\}$.

---

## 3. SpectraNova FFT Decomposition

Frequency resolution: $\Delta f = f_s / N = 0.977$ Hz.

### 3.1 Component Recovery

| $f_{\rm true}$ (Hz) | $A_{\rm true}$ | $f_{\rm det}$ (Hz) | $A_{\rm det}$ | $|f_{\rm err}|$ (Hz) | $C_k$ (bits) |
|---:|---:|---:|---:|---:|---:|
| 86.77 | 0.8151 | 86.91 | 0.8006 | 0.143 | 16.215 |
| 107.21 | 0.7827 | 107.42 | 0.7229 | 0.209 | 15.920 |
| 128.26 | 0.3737 | 127.93 | 0.3030 | 0.334 | 13.412 |
| 144.74 | 0.1677 | 144.53 | 0.1527 | 0.204 | 11.434 |

### 3.2 System-Level Statistics

| Metric | Value |
|---|---|
| Total signal SNR | 23.65 dB |
| Total FRAE encoding capacity $\sum_k C_k$ | **56.9804 bits** |
| Spectral flatness (Wiener entropy proxy) | 0.006188 |
| Participation ratio (effective components) | 4.31 |
| Noise floor $\sigma_n$ | 0.06569 |

---

## 4. Shannon-Hartley Per-Component Capacity

For each detected component with amplitude $A_k$ in additive white noise
of variance $\sigma_n^2$, the per-component encoding capacity is:

$$C_k = \log_2\!\left(1 + \frac{A_k^2/2}{\sigma_n^2/N}\right) \text{ bits}$$

Total capacity across 4 matched components:
$C_{\rm total} = 56.9804$ bits.

---

## 5. Spectral Flatness

The **Wiener entropy** (spectral flatness measure):

$$\mathrm{SFM} = \frac{\exp\bigl(\langle \ln S(f) \rangle\bigr)}{\langle S(f) \rangle}
  = 0.006188$$

$\mathrm{SFM} \to 1$: white noise (maximally flat).
$\mathrm{SFM} \to 0$: tonal / highly structured signal.
The value $0.0062$ indicates a
highly structured signal with clear tonal components.

---
*IA-2026-308-T6 · 2026-11-04 · seed 20261104*
