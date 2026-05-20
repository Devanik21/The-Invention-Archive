--
session_id: IA-2026-140-T6
date: 2026-05-20
topic: Spectral Encoding Capacity
seed: 20260520
N_signal: 1024
fs_hz: 1000.0
n_components: 5
---

# Invention Archive — Daily Session 2026-05-20

**Session ID:** `IA-2026-140-T6`
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
plus Gaussian noise ($\sigma_n = 0.0617$):

$$x(t) = \sum_{k=1}^{5} A_k \cos(2\pi f_k t + \phi_k) + \eta(t)$$

True components: $f \in \{7.68, 33.29, 54.00, 72.62, 146.17\}$ Hz,
$A \in \{0.633, 0.593, 0.576, 0.472, 0.389\}$.

---

## 3. SpectraNova FFT Decomposition

Frequency resolution: $\Delta f = f_s / N = 0.977$ Hz.

### 3.1 Component Recovery

| $f_{\rm true}$ (Hz) | $A_{\rm true}$ | $f_{\rm det}$ (Hz) | $A_{\rm det}$ | $|f_{\rm err}|$ (Hz) | $C_k$ (bits) |
|---:|---:|---:|---:|---:|---:|
| 7.68 | 0.6328 | 7.81 | 0.6163 | 0.136 | 15.640 |
| 33.29 | 0.5931 | 33.20 | 0.5802 | 0.090 | 15.466 |
| 54.00 | 0.5764 | 53.71 | 0.4983 | 0.286 | 15.027 |
| 72.62 | 0.4723 | 72.27 | 0.3699 | 0.350 | 14.167 |
| 146.17 | 0.3886 | 146.48 | 0.3232 | 0.316 | 13.778 |

### 3.2 System-Level Statistics

| Metric | Value |
|---|---|
| Total signal SNR | 22.82 dB |
| Total FRAE encoding capacity $\sum_k C_k$ | **74.0782 bits** |
| Spectral flatness (Wiener entropy proxy) | 0.013785 |
| Participation ratio (effective components) | 6.05 |
| Noise floor $\sigma_n$ | 0.06171 |

---

## 4. Shannon-Hartley Per-Component Capacity

For each detected component with amplitude $A_k$ in additive white noise
of variance $\sigma_n^2$, the per-component encoding capacity is:

$$C_k = \log_2\!\left(1 + \frac{A_k^2/2}{\sigma_n^2/N}\right) \text{ bits}$$

Total capacity across 5 matched components:
$C_{\rm total} = 74.0782$ bits.

---

## 5. Spectral Flatness

The **Wiener entropy** (spectral flatness measure):

$$\mathrm{SFM} = \frac{\exp\bigl(\langle \ln S(f) \rangle\bigr)}{\langle S(f) \rangle}
  = 0.013785$$

$\mathrm{SFM} \to 1$: white noise (maximally flat).
$\mathrm{SFM} \to 0$: tonal / highly structured signal.
The value $0.0138$ indicates a
highly structured signal with clear tonal components.

---
*IA-2026-140-T6 · 2026-05-20 · seed 20260520*
