--
session_id: IA-2026-119-T6
date: 2026-04-29
topic: Spectral Encoding Capacity
seed: 20260429
N_signal: 1024
fs_hz: 1000.0
n_components: 5
---

# Invention Archive — Daily Session 2026-04-29

**Session ID:** `IA-2026-119-T6`
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
plus Gaussian noise ($\sigma_n = 0.0349$):

$$x(t) = \sum_{k=1}^{5} A_k \cos(2\pi f_k t + \phi_k) + \eta(t)$$

True components: $f \in \{10.03, 52.21, 62.31, 67.13, 130.57\}$ Hz,
$A \in \{0.741, 0.675, 0.610, 0.311, 0.109\}$.

---

## 3. SpectraNova FFT Decomposition

Frequency resolution: $\Delta f = f_s / N = 0.977$ Hz.

### 3.1 Component Recovery

| $f_{\rm true}$ (Hz) | $A_{\rm true}$ | $f_{\rm det}$ (Hz) | $A_{\rm det}$ | $|f_{\rm err}|$ (Hz) | $C_k$ (bits) |
|---:|---:|---:|---:|---:|---:|
| 10.03 | 0.7410 | 9.77 | 0.6564 | 0.263 | 17.466 |
| 52.21 | 0.6749 | 51.76 | 0.4569 | 0.452 | 16.421 |
| 62.31 | 0.6105 | 62.50 | 0.5581 | 0.189 | 16.998 |
| 67.13 | 0.3111 | 67.38 | 0.2815 | 0.256 | 15.024 |
| 130.57 | 0.1085 | 130.86 | 0.0953 | 0.290 | 11.898 |

### 3.2 System-Level Statistics

| Metric | Value |
|---|---|
| Total signal SNR | 27.85 dB |
| Total FRAE encoding capacity $\sum_k C_k$ | **77.8068 bits** |
| Spectral flatness (Wiener entropy proxy) | 0.004243 |
| Participation ratio (effective components) | 5.97 |
| Noise floor $\sigma_n$ | 0.03490 |

---

## 4. Shannon-Hartley Per-Component Capacity

For each detected component with amplitude $A_k$ in additive white noise
of variance $\sigma_n^2$, the per-component encoding capacity is:

$$C_k = \log_2\!\left(1 + \frac{A_k^2/2}{\sigma_n^2/N}\right) \text{ bits}$$

Total capacity across 5 matched components:
$C_{\rm total} = 77.8068$ bits.

---

## 5. Spectral Flatness

The **Wiener entropy** (spectral flatness measure):

$$\mathrm{SFM} = \frac{\exp\bigl(\langle \ln S(f) \rangle\bigr)}{\langle S(f) \rangle}
  = 0.004243$$

$\mathrm{SFM} \to 1$: white noise (maximally flat).
$\mathrm{SFM} \to 0$: tonal / highly structured signal.
The value $0.0042$ indicates a
highly structured signal with clear tonal components.

---
*IA-2026-119-T6 · 2026-04-29 · seed 20260429*
