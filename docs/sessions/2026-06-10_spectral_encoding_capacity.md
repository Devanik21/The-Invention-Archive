--
session_id: IA-2026-161-T6
date: 2026-06-10
topic: Spectral Encoding Capacity
seed: 20260610
N_signal: 1024
fs_hz: 1000.0
n_components: 5
---

# Invention Archive — Daily Session 2026-06-10

**Session ID:** `IA-2026-161-T6`
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
plus Gaussian noise ($\sigma_n = 0.0653$):

$$x(t) = \sum_{k=1}^{5} A_k \cos(2\pi f_k t + \phi_k) + \eta(t)$$

True components: $f \in \{4.19, 47.81, 56.16, 100.35, 133.13\}$ Hz,
$A \in \{0.872, 0.556, 0.191, 0.141, 0.138\}$.

---

## 3. SpectraNova FFT Decomposition

Frequency resolution: $\Delta f = f_s / N = 0.977$ Hz.

### 3.1 Component Recovery

| $f_{\rm true}$ (Hz) | $A_{\rm true}$ | $f_{\rm det}$ (Hz) | $A_{\rm det}$ | $|f_{\rm err}|$ (Hz) | $C_k$ (bits) |
|---:|---:|---:|---:|---:|---:|
| 4.19 | 0.8725 | 3.91 | 0.7339 | 0.287 | 15.981 |
| 47.81 | 0.5564 | 47.85 | 0.5694 | 0.046 | 15.249 |
| 56.16 | 0.1912 | 55.66 | 0.1240 | 0.495 | 10.851 |
| 100.35 | 0.1413 | 100.59 | 0.1279 | 0.238 | 10.941 |
| 133.13 | 0.1383 | 132.81 | 0.1124 | 0.315 | 10.567 |

### 3.2 System-Level Statistics

| Metric | Value |
|---|---|
| Total signal SNR | 21.29 dB |
| Total FRAE encoding capacity $\sum_k C_k$ | **63.5893 bits** |
| Spectral flatness (Wiener entropy proxy) | 0.011479 |
| Participation ratio (effective components) | 3.22 |
| Noise floor $\sigma_n$ | 0.06529 |

---

## 4. Shannon-Hartley Per-Component Capacity

For each detected component with amplitude $A_k$ in additive white noise
of variance $\sigma_n^2$, the per-component encoding capacity is:

$$C_k = \log_2\!\left(1 + \frac{A_k^2/2}{\sigma_n^2/N}\right) \text{ bits}$$

Total capacity across 5 matched components:
$C_{\rm total} = 63.5893$ bits.

---

## 5. Spectral Flatness

The **Wiener entropy** (spectral flatness measure):

$$\mathrm{SFM} = \frac{\exp\bigl(\langle \ln S(f) \rangle\bigr)}{\langle S(f) \rangle}
  = 0.011479$$

$\mathrm{SFM} \to 1$: white noise (maximally flat).
$\mathrm{SFM} \to 0$: tonal / highly structured signal.
The value $0.0115$ indicates a
highly structured signal with clear tonal components.

---
*IA-2026-161-T6 · 2026-06-10 · seed 20260610*
