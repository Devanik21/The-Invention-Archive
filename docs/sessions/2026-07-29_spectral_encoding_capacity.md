--
session_id: IA-2026-210-T6
date: 2026-07-29
topic: Spectral Encoding Capacity
seed: 20260729
N_signal: 1024
fs_hz: 1000.0
n_components: 5
---

# Invention Archive — Daily Session 2026-07-29

**Session ID:** `IA-2026-210-T6`
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
plus Gaussian noise ($\sigma_n = 0.0475$):

$$x(t) = \sum_{k=1}^{5} A_k \cos(2\pi f_k t + \phi_k) + \eta(t)$$

True components: $f \in \{3.64, 7.49, 36.52, 88.10, 102.42\}$ Hz,
$A \in \{0.881, 0.812, 0.478, 0.414, 0.222\}$.

---

## 3. SpectraNova FFT Decomposition

Frequency resolution: $\Delta f = f_s / N = 0.977$ Hz.

### 3.1 Component Recovery

| $f_{\rm true}$ (Hz) | $A_{\rm true}$ | $f_{\rm det}$ (Hz) | $A_{\rm det}$ | $|f_{\rm err}|$ (Hz) | $C_k$ (bits) |
|---:|---:|---:|---:|---:|---:|
| 3.64 | 0.8811 | 3.91 | 0.7961 | 0.266 | 17.134 |
| 7.49 | 0.8118 | 7.81 | 0.6854 | 0.320 | 16.702 |
| 36.52 | 0.4775 | 36.13 | 0.3521 | 0.390 | 14.780 |
| 88.10 | 0.4137 | 87.89 | 0.3768 | 0.212 | 14.976 |
| 102.42 | 0.2222 | 102.54 | 0.2224 | 0.120 | 13.454 |

### 3.2 System-Level Statistics

| Metric | Value |
|---|---|
| Total signal SNR | 26.21 dB |
| Total FRAE encoding capacity $\sum_k C_k$ | **77.0452 bits** |
| Spectral flatness (Wiener entropy proxy) | 0.010694 |
| Participation ratio (effective components) | 5.35 |
| Noise floor $\sigma_n$ | 0.04750 |

---

## 4. Shannon-Hartley Per-Component Capacity

For each detected component with amplitude $A_k$ in additive white noise
of variance $\sigma_n^2$, the per-component encoding capacity is:

$$C_k = \log_2\!\left(1 + \frac{A_k^2/2}{\sigma_n^2/N}\right) \text{ bits}$$

Total capacity across 5 matched components:
$C_{\rm total} = 77.0452$ bits.

---

## 5. Spectral Flatness

The **Wiener entropy** (spectral flatness measure):

$$\mathrm{SFM} = \frac{\exp\bigl(\langle \ln S(f) \rangle\bigr)}{\langle S(f) \rangle}
  = 0.010694$$

$\mathrm{SFM} \to 1$: white noise (maximally flat).
$\mathrm{SFM} \to 0$: tonal / highly structured signal.
The value $0.0107$ indicates a
highly structured signal with clear tonal components.

---
*IA-2026-210-T6 · 2026-07-29 · seed 20260729*
