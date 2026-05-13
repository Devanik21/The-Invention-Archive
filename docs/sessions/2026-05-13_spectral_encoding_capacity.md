--
session_id: IA-2026-133-T6
date: 2026-05-13
topic: Spectral Encoding Capacity
seed: 20260513
N_signal: 1024
fs_hz: 1000.0
n_components: 5
---

# Invention Archive — Daily Session 2026-05-13

**Session ID:** `IA-2026-133-T6`
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
plus Gaussian noise ($\sigma_n = 0.0530$):

$$x(t) = \sum_{k=1}^{5} A_k \cos(2\pi f_k t + \phi_k) + \eta(t)$$

True components: $f \in \{16.24, 80.55, 97.27, 99.39, 131.84\}$ Hz,
$A \in \{0.873, 0.490, 0.457, 0.253, 0.178\}$.

---

## 3. SpectraNova FFT Decomposition

Frequency resolution: $\Delta f = f_s / N = 0.977$ Hz.

### 3.1 Component Recovery

| $f_{\rm true}$ (Hz) | $A_{\rm true}$ | $f_{\rm det}$ (Hz) | $A_{\rm det}$ | $|f_{\rm err}|$ (Hz) | $C_k$ (bits) |
|---:|---:|---:|---:|---:|---:|
| 16.24 | 0.8726 | 16.60 | 0.6818 | 0.364 | 16.370 |
| 80.55 | 0.4902 | 80.08 | 0.3162 | 0.476 | 14.153 |
| 97.27 | 0.4568 | 97.66 | 0.3647 | 0.389 | 14.564 |
| 131.84 | 0.1781 | 131.84 | 0.1782 | 0.007 | 12.499 |

### 3.2 System-Level Statistics

| Metric | Value |
|---|---|
| Total signal SNR | 23.66 dB |
| Total FRAE encoding capacity $\sum_k C_k$ | **57.5858 bits** |
| Spectral flatness (Wiener entropy proxy) | 0.007144 |
| Participation ratio (effective components) | 5.73 |
| Noise floor $\sigma_n$ | 0.05302 |

---

## 4. Shannon-Hartley Per-Component Capacity

For each detected component with amplitude $A_k$ in additive white noise
of variance $\sigma_n^2$, the per-component encoding capacity is:

$$C_k = \log_2\!\left(1 + \frac{A_k^2/2}{\sigma_n^2/N}\right) \text{ bits}$$

Total capacity across 4 matched components:
$C_{\rm total} = 57.5858$ bits.

---

## 5. Spectral Flatness

The **Wiener entropy** (spectral flatness measure):

$$\mathrm{SFM} = \frac{\exp\bigl(\langle \ln S(f) \rangle\bigr)}{\langle S(f) \rangle}
  = 0.007144$$

$\mathrm{SFM} \to 1$: white noise (maximally flat).
$\mathrm{SFM} \to 0$: tonal / highly structured signal.
The value $0.0071$ indicates a
highly structured signal with clear tonal components.

---
*IA-2026-133-T6 · 2026-05-13 · seed 20260513*
