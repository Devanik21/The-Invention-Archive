--
session_id: IA-2026-154-T6
date: 2026-06-03
topic: Spectral Encoding Capacity
seed: 20260603
N_signal: 1024
fs_hz: 1000.0
n_components: 5
---

# Invention Archive — Daily Session 2026-06-03

**Session ID:** `IA-2026-154-T6`
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
plus Gaussian noise ($\sigma_n = 0.0480$):

$$x(t) = \sum_{k=1}^{5} A_k \cos(2\pi f_k t + \phi_k) + \eta(t)$$

True components: $f \in \{8.67, 13.60, 71.72, 88.81, 102.64\}$ Hz,
$A \in \{0.891, 0.600, 0.281, 0.232, 0.156\}$.

---

## 3. SpectraNova FFT Decomposition

Frequency resolution: $\Delta f = f_s / N = 0.977$ Hz.

### 3.1 Component Recovery

| $f_{\rm true}$ (Hz) | $A_{\rm true}$ | $f_{\rm det}$ (Hz) | $A_{\rm det}$ | $|f_{\rm err}|$ (Hz) | $C_k$ (bits) |
|---:|---:|---:|---:|---:|---:|
| 8.67 | 0.8907 | 8.79 | 0.8671 | 0.123 | 17.351 |
| 13.60 | 0.5997 | 13.67 | 0.6190 | 0.071 | 16.378 |
| 71.72 | 0.2815 | 71.29 | 0.2014 | 0.430 | 13.139 |
| 88.81 | 0.2317 | 88.87 | 0.2298 | 0.059 | 13.519 |
| 102.64 | 0.1564 | 102.54 | 0.1545 | 0.098 | 12.375 |

### 3.2 System-Level Statistics

| Metric | Value |
|---|---|
| Total signal SNR | 24.54 dB |
| Total FRAE encoding capacity $\sum_k C_k$ | **72.7616 bits** |
| Spectral flatness (Wiener entropy proxy) | 0.004754 |
| Participation ratio (effective components) | 2.51 |
| Noise floor $\sigma_n$ | 0.04799 |

---

## 4. Shannon-Hartley Per-Component Capacity

For each detected component with amplitude $A_k$ in additive white noise
of variance $\sigma_n^2$, the per-component encoding capacity is:

$$C_k = \log_2\!\left(1 + \frac{A_k^2/2}{\sigma_n^2/N}\right) \text{ bits}$$

Total capacity across 5 matched components:
$C_{\rm total} = 72.7616$ bits.

---

## 5. Spectral Flatness

The **Wiener entropy** (spectral flatness measure):

$$\mathrm{SFM} = \frac{\exp\bigl(\langle \ln S(f) \rangle\bigr)}{\langle S(f) \rangle}
  = 0.004754$$

$\mathrm{SFM} \to 1$: white noise (maximally flat).
$\mathrm{SFM} \to 0$: tonal / highly structured signal.
The value $0.0048$ indicates a
highly structured signal with clear tonal components.

---
*IA-2026-154-T6 · 2026-06-03 · seed 20260603*
