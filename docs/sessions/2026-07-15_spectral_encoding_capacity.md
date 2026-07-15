--
session_id: IA-2026-196-T6
date: 2026-07-15
topic: Spectral Encoding Capacity
seed: 20260715
N_signal: 1024
fs_hz: 1000.0
n_components: 5
---

# Invention Archive — Daily Session 2026-07-15

**Session ID:** `IA-2026-196-T6`
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
plus Gaussian noise ($\sigma_n = 0.0588$):

$$x(t) = \sum_{k=1}^{5} A_k \cos(2\pi f_k t + \phi_k) + \eta(t)$$

True components: $f \in \{7.02, 25.80, 59.67, 84.95, 94.69\}$ Hz,
$A \in \{0.912, 0.663, 0.633, 0.409, 0.233\}$.

---

## 3. SpectraNova FFT Decomposition

Frequency resolution: $\Delta f = f_s / N = 0.977$ Hz.

### 3.1 Component Recovery

| $f_{\rm true}$ (Hz) | $A_{\rm true}$ | $f_{\rm det}$ (Hz) | $A_{\rm det}$ | $|f_{\rm err}|$ (Hz) | $C_k$ (bits) |
|---:|---:|---:|---:|---:|---:|
| 7.02 | 0.9123 | 6.84 | 0.8733 | 0.179 | 16.787 |
| 25.80 | 0.6629 | 25.39 | 0.4813 | 0.406 | 15.068 |
| 59.67 | 0.6327 | 59.57 | 0.6097 | 0.095 | 15.751 |
| 84.95 | 0.4090 | 84.96 | 0.4041 | 0.010 | 14.563 |
| 94.69 | 0.2334 | 94.73 | 0.2235 | 0.038 | 12.855 |

### 3.2 System-Level Statistics

| Metric | Value |
|---|---|
| Total signal SNR | 24.38 dB |
| Total FRAE encoding capacity $\sum_k C_k$ | **75.0243 bits** |
| Spectral flatness (Wiener entropy proxy) | 0.007490 |
| Participation ratio (effective components) | 4.35 |
| Noise floor $\sigma_n$ | 0.05876 |

---

## 4. Shannon-Hartley Per-Component Capacity

For each detected component with amplitude $A_k$ in additive white noise
of variance $\sigma_n^2$, the per-component encoding capacity is:

$$C_k = \log_2\!\left(1 + \frac{A_k^2/2}{\sigma_n^2/N}\right) \text{ bits}$$

Total capacity across 5 matched components:
$C_{\rm total} = 75.0243$ bits.

---

## 5. Spectral Flatness

The **Wiener entropy** (spectral flatness measure):

$$\mathrm{SFM} = \frac{\exp\bigl(\langle \ln S(f) \rangle\bigr)}{\langle S(f) \rangle}
  = 0.007490$$

$\mathrm{SFM} \to 1$: white noise (maximally flat).
$\mathrm{SFM} \to 0$: tonal / highly structured signal.
The value $0.0075$ indicates a
highly structured signal with clear tonal components.

---
*IA-2026-196-T6 · 2026-07-15 · seed 20260715*
