--
session_id: IA-2026-182-T6
date: 2026-07-01
topic: Spectral Encoding Capacity
seed: 20260701
N_signal: 1024
fs_hz: 1000.0
n_components: 5
---

# Invention Archive — Daily Session 2026-07-01

**Session ID:** `IA-2026-182-T6`
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
plus Gaussian noise ($\sigma_n = 0.0534$):

$$x(t) = \sum_{k=1}^{5} A_k \cos(2\pi f_k t + \phi_k) + \eta(t)$$

True components: $f \in \{19.92, 58.00, 79.05, 98.58, 143.78\}$ Hz,
$A \in \{0.826, 0.784, 0.672, 0.114, 0.103\}$.

---

## 3. SpectraNova FFT Decomposition

Frequency resolution: $\Delta f = f_s / N = 0.977$ Hz.

### 3.1 Component Recovery

| $f_{\rm true}$ (Hz) | $A_{\rm true}$ | $f_{\rm det}$ (Hz) | $A_{\rm det}$ | $|f_{\rm err}|$ (Hz) | $C_k$ (bits) |
|---:|---:|---:|---:|---:|---:|
| 19.92 | 0.8257 | 19.53 | 0.6343 | 0.393 | 16.143 |
| 58.00 | 0.7841 | 57.62 | 0.5965 | 0.386 | 15.965 |
| 79.05 | 0.6718 | 79.10 | 0.6570 | 0.048 | 16.244 |
| 98.58 | 0.1142 | 98.63 | 0.1086 | 0.054 | 11.051 |
| 143.78 | 0.1030 | 143.55 | 0.0976 | 0.229 | 10.742 |

### 3.2 System-Level Statistics

| Metric | Value |
|---|---|
| Total signal SNR | 24.93 dB |
| Total FRAE encoding capacity $\sum_k C_k$ | **70.1457 bits** |
| Spectral flatness (Wiener entropy proxy) | 0.005684 |
| Participation ratio (effective components) | 5.89 |
| Noise floor $\sigma_n$ | 0.05336 |

---

## 4. Shannon-Hartley Per-Component Capacity

For each detected component with amplitude $A_k$ in additive white noise
of variance $\sigma_n^2$, the per-component encoding capacity is:

$$C_k = \log_2\!\left(1 + \frac{A_k^2/2}{\sigma_n^2/N}\right) \text{ bits}$$

Total capacity across 5 matched components:
$C_{\rm total} = 70.1457$ bits.

---

## 5. Spectral Flatness

The **Wiener entropy** (spectral flatness measure):

$$\mathrm{SFM} = \frac{\exp\bigl(\langle \ln S(f) \rangle\bigr)}{\langle S(f) \rangle}
  = 0.005684$$

$\mathrm{SFM} \to 1$: white noise (maximally flat).
$\mathrm{SFM} \to 0$: tonal / highly structured signal.
The value $0.0057$ indicates a
highly structured signal with clear tonal components.

---
*IA-2026-182-T6 · 2026-07-01 · seed 20260701*
