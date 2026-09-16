--
session_id: IA-2026-259-T6
date: 2026-09-16
topic: Spectral Encoding Capacity
seed: 20260916
N_signal: 1024
fs_hz: 1000.0
n_components: 5
---

# Invention Archive — Daily Session 2026-09-16

**Session ID:** `IA-2026-259-T6`
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
plus Gaussian noise ($\sigma_n = 0.0517$):

$$x(t) = \sum_{k=1}^{5} A_k \cos(2\pi f_k t + \phi_k) + \eta(t)$$

True components: $f \in \{24.79, 27.80, 51.14, 51.80, 76.89\}$ Hz,
$A \in \{0.715, 0.700, 0.567, 0.537, 0.253\}$.

---

## 3. SpectraNova FFT Decomposition

Frequency resolution: $\Delta f = f_s / N = 0.977$ Hz.

### 3.1 Component Recovery

| $f_{\rm true}$ (Hz) | $A_{\rm true}$ | $f_{\rm det}$ (Hz) | $A_{\rm det}$ | $|f_{\rm err}|$ (Hz) | $C_k$ (bits) |
|---:|---:|---:|---:|---:|---:|
| 24.79 | 0.7154 | 27.34 | 0.5544 | 2.558 | 15.847 |
| 51.14 | 0.5668 | 51.76 | 0.7788 | 0.620 | 16.827 |
| 76.89 | 0.2526 | 77.15 | 0.2197 | 0.258 | 13.177 |

### 3.2 System-Level Statistics

| Metric | Value |
|---|---|
| Total signal SNR | 24.96 dB |
| Total FRAE encoding capacity $\sum_k C_k$ | **45.8506 bits** |
| Spectral flatness (Wiener entropy proxy) | 0.003398 |
| Participation ratio (effective components) | 5.71 |
| Noise floor $\sigma_n$ | 0.05168 |

---

## 4. Shannon-Hartley Per-Component Capacity

For each detected component with amplitude $A_k$ in additive white noise
of variance $\sigma_n^2$, the per-component encoding capacity is:

$$C_k = \log_2\!\left(1 + \frac{A_k^2/2}{\sigma_n^2/N}\right) \text{ bits}$$

Total capacity across 3 matched components:
$C_{\rm total} = 45.8506$ bits.

---

## 5. Spectral Flatness

The **Wiener entropy** (spectral flatness measure):

$$\mathrm{SFM} = \frac{\exp\bigl(\langle \ln S(f) \rangle\bigr)}{\langle S(f) \rangle}
  = 0.003398$$

$\mathrm{SFM} \to 1$: white noise (maximally flat).
$\mathrm{SFM} \to 0$: tonal / highly structured signal.
The value $0.0034$ indicates a
highly structured signal with clear tonal components.

---
*IA-2026-259-T6 · 2026-09-16 · seed 20260916*
