--
session_id: IA-2026-294-T6
date: 2026-10-21
topic: Spectral Encoding Capacity
seed: 20261021
N_signal: 1024
fs_hz: 1000.0
n_components: 5
---

# Invention Archive — Daily Session 2026-10-21

**Session ID:** `IA-2026-294-T6`
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
plus Gaussian noise ($\sigma_n = 0.0529$):

$$x(t) = \sum_{k=1}^{5} A_k \cos(2\pi f_k t + \phi_k) + \eta(t)$$

True components: $f \in \{6.81, 10.66, 21.95, 26.97, 34.62\}$ Hz,
$A \in \{0.918, 0.834, 0.732, 0.357, 0.298\}$.

---

## 3. SpectraNova FFT Decomposition

Frequency resolution: $\Delta f = f_s / N = 0.977$ Hz.

### 3.1 Component Recovery

| $f_{\rm true}$ (Hz) | $A_{\rm true}$ | $f_{\rm det}$ (Hz) | $A_{\rm det}$ | $|f_{\rm err}|$ (Hz) | $C_k$ (bits) |
|---:|---:|---:|---:|---:|---:|
| 6.81 | 0.9175 | 6.84 | 0.9298 | 0.029 | 17.272 |
| 10.66 | 0.8341 | 10.74 | 0.8198 | 0.085 | 16.908 |
| 21.95 | 0.7322 | 21.48 | 0.4776 | 0.464 | 15.350 |
| 26.97 | 0.3567 | 27.34 | 0.2571 | 0.375 | 13.563 |
| 34.62 | 0.2980 | 34.18 | 0.2210 | 0.442 | 13.126 |

### 3.2 System-Level Statistics

| Metric | Value |
|---|---|
| Total signal SNR | 26.12 dB |
| Total FRAE encoding capacity $\sum_k C_k$ | **76.2197 bits** |
| Spectral flatness (Wiener entropy proxy) | 0.002586 |
| Participation ratio (effective components) | 4.09 |
| Noise floor $\sigma_n$ | 0.05289 |

---

## 4. Shannon-Hartley Per-Component Capacity

For each detected component with amplitude $A_k$ in additive white noise
of variance $\sigma_n^2$, the per-component encoding capacity is:

$$C_k = \log_2\!\left(1 + \frac{A_k^2/2}{\sigma_n^2/N}\right) \text{ bits}$$

Total capacity across 5 matched components:
$C_{\rm total} = 76.2197$ bits.

---

## 5. Spectral Flatness

The **Wiener entropy** (spectral flatness measure):

$$\mathrm{SFM} = \frac{\exp\bigl(\langle \ln S(f) \rangle\bigr)}{\langle S(f) \rangle}
  = 0.002586$$

$\mathrm{SFM} \to 1$: white noise (maximally flat).
$\mathrm{SFM} \to 0$: tonal / highly structured signal.
The value $0.0026$ indicates a
highly structured signal with clear tonal components.

---
*IA-2026-294-T6 · 2026-10-21 · seed 20261021*
