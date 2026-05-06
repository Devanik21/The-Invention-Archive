--
session_id: IA-2026-126-T6
date: 2026-05-06
topic: Spectral Encoding Capacity
seed: 20260506
N_signal: 1024
fs_hz: 1000.0
n_components: 5
---

# Invention Archive — Daily Session 2026-05-06

**Session ID:** `IA-2026-126-T6`
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
plus Gaussian noise ($\sigma_n = 0.0672$):

$$x(t) = \sum_{k=1}^{5} A_k \cos(2\pi f_k t + \phi_k) + \eta(t)$$

True components: $f \in \{8.13, 41.28, 60.43, 77.66, 130.73\}$ Hz,
$A \in \{0.891, 0.834, 0.418, 0.279, 0.184\}$.

---

## 3. SpectraNova FFT Decomposition

Frequency resolution: $\Delta f = f_s / N = 0.977$ Hz.

### 3.1 Component Recovery

| $f_{\rm true}$ (Hz) | $A_{\rm true}$ | $f_{\rm det}$ (Hz) | $A_{\rm det}$ | $|f_{\rm err}|$ (Hz) | $C_k$ (bits) |
|---:|---:|---:|---:|---:|---:|
| 8.13 | 0.8906 | 7.81 | 0.7493 | 0.320 | 15.958 |
| 41.28 | 0.8344 | 41.02 | 0.7346 | 0.269 | 15.901 |
| 60.43 | 0.4175 | 60.55 | 0.4038 | 0.117 | 14.174 |
| 77.66 | 0.2788 | 78.12 | 0.1908 | 0.466 | 12.011 |
| 130.73 | 0.1840 | 130.86 | 0.1756 | 0.132 | 11.771 |

### 3.2 System-Level Statistics

| Metric | Value |
|---|---|
| Total signal SNR | 22.94 dB |
| Total FRAE encoding capacity $\sum_k C_k$ | **69.8152 bits** |
| Spectral flatness (Wiener entropy proxy) | 0.008173 |
| Participation ratio (effective components) | 4.86 |
| Noise floor $\sigma_n$ | 0.06720 |

---

## 4. Shannon-Hartley Per-Component Capacity

For each detected component with amplitude $A_k$ in additive white noise
of variance $\sigma_n^2$, the per-component encoding capacity is:

$$C_k = \log_2\!\left(1 + \frac{A_k^2/2}{\sigma_n^2/N}\right) \text{ bits}$$

Total capacity across 5 matched components:
$C_{\rm total} = 69.8152$ bits.

---

## 5. Spectral Flatness

The **Wiener entropy** (spectral flatness measure):

$$\mathrm{SFM} = \frac{\exp\bigl(\langle \ln S(f) \rangle\bigr)}{\langle S(f) \rangle}
  = 0.008173$$

$\mathrm{SFM} \to 1$: white noise (maximally flat).
$\mathrm{SFM} \to 0$: tonal / highly structured signal.
The value $0.0082$ indicates a
highly structured signal with clear tonal components.

---
*IA-2026-126-T6 · 2026-05-06 · seed 20260506*
