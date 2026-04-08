--
session_id: IA-2026-098-T6
date: 2026-04-08
topic: Spectral Encoding Capacity
seed: 20260408
N_signal: 1024
fs_hz: 1000.0
n_components: 5
---

# Invention Archive — Daily Session 2026-04-08

**Session ID:** `IA-2026-098-T6`
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
plus Gaussian noise ($\sigma_n = 0.0334$):

$$x(t) = \sum_{k=1}^{5} A_k \cos(2\pi f_k t + \phi_k) + \eta(t)$$

True components: $f \in \{5.02, 21.85, 67.93, 76.34, 137.87\}$ Hz,
$A \in \{0.997, 0.919, 0.875, 0.372, 0.222\}$.

---

## 3. SpectraNova FFT Decomposition

Frequency resolution: $\Delta f = f_s / N = 0.977$ Hz.

### 3.1 Component Recovery

| $f_{\rm true}$ (Hz) | $A_{\rm true}$ | $f_{\rm det}$ (Hz) | $A_{\rm det}$ | $|f_{\rm err}|$ (Hz) | $C_k$ (bits) |
|---:|---:|---:|---:|---:|---:|
| 5.02 | 0.9972 | 4.88 | 0.9451 | 0.138 | 18.645 |
| 21.85 | 0.9188 | 21.48 | 0.7343 | 0.368 | 17.917 |
| 67.93 | 0.8747 | 68.36 | 0.6150 | 0.431 | 17.405 |
| 76.34 | 0.3717 | 76.17 | 0.3212 | 0.170 | 15.531 |
| 137.87 | 0.2223 | 137.70 | 0.2089 | 0.179 | 14.289 |

### 3.2 System-Level Statistics

| Metric | Value |
|---|---|
| Total signal SNR | 30.97 dB |
| Total FRAE encoding capacity $\sum_k C_k$ | **83.7880 bits** |
| Spectral flatness (Wiener entropy proxy) | 0.002771 |
| Participation ratio (effective components) | 5.65 |
| Noise floor $\sigma_n$ | 0.03340 |

---

## 4. Shannon-Hartley Per-Component Capacity

For each detected component with amplitude $A_k$ in additive white noise
of variance $\sigma_n^2$, the per-component encoding capacity is:

$$C_k = \log_2\!\left(1 + \frac{A_k^2/2}{\sigma_n^2/N}\right) \text{ bits}$$

Total capacity across 5 matched components:
$C_{\rm total} = 83.7880$ bits.

---

## 5. Spectral Flatness

The **Wiener entropy** (spectral flatness measure):

$$\mathrm{SFM} = \frac{\exp\bigl(\langle \ln S(f) \rangle\bigr)}{\langle S(f) \rangle}
  = 0.002771$$

$\mathrm{SFM} \to 1$: white noise (maximally flat).
$\mathrm{SFM} \to 0$: tonal / highly structured signal.
The value $0.0028$ indicates a
highly structured signal with clear tonal components.

---
*IA-2026-098-T6 · 2026-04-08 · seed 20260408*
