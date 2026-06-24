--
session_id: IA-2026-175-T6
date: 2026-06-24
topic: Spectral Encoding Capacity
seed: 20260624
N_signal: 1024
fs_hz: 1000.0
n_components: 5
---

# Invention Archive — Daily Session 2026-06-24

**Session ID:** `IA-2026-175-T6`
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
plus Gaussian noise ($\sigma_n = 0.0439$):

$$x(t) = \sum_{k=1}^{5} A_k \cos(2\pi f_k t + \phi_k) + \eta(t)$$

True components: $f \in \{23.09, 36.36, 105.48, 120.73, 138.25\}$ Hz,
$A \in \{0.938, 0.807, 0.657, 0.476, 0.150\}$.

---

## 3. SpectraNova FFT Decomposition

Frequency resolution: $\Delta f = f_s / N = 0.977$ Hz.

### 3.1 Component Recovery

| $f_{\rm true}$ (Hz) | $A_{\rm true}$ | $f_{\rm det}$ (Hz) | $A_{\rm det}$ | $|f_{\rm err}|$ (Hz) | $C_k$ (bits) |
|---:|---:|---:|---:|---:|---:|
| 23.09 | 0.9375 | 23.44 | 0.7342 | 0.352 | 17.125 |
| 36.36 | 0.8071 | 36.13 | 0.7291 | 0.232 | 17.105 |
| 105.48 | 0.6566 | 105.47 | 0.6577 | 0.009 | 16.808 |
| 120.73 | 0.4755 | 121.09 | 0.3732 | 0.368 | 15.173 |
| 138.25 | 0.1498 | 138.67 | 0.1039 | 0.426 | 11.483 |

### 3.2 System-Level Statistics

| Metric | Value |
|---|---|
| Total signal SNR | 27.58 dB |
| Total FRAE encoding capacity $\sum_k C_k$ | **77.6934 bits** |
| Spectral flatness (Wiener entropy proxy) | 0.004181 |
| Participation ratio (effective components) | 5.87 |
| Noise floor $\sigma_n$ | 0.04394 |

---

## 4. Shannon-Hartley Per-Component Capacity

For each detected component with amplitude $A_k$ in additive white noise
of variance $\sigma_n^2$, the per-component encoding capacity is:

$$C_k = \log_2\!\left(1 + \frac{A_k^2/2}{\sigma_n^2/N}\right) \text{ bits}$$

Total capacity across 5 matched components:
$C_{\rm total} = 77.6934$ bits.

---

## 5. Spectral Flatness

The **Wiener entropy** (spectral flatness measure):

$$\mathrm{SFM} = \frac{\exp\bigl(\langle \ln S(f) \rangle\bigr)}{\langle S(f) \rangle}
  = 0.004181$$

$\mathrm{SFM} \to 1$: white noise (maximally flat).
$\mathrm{SFM} \to 0$: tonal / highly structured signal.
The value $0.0042$ indicates a
highly structured signal with clear tonal components.

---
*IA-2026-175-T6 · 2026-06-24 · seed 20260624*
