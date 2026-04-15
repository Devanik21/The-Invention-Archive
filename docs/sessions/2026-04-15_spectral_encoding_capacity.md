--
session_id: IA-2026-105-T6
date: 2026-04-15
topic: Spectral Encoding Capacity
seed: 20260415
N_signal: 1024
fs_hz: 1000.0
n_components: 5
---

# Invention Archive — Daily Session 2026-04-15

**Session ID:** `IA-2026-105-T6`
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
plus Gaussian noise ($\sigma_n = 0.0317$):

$$x(t) = \sum_{k=1}^{5} A_k \cos(2\pi f_k t + \phi_k) + \eta(t)$$

True components: $f \in \{23.54, 66.91, 76.00, 108.55, 138.53\}$ Hz,
$A \in \{0.951, 0.641, 0.196, 0.195, 0.118\}$.

---

## 3. SpectraNova FFT Decomposition

Frequency resolution: $\Delta f = f_s / N = 0.977$ Hz.

### 3.1 Component Recovery

| $f_{\rm true}$ (Hz) | $A_{\rm true}$ | $f_{\rm det}$ (Hz) | $A_{\rm det}$ | $|f_{\rm err}|$ (Hz) | $C_k$ (bits) |
|---:|---:|---:|---:|---:|---:|
| 23.54 | 0.9505 | 23.44 | 0.9274 | 0.107 | 18.737 |
| 66.91 | 0.6406 | 67.38 | 0.4175 | 0.472 | 16.434 |
| 76.00 | 0.1961 | 76.17 | 0.1981 | 0.171 | 14.283 |
| 108.55 | 0.1952 | 108.40 | 0.1882 | 0.148 | 14.135 |
| 138.53 | 0.1180 | 138.67 | 0.1166 | 0.142 | 12.754 |

### 3.2 System-Level Statistics

| Metric | Value |
|---|---|
| Total signal SNR | 28.43 dB |
| Total FRAE encoding capacity $\sum_k C_k$ | **76.3418 bits** |
| Spectral flatness (Wiener entropy proxy) | 0.003454 |
| Participation ratio (effective components) | 2.47 |
| Noise floor $\sigma_n$ | 0.03175 |

---

## 4. Shannon-Hartley Per-Component Capacity

For each detected component with amplitude $A_k$ in additive white noise
of variance $\sigma_n^2$, the per-component encoding capacity is:

$$C_k = \log_2\!\left(1 + \frac{A_k^2/2}{\sigma_n^2/N}\right) \text{ bits}$$

Total capacity across 5 matched components:
$C_{\rm total} = 76.3418$ bits.

---

## 5. Spectral Flatness

The **Wiener entropy** (spectral flatness measure):

$$\mathrm{SFM} = \frac{\exp\bigl(\langle \ln S(f) \rangle\bigr)}{\langle S(f) \rangle}
  = 0.003454$$

$\mathrm{SFM} \to 1$: white noise (maximally flat).
$\mathrm{SFM} \to 0$: tonal / highly structured signal.
The value $0.0035$ indicates a
highly structured signal with clear tonal components.

---
*IA-2026-105-T6 · 2026-04-15 · seed 20260415*
