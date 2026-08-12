--
session_id: IA-2026-224-T6
date: 2026-08-12
topic: Spectral Encoding Capacity
seed: 20260812
N_signal: 1024
fs_hz: 1000.0
n_components: 5
---

# Invention Archive — Daily Session 2026-08-12

**Session ID:** `IA-2026-224-T6`
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
plus Gaussian noise ($\sigma_n = 0.0695$):

$$x(t) = \sum_{k=1}^{5} A_k \cos(2\pi f_k t + \phi_k) + \eta(t)$$

True components: $f \in \{10.09, 27.70, 62.97, 119.16, 140.84\}$ Hz,
$A \in \{0.608, 0.564, 0.342, 0.176, 0.144\}$.

---

## 3. SpectraNova FFT Decomposition

Frequency resolution: $\Delta f = f_s / N = 0.977$ Hz.

### 3.1 Component Recovery

| $f_{\rm true}$ (Hz) | $A_{\rm true}$ | $f_{\rm det}$ (Hz) | $A_{\rm det}$ | $|f_{\rm err}|$ (Hz) | $C_k$ (bits) |
|---:|---:|---:|---:|---:|---:|
| 10.09 | 0.6076 | 9.77 | 0.5104 | 0.322 | 14.751 |
| 27.70 | 0.5640 | 27.34 | 0.4512 | 0.360 | 14.396 |
| 62.97 | 0.3423 | 62.50 | 0.2272 | 0.466 | 12.416 |
| 119.16 | 0.1762 | 119.14 | 0.1788 | 0.020 | 11.726 |
| 140.84 | 0.1435 | 140.62 | 0.1290 | 0.215 | 10.784 |

### 3.2 System-Level Statistics

| Metric | Value |
|---|---|
| Total signal SNR | 19.47 dB |
| Total FRAE encoding capacity $\sum_k C_k$ | **64.0722 bits** |
| Spectral flatness (Wiener entropy proxy) | 0.014171 |
| Participation ratio (effective components) | 6.12 |
| Noise floor $\sigma_n$ | 0.06955 |

---

## 4. Shannon-Hartley Per-Component Capacity

For each detected component with amplitude $A_k$ in additive white noise
of variance $\sigma_n^2$, the per-component encoding capacity is:

$$C_k = \log_2\!\left(1 + \frac{A_k^2/2}{\sigma_n^2/N}\right) \text{ bits}$$

Total capacity across 5 matched components:
$C_{\rm total} = 64.0722$ bits.

---

## 5. Spectral Flatness

The **Wiener entropy** (spectral flatness measure):

$$\mathrm{SFM} = \frac{\exp\bigl(\langle \ln S(f) \rangle\bigr)}{\langle S(f) \rangle}
  = 0.014171$$

$\mathrm{SFM} \to 1$: white noise (maximally flat).
$\mathrm{SFM} \to 0$: tonal / highly structured signal.
The value $0.0142$ indicates a
highly structured signal with clear tonal components.

---
*IA-2026-224-T6 · 2026-08-12 · seed 20260812*
