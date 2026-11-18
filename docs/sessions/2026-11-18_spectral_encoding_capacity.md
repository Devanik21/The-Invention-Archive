--
session_id: IA-2026-322-T6
date: 2026-11-18
topic: Spectral Encoding Capacity
seed: 20261118
N_signal: 1024
fs_hz: 1000.0
n_components: 5
---

# Invention Archive — Daily Session 2026-11-18

**Session ID:** `IA-2026-322-T6`
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
plus Gaussian noise ($\sigma_n = 0.0575$):

$$x(t) = \sum_{k=1}^{5} A_k \cos(2\pi f_k t + \phi_k) + \eta(t)$$

True components: $f \in \{10.40, 42.64, 55.87, 59.90, 82.40\}$ Hz,
$A \in \{0.985, 0.770, 0.675, 0.220, 0.142\}$.

---

## 3. SpectraNova FFT Decomposition

Frequency resolution: $\Delta f = f_s / N = 0.977$ Hz.

### 3.1 Component Recovery

| $f_{\rm true}$ (Hz) | $A_{\rm true}$ | $f_{\rm det}$ (Hz) | $A_{\rm det}$ | $|f_{\rm err}|$ (Hz) | $C_k$ (bits) |
|---:|---:|---:|---:|---:|---:|
| 10.40 | 0.9846 | 10.74 | 0.7980 | 0.342 | 16.590 |
| 42.64 | 0.7704 | 42.97 | 0.6265 | 0.330 | 15.892 |
| 55.87 | 0.6746 | 55.66 | 0.6132 | 0.210 | 15.830 |
| 59.90 | 0.2199 | 59.57 | 0.2107 | 0.328 | 12.748 |
| 82.40 | 0.1418 | 82.03 | 0.1046 | 0.372 | 10.728 |

### 3.2 System-Level Statistics

| Metric | Value |
|---|---|
| Total signal SNR | 24.99 dB |
| Total FRAE encoding capacity $\sum_k C_k$ | **71.7877 bits** |
| Spectral flatness (Wiener entropy proxy) | 0.006276 |
| Participation ratio (effective components) | 5.74 |
| Noise floor $\sigma_n$ | 0.05749 |

---

## 4. Shannon-Hartley Per-Component Capacity

For each detected component with amplitude $A_k$ in additive white noise
of variance $\sigma_n^2$, the per-component encoding capacity is:

$$C_k = \log_2\!\left(1 + \frac{A_k^2/2}{\sigma_n^2/N}\right) \text{ bits}$$

Total capacity across 5 matched components:
$C_{\rm total} = 71.7877$ bits.

---

## 5. Spectral Flatness

The **Wiener entropy** (spectral flatness measure):

$$\mathrm{SFM} = \frac{\exp\bigl(\langle \ln S(f) \rangle\bigr)}{\langle S(f) \rangle}
  = 0.006276$$

$\mathrm{SFM} \to 1$: white noise (maximally flat).
$\mathrm{SFM} \to 0$: tonal / highly structured signal.
The value $0.0063$ indicates a
highly structured signal with clear tonal components.

---
*IA-2026-322-T6 · 2026-11-18 · seed 20261118*
