--
session_id: IA-2026-266-T6
date: 2026-09-23
topic: Spectral Encoding Capacity
seed: 20260923
N_signal: 1024
fs_hz: 1000.0
n_components: 5
---

# Invention Archive — Daily Session 2026-09-23

**Session ID:** `IA-2026-266-T6`
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
plus Gaussian noise ($\sigma_n = 0.0618$):

$$x(t) = \sum_{k=1}^{5} A_k \cos(2\pi f_k t + \phi_k) + \eta(t)$$

True components: $f \in \{34.76, 47.15, 79.31, 128.31, 148.50\}$ Hz,
$A \in \{0.993, 0.783, 0.341, 0.216, 0.137\}$.

---

## 3. SpectraNova FFT Decomposition

Frequency resolution: $\Delta f = f_s / N = 0.977$ Hz.

### 3.1 Component Recovery

| $f_{\rm true}$ (Hz) | $A_{\rm true}$ | $f_{\rm det}$ (Hz) | $A_{\rm det}$ | $|f_{\rm err}|$ (Hz) | $C_k$ (bits) |
|---:|---:|---:|---:|---:|---:|
| 34.76 | 0.9925 | 35.16 | 0.7476 | 0.393 | 16.195 |
| 47.15 | 0.7828 | 46.88 | 0.6919 | 0.275 | 15.972 |
| 79.31 | 0.3411 | 79.10 | 0.3057 | 0.210 | 13.615 |
| 128.31 | 0.2163 | 127.93 | 0.1717 | 0.384 | 11.950 |
| 148.50 | 0.1372 | 148.44 | 0.1339 | 0.062 | 11.234 |

### 3.2 System-Level Statistics

| Metric | Value |
|---|---|
| Total signal SNR | 23.68 dB |
| Total FRAE encoding capacity $\sum_k C_k$ | **68.9668 bits** |
| Spectral flatness (Wiener entropy proxy) | 0.008316 |
| Participation ratio (effective components) | 5.09 |
| Noise floor $\sigma_n$ | 0.06176 |

---

## 4. Shannon-Hartley Per-Component Capacity

For each detected component with amplitude $A_k$ in additive white noise
of variance $\sigma_n^2$, the per-component encoding capacity is:

$$C_k = \log_2\!\left(1 + \frac{A_k^2/2}{\sigma_n^2/N}\right) \text{ bits}$$

Total capacity across 5 matched components:
$C_{\rm total} = 68.9668$ bits.

---

## 5. Spectral Flatness

The **Wiener entropy** (spectral flatness measure):

$$\mathrm{SFM} = \frac{\exp\bigl(\langle \ln S(f) \rangle\bigr)}{\langle S(f) \rangle}
  = 0.008316$$

$\mathrm{SFM} \to 1$: white noise (maximally flat).
$\mathrm{SFM} \to 0$: tonal / highly structured signal.
The value $0.0083$ indicates a
highly structured signal with clear tonal components.

---
*IA-2026-266-T6 · 2026-09-23 · seed 20260923*
