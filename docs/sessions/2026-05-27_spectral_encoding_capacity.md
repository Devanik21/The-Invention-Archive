--
session_id: IA-2026-147-T6
date: 2026-05-27
topic: Spectral Encoding Capacity
seed: 20260527
N_signal: 1024
fs_hz: 1000.0
n_components: 5
---

# Invention Archive — Daily Session 2026-05-27

**Session ID:** `IA-2026-147-T6`
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
plus Gaussian noise ($\sigma_n = 0.0302$):

$$x(t) = \sum_{k=1}^{5} A_k \cos(2\pi f_k t + \phi_k) + \eta(t)$$

True components: $f \in \{14.82, 72.78, 93.02, 95.65, 102.25\}$ Hz,
$A \in \{0.891, 0.736, 0.687, 0.386, 0.201\}$.

---

## 3. SpectraNova FFT Decomposition

Frequency resolution: $\Delta f = f_s / N = 0.977$ Hz.

### 3.1 Component Recovery

| $f_{\rm true}$ (Hz) | $A_{\rm true}$ | $f_{\rm det}$ (Hz) | $A_{\rm det}$ | $|f_{\rm err}|$ (Hz) | $C_k$ (bits) |
|---:|---:|---:|---:|---:|---:|
| 14.82 | 0.8909 | 14.65 | 0.8418 | 0.172 | 18.598 |
| 72.78 | 0.7358 | 73.24 | 0.4793 | 0.467 | 16.973 |
| 93.02 | 0.6867 | 92.77 | 0.6120 | 0.247 | 17.678 |
| 102.25 | 0.2013 | 102.54 | 0.1949 | 0.287 | 14.378 |

### 3.2 System-Level Statistics

| Metric | Value |
|---|---|
| Total signal SNR | 30.38 dB |
| Total FRAE encoding capacity $\sum_k C_k$ | **67.6275 bits** |
| Spectral flatness (Wiener entropy proxy) | 0.004852 |
| Participation ratio (effective components) | 5.14 |
| Noise floor $\sigma_n$ | 0.03024 |

---

## 4. Shannon-Hartley Per-Component Capacity

For each detected component with amplitude $A_k$ in additive white noise
of variance $\sigma_n^2$, the per-component encoding capacity is:

$$C_k = \log_2\!\left(1 + \frac{A_k^2/2}{\sigma_n^2/N}\right) \text{ bits}$$

Total capacity across 4 matched components:
$C_{\rm total} = 67.6275$ bits.

---

## 5. Spectral Flatness

The **Wiener entropy** (spectral flatness measure):

$$\mathrm{SFM} = \frac{\exp\bigl(\langle \ln S(f) \rangle\bigr)}{\langle S(f) \rangle}
  = 0.004852$$

$\mathrm{SFM} \to 1$: white noise (maximally flat).
$\mathrm{SFM} \to 0$: tonal / highly structured signal.
The value $0.0049$ indicates a
highly structured signal with clear tonal components.

---
*IA-2026-147-T6 · 2026-05-27 · seed 20260527*
