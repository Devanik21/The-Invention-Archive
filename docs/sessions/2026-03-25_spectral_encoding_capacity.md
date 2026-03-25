--
session_id: IA-2026-084-T6
date: 2026-03-25
topic: Spectral Encoding Capacity
seed: 20260325
N_signal: 1024
fs_hz: 1000.0
n_components: 5
---

# Invention Archive — Daily Session 2026-03-25

**Session ID:** `IA-2026-084-T6`
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
plus Gaussian noise ($\sigma_n = 0.0379$):

$$x(t) = \sum_{k=1}^{5} A_k \cos(2\pi f_k t + \phi_k) + \eta(t)$$

True components: $f \in \{8.24, 22.60, 89.52, 133.49, 133.70\}$ Hz,
$A \in \{0.924, 0.743, 0.587, 0.357, 0.315\}$.

---

## 3. SpectraNova FFT Decomposition

Frequency resolution: $\Delta f = f_s / N = 0.977$ Hz.

### 3.1 Component Recovery

| $f_{\rm true}$ (Hz) | $A_{\rm true}$ | $f_{\rm det}$ (Hz) | $A_{\rm det}$ | $|f_{\rm err}|$ (Hz) | $C_k$ (bits) |
|---:|---:|---:|---:|---:|---:|
| 8.24 | 0.9237 | 7.81 | 0.6528 | 0.426 | 17.212 |
| 22.60 | 0.7426 | 22.46 | 0.7218 | 0.139 | 17.502 |
| 89.52 | 0.5872 | 89.84 | 0.4864 | 0.322 | 16.363 |
| 133.49 | 0.3570 | 133.79 | 0.5085 | 0.301 | 16.491 |

### 3.2 System-Level Statistics

| Metric | Value |
|---|---|
| Total signal SNR | 28.37 dB |
| Total FRAE encoding capacity $\sum_k C_k$ | **67.5683 bits** |
| Spectral flatness (Wiener entropy proxy) | 0.008762 |
| Participation ratio (effective components) | 6.45 |
| Noise floor $\sigma_n$ | 0.03791 |

---

## 4. Shannon-Hartley Per-Component Capacity

For each detected component with amplitude $A_k$ in additive white noise
of variance $\sigma_n^2$, the per-component encoding capacity is:

$$C_k = \log_2\!\left(1 + \frac{A_k^2/2}{\sigma_n^2/N}\right) \text{ bits}$$

Total capacity across 4 matched components:
$C_{\rm total} = 67.5683$ bits.

---

## 5. Spectral Flatness

The **Wiener entropy** (spectral flatness measure):

$$\mathrm{SFM} = \frac{\exp\bigl(\langle \ln S(f) \rangle\bigr)}{\langle S(f) \rangle}
  = 0.008762$$

$\mathrm{SFM} \to 1$: white noise (maximally flat).
$\mathrm{SFM} \to 0$: tonal / highly structured signal.
The value $0.0088$ indicates a
highly structured signal with clear tonal components.

---
*IA-2026-084-T6 · 2026-03-25 · seed 20260325*
