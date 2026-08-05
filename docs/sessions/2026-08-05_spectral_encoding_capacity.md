--
session_id: IA-2026-217-T6
date: 2026-08-05
topic: Spectral Encoding Capacity
seed: 20260805
N_signal: 1024
fs_hz: 1000.0
n_components: 5
---

# Invention Archive — Daily Session 2026-08-05

**Session ID:** `IA-2026-217-T6`
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
plus Gaussian noise ($\sigma_n = 0.0355$):

$$x(t) = \sum_{k=1}^{5} A_k \cos(2\pi f_k t + \phi_k) + \eta(t)$$

True components: $f \in \{33.13, 81.60, 108.77, 112.06, 136.71\}$ Hz,
$A \in \{0.919, 0.769, 0.576, 0.341, 0.219\}$.

---

## 3. SpectraNova FFT Decomposition

Frequency resolution: $\Delta f = f_s / N = 0.977$ Hz.

### 3.1 Component Recovery

| $f_{\rm true}$ (Hz) | $A_{\rm true}$ | $f_{\rm det}$ (Hz) | $A_{\rm det}$ | $|f_{\rm err}|$ (Hz) | $C_k$ (bits) |
|---:|---:|---:|---:|---:|---:|
| 33.13 | 0.9187 | 33.20 | 0.9155 | 0.069 | 18.377 |
| 81.60 | 0.7695 | 82.03 | 0.5383 | 0.434 | 16.844 |
| 108.77 | 0.5758 | 108.40 | 0.4198 | 0.372 | 16.127 |
| 112.06 | 0.3406 | 112.30 | 0.2553 | 0.248 | 14.692 |
| 136.71 | 0.2186 | 136.72 | 0.2126 | 0.014 | 14.164 |

### 3.2 System-Level Statistics

| Metric | Value |
|---|---|
| Total signal SNR | 28.84 dB |
| Total FRAE encoding capacity $\sum_k C_k$ | **80.2041 bits** |
| Spectral flatness (Wiener entropy proxy) | 0.004339 |
| Participation ratio (effective components) | 4.11 |
| Noise floor $\sigma_n$ | 0.03551 |

---

## 4. Shannon-Hartley Per-Component Capacity

For each detected component with amplitude $A_k$ in additive white noise
of variance $\sigma_n^2$, the per-component encoding capacity is:

$$C_k = \log_2\!\left(1 + \frac{A_k^2/2}{\sigma_n^2/N}\right) \text{ bits}$$

Total capacity across 5 matched components:
$C_{\rm total} = 80.2041$ bits.

---

## 5. Spectral Flatness

The **Wiener entropy** (spectral flatness measure):

$$\mathrm{SFM} = \frac{\exp\bigl(\langle \ln S(f) \rangle\bigr)}{\langle S(f) \rangle}
  = 0.004339$$

$\mathrm{SFM} \to 1$: white noise (maximally flat).
$\mathrm{SFM} \to 0$: tonal / highly structured signal.
The value $0.0043$ indicates a
highly structured signal with clear tonal components.

---
*IA-2026-217-T6 · 2026-08-05 · seed 20260805*
