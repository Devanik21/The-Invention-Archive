--
session_id: IA-2026-238-T6
date: 2026-08-26
topic: Spectral Encoding Capacity
seed: 20260826
N_signal: 1024
fs_hz: 1000.0
n_components: 5
---

# Invention Archive — Daily Session 2026-08-26

**Session ID:** `IA-2026-238-T6`
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

True components: $f \in \{26.47, 29.61, 60.37, 87.23, 126.22\}$ Hz,
$A \in \{0.888, 0.657, 0.575, 0.557, 0.209\}$.

---

## 3. SpectraNova FFT Decomposition

Frequency resolution: $\Delta f = f_s / N = 0.977$ Hz.

### 3.1 Component Recovery

| $f_{\rm true}$ (Hz) | $A_{\rm true}$ | $f_{\rm det}$ (Hz) | $A_{\rm det}$ | $|f_{\rm err}|$ (Hz) | $C_k$ (bits) |
|---:|---:|---:|---:|---:|---:|
| 26.47 | 0.8882 | 26.37 | 0.8428 | 0.101 | 16.299 |
| 60.37 | 0.5750 | 60.55 | 0.5410 | 0.175 | 15.020 |
| 87.23 | 0.5574 | 86.91 | 0.4676 | 0.321 | 14.599 |
| 126.22 | 0.2085 | 125.98 | 0.1877 | 0.242 | 11.966 |

### 3.2 System-Level Statistics

| Metric | Value |
|---|---|
| Total signal SNR | 23.25 dB |
| Total FRAE encoding capacity $\sum_k C_k$ | **57.8832 bits** |
| Spectral flatness (Wiener entropy proxy) | 0.007049 |
| Participation ratio (effective components) | 4.67 |
| Noise floor $\sigma_n$ | 0.06717 |

---

## 4. Shannon-Hartley Per-Component Capacity

For each detected component with amplitude $A_k$ in additive white noise
of variance $\sigma_n^2$, the per-component encoding capacity is:

$$C_k = \log_2\!\left(1 + \frac{A_k^2/2}{\sigma_n^2/N}\right) \text{ bits}$$

Total capacity across 4 matched components:
$C_{\rm total} = 57.8832$ bits.

---

## 5. Spectral Flatness

The **Wiener entropy** (spectral flatness measure):

$$\mathrm{SFM} = \frac{\exp\bigl(\langle \ln S(f) \rangle\bigr)}{\langle S(f) \rangle}
  = 0.007049$$

$\mathrm{SFM} \to 1$: white noise (maximally flat).
$\mathrm{SFM} \to 0$: tonal / highly structured signal.
The value $0.0070$ indicates a
highly structured signal with clear tonal components.

---
*IA-2026-238-T6 · 2026-08-26 · seed 20260826*
