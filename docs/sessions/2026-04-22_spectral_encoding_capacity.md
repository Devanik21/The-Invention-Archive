--
session_id: IA-2026-112-T6
date: 2026-04-22
topic: Spectral Encoding Capacity
seed: 20260422
N_signal: 1024
fs_hz: 1000.0
n_components: 5
---

# Invention Archive — Daily Session 2026-04-22

**Session ID:** `IA-2026-112-T6`
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
plus Gaussian noise ($\sigma_n = 0.0540$):

$$x(t) = \sum_{k=1}^{5} A_k \cos(2\pi f_k t + \phi_k) + \eta(t)$$

True components: $f \in \{4.92, 62.42, 68.96, 69.70, 78.31\}$ Hz,
$A \in \{0.931, 0.720, 0.450, 0.413, 0.400\}$.

---

## 3. SpectraNova FFT Decomposition

Frequency resolution: $\Delta f = f_s / N = 0.977$ Hz.

### 3.1 Component Recovery

| $f_{\rm true}$ (Hz) | $A_{\rm true}$ | $f_{\rm det}$ (Hz) | $A_{\rm det}$ | $|f_{\rm err}|$ (Hz) | $C_k$ (bits) |
|---:|---:|---:|---:|---:|---:|
| 4.92 | 0.9310 | 4.88 | 0.9298 | 0.034 | 17.214 |
| 62.42 | 0.7201 | 62.50 | 0.7058 | 0.078 | 16.419 |
| 68.96 | 0.4496 | 68.36 | 0.2947 | 0.601 | 13.899 |
| 78.31 | 0.4005 | 78.12 | 0.3557 | 0.182 | 14.442 |

### 3.2 System-Level Statistics

| Metric | Value |
|---|---|
| Total signal SNR | 25.18 dB |
| Total FRAE encoding capacity $\sum_k C_k$ | **61.9733 bits** |
| Spectral flatness (Wiener entropy proxy) | 0.008170 |
| Participation ratio (effective components) | 3.16 |
| Noise floor $\sigma_n$ | 0.05395 |

---

## 4. Shannon-Hartley Per-Component Capacity

For each detected component with amplitude $A_k$ in additive white noise
of variance $\sigma_n^2$, the per-component encoding capacity is:

$$C_k = \log_2\!\left(1 + \frac{A_k^2/2}{\sigma_n^2/N}\right) \text{ bits}$$

Total capacity across 4 matched components:
$C_{\rm total} = 61.9733$ bits.

---

## 5. Spectral Flatness

The **Wiener entropy** (spectral flatness measure):

$$\mathrm{SFM} = \frac{\exp\bigl(\langle \ln S(f) \rangle\bigr)}{\langle S(f) \rangle}
  = 0.008170$$

$\mathrm{SFM} \to 1$: white noise (maximally flat).
$\mathrm{SFM} \to 0$: tonal / highly structured signal.
The value $0.0082$ indicates a
highly structured signal with clear tonal components.

---
*IA-2026-112-T6 · 2026-04-22 · seed 20260422*
