--
session_id: IA-2026-301-T6
date: 2026-10-28
topic: Spectral Encoding Capacity
seed: 20261028
N_signal: 1024
fs_hz: 1000.0
n_components: 5
---

# Invention Archive — Daily Session 2026-10-28

**Session ID:** `IA-2026-301-T6`
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
plus Gaussian noise ($\sigma_n = 0.0375$):

$$x(t) = \sum_{k=1}^{5} A_k \cos(2\pi f_k t + \phi_k) + \eta(t)$$

True components: $f \in \{5.36, 6.18, 27.86, 85.37, 99.22\}$ Hz,
$A \in \{0.711, 0.667, 0.278, 0.264, 0.156\}$.

---

## 3. SpectraNova FFT Decomposition

Frequency resolution: $\Delta f = f_s / N = 0.977$ Hz.

### 3.1 Component Recovery

| $f_{\rm true}$ (Hz) | $A_{\rm true}$ | $f_{\rm det}$ (Hz) | $A_{\rm det}$ | $|f_{\rm err}|$ (Hz) | $C_k$ (bits) |
|---:|---:|---:|---:|---:|---:|
| 5.36 | 0.7110 | 5.86 | 0.5657 | 0.503 | 16.830 |
| 27.86 | 0.2784 | 28.32 | 0.2069 | 0.459 | 13.929 |
| 85.37 | 0.2644 | 84.96 | 0.1901 | 0.414 | 13.685 |
| 99.22 | 0.1559 | 99.61 | 0.1146 | 0.394 | 12.224 |

### 3.2 System-Level Statistics

| Metric | Value |
|---|---|
| Total signal SNR | 26.01 dB |
| Total FRAE encoding capacity $\sum_k C_k$ | **56.6673 bits** |
| Spectral flatness (Wiener entropy proxy) | 0.010752 |
| Participation ratio (effective components) | 5.48 |
| Noise floor $\sigma_n$ | 0.03750 |

---

## 4. Shannon-Hartley Per-Component Capacity

For each detected component with amplitude $A_k$ in additive white noise
of variance $\sigma_n^2$, the per-component encoding capacity is:

$$C_k = \log_2\!\left(1 + \frac{A_k^2/2}{\sigma_n^2/N}\right) \text{ bits}$$

Total capacity across 4 matched components:
$C_{\rm total} = 56.6673$ bits.

---

## 5. Spectral Flatness

The **Wiener entropy** (spectral flatness measure):

$$\mathrm{SFM} = \frac{\exp\bigl(\langle \ln S(f) \rangle\bigr)}{\langle S(f) \rangle}
  = 0.010752$$

$\mathrm{SFM} \to 1$: white noise (maximally flat).
$\mathrm{SFM} \to 0$: tonal / highly structured signal.
The value $0.0108$ indicates a
highly structured signal with clear tonal components.

---
*IA-2026-301-T6 · 2026-10-28 · seed 20261028*
