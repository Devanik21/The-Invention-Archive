--
session_id: IA-2026-287-T6
date: 2026-10-14
topic: Spectral Encoding Capacity
seed: 20261014
N_signal: 1024
fs_hz: 1000.0
n_components: 5
---

# Invention Archive — Daily Session 2026-10-14

**Session ID:** `IA-2026-287-T6`
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
plus Gaussian noise ($\sigma_n = 0.0602$):

$$x(t) = \sum_{k=1}^{5} A_k \cos(2\pi f_k t + \phi_k) + \eta(t)$$

True components: $f \in \{20.05, 73.08, 90.61, 118.00, 130.91\}$ Hz,
$A \in \{0.757, 0.610, 0.575, 0.466, 0.417\}$.

---

## 3. SpectraNova FFT Decomposition

Frequency resolution: $\Delta f = f_s / N = 0.977$ Hz.

### 3.1 Component Recovery

| $f_{\rm true}$ (Hz) | $A_{\rm true}$ | $f_{\rm det}$ (Hz) | $A_{\rm det}$ | $|f_{\rm err}|$ (Hz) | $C_k$ (bits) |
|---:|---:|---:|---:|---:|---:|
| 20.05 | 0.7573 | 20.51 | 0.5117 | 0.461 | 15.174 |
| 73.08 | 0.6100 | 73.24 | 0.5829 | 0.159 | 15.550 |
| 90.61 | 0.5751 | 90.82 | 0.5377 | 0.206 | 15.317 |
| 118.00 | 0.4660 | 118.16 | 0.4305 | 0.166 | 14.676 |
| 130.91 | 0.4167 | 130.86 | 0.4141 | 0.049 | 14.564 |

### 3.2 System-Level Statistics

| Metric | Value |
|---|---|
| Total signal SNR | 23.62 dB |
| Total FRAE encoding capacity $\sum_k C_k$ | **75.2813 bits** |
| Spectral flatness (Wiener entropy proxy) | 0.008109 |
| Participation ratio (effective components) | 7.39 |
| Noise floor $\sigma_n$ | 0.06022 |

---

## 4. Shannon-Hartley Per-Component Capacity

For each detected component with amplitude $A_k$ in additive white noise
of variance $\sigma_n^2$, the per-component encoding capacity is:

$$C_k = \log_2\!\left(1 + \frac{A_k^2/2}{\sigma_n^2/N}\right) \text{ bits}$$

Total capacity across 5 matched components:
$C_{\rm total} = 75.2813$ bits.

---

## 5. Spectral Flatness

The **Wiener entropy** (spectral flatness measure):

$$\mathrm{SFM} = \frac{\exp\bigl(\langle \ln S(f) \rangle\bigr)}{\langle S(f) \rangle}
  = 0.008109$$

$\mathrm{SFM} \to 1$: white noise (maximally flat).
$\mathrm{SFM} \to 0$: tonal / highly structured signal.
The value $0.0081$ indicates a
highly structured signal with clear tonal components.

---
*IA-2026-287-T6 · 2026-10-14 · seed 20261014*
