--
session_id: IA-2026-203-T6
date: 2026-07-22
topic: Spectral Encoding Capacity
seed: 20260722
N_signal: 1024
fs_hz: 1000.0
n_components: 5
---

# Invention Archive — Daily Session 2026-07-22

**Session ID:** `IA-2026-203-T6`
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
plus Gaussian noise ($\sigma_n = 0.0404$):

$$x(t) = \sum_{k=1}^{5} A_k \cos(2\pi f_k t + \phi_k) + \eta(t)$$

True components: $f \in \{63.30, 94.99, 113.01, 142.72, 149.35\}$ Hz,
$A \in \{0.805, 0.801, 0.665, 0.630, 0.320\}$.

---

## 3. SpectraNova FFT Decomposition

Frequency resolution: $\Delta f = f_s / N = 0.977$ Hz.

### 3.1 Component Recovery

| $f_{\rm true}$ (Hz) | $A_{\rm true}$ | $f_{\rm det}$ (Hz) | $A_{\rm det}$ | $|f_{\rm err}|$ (Hz) | $C_k$ (bits) |
|---:|---:|---:|---:|---:|---:|
| 63.30 | 0.8053 | 63.48 | 0.7560 | 0.181 | 17.454 |
| 94.99 | 0.8009 | 94.73 | 0.7043 | 0.264 | 17.249 |
| 113.01 | 0.6646 | 113.28 | 0.5939 | 0.270 | 16.758 |
| 142.72 | 0.6302 | 142.58 | 0.6192 | 0.143 | 16.878 |
| 149.35 | 0.3203 | 149.41 | 0.3253 | 0.067 | 15.020 |

### 3.2 System-Level Statistics

| Metric | Value |
|---|---|
| Total signal SNR | 28.35 dB |
| Total FRAE encoding capacity $\sum_k C_k$ | **83.3592 bits** |
| Spectral flatness (Wiener entropy proxy) | 0.007742 |
| Participation ratio (effective components) | 5.84 |
| Noise floor $\sigma_n$ | 0.04037 |

---

## 4. Shannon-Hartley Per-Component Capacity

For each detected component with amplitude $A_k$ in additive white noise
of variance $\sigma_n^2$, the per-component encoding capacity is:

$$C_k = \log_2\!\left(1 + \frac{A_k^2/2}{\sigma_n^2/N}\right) \text{ bits}$$

Total capacity across 5 matched components:
$C_{\rm total} = 83.3592$ bits.

---

## 5. Spectral Flatness

The **Wiener entropy** (spectral flatness measure):

$$\mathrm{SFM} = \frac{\exp\bigl(\langle \ln S(f) \rangle\bigr)}{\langle S(f) \rangle}
  = 0.007742$$

$\mathrm{SFM} \to 1$: white noise (maximally flat).
$\mathrm{SFM} \to 0$: tonal / highly structured signal.
The value $0.0077$ indicates a
highly structured signal with clear tonal components.

---
*IA-2026-203-T6 · 2026-07-22 · seed 20260722*
