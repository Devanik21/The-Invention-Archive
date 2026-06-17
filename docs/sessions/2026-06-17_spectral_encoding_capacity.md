--
session_id: IA-2026-168-T6
date: 2026-06-17
topic: Spectral Encoding Capacity
seed: 20260617
N_signal: 1024
fs_hz: 1000.0
n_components: 5
---

# Invention Archive — Daily Session 2026-06-17

**Session ID:** `IA-2026-168-T6`
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
plus Gaussian noise ($\sigma_n = 0.0457$):

$$x(t) = \sum_{k=1}^{5} A_k \cos(2\pi f_k t + \phi_k) + \eta(t)$$

True components: $f \in \{10.23, 28.87, 37.00, 50.63, 100.23\}$ Hz,
$A \in \{0.945, 0.912, 0.889, 0.848, 0.468\}$.

---

## 3. SpectraNova FFT Decomposition

Frequency resolution: $\Delta f = f_s / N = 0.977$ Hz.

### 3.1 Component Recovery

| $f_{\rm true}$ (Hz) | $A_{\rm true}$ | $f_{\rm det}$ (Hz) | $A_{\rm det}$ | $|f_{\rm err}|$ (Hz) | $C_k$ (bits) |
|---:|---:|---:|---:|---:|---:|
| 10.23 | 0.9448 | 9.77 | 0.6376 | 0.461 | 16.603 |
| 28.87 | 0.9116 | 29.30 | 0.6582 | 0.422 | 16.695 |
| 37.00 | 0.8894 | 37.11 | 0.8889 | 0.104 | 17.562 |
| 50.63 | 0.8480 | 50.78 | 0.7962 | 0.153 | 17.244 |
| 100.23 | 0.4680 | 100.59 | 0.3689 | 0.358 | 15.024 |

### 3.2 System-Level Statistics

| Metric | Value |
|---|---|
| Total signal SNR | 29.17 dB |
| Total FRAE encoding capacity $\sum_k C_k$ | **83.1283 bits** |
| Spectral flatness (Wiener entropy proxy) | 0.002945 |
| Participation ratio (effective components) | 7.59 |
| Noise floor $\sigma_n$ | 0.04572 |

---

## 4. Shannon-Hartley Per-Component Capacity

For each detected component with amplitude $A_k$ in additive white noise
of variance $\sigma_n^2$, the per-component encoding capacity is:

$$C_k = \log_2\!\left(1 + \frac{A_k^2/2}{\sigma_n^2/N}\right) \text{ bits}$$

Total capacity across 5 matched components:
$C_{\rm total} = 83.1283$ bits.

---

## 5. Spectral Flatness

The **Wiener entropy** (spectral flatness measure):

$$\mathrm{SFM} = \frac{\exp\bigl(\langle \ln S(f) \rangle\bigr)}{\langle S(f) \rangle}
  = 0.002945$$

$\mathrm{SFM} \to 1$: white noise (maximally flat).
$\mathrm{SFM} \to 0$: tonal / highly structured signal.
The value $0.0029$ indicates a
highly structured signal with clear tonal components.

---
*IA-2026-168-T6 · 2026-06-17 · seed 20260617*
