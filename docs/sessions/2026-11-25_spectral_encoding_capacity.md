--
session_id: IA-2026-329-T6
date: 2026-11-25
topic: Spectral Encoding Capacity
seed: 20261125
N_signal: 1024
fs_hz: 1000.0
n_components: 5
---

# Invention Archive — Daily Session 2026-11-25

**Session ID:** `IA-2026-329-T6`
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
plus Gaussian noise ($\sigma_n = 0.0303$):

$$x(t) = \sum_{k=1}^{5} A_k \cos(2\pi f_k t + \phi_k) + \eta(t)$$

True components: $f \in \{8.47, 36.83, 48.49, 55.19, 103.79\}$ Hz,
$A \in \{0.807, 0.802, 0.550, 0.254, 0.111\}$.

---

## 3. SpectraNova FFT Decomposition

Frequency resolution: $\Delta f = f_s / N = 0.977$ Hz.

### 3.1 Component Recovery

| $f_{\rm true}$ (Hz) | $A_{\rm true}$ | $f_{\rm det}$ (Hz) | $A_{\rm det}$ | $|f_{\rm err}|$ (Hz) | $C_k$ (bits) |
|---:|---:|---:|---:|---:|---:|
| 8.47 | 0.8068 | 8.79 | 0.6768 | 0.323 | 17.958 |
| 36.83 | 0.8023 | 37.11 | 0.6923 | 0.280 | 18.024 |
| 48.49 | 0.5502 | 48.83 | 0.4689 | 0.333 | 16.899 |
| 55.19 | 0.2535 | 55.66 | 0.1873 | 0.477 | 14.251 |
| 103.79 | 0.1105 | 103.52 | 0.1048 | 0.278 | 12.577 |

### 3.2 System-Level Statistics

| Metric | Value |
|---|---|
| Total signal SNR | 29.58 dB |
| Total FRAE encoding capacity $\sum_k C_k$ | **79.7088 bits** |
| Spectral flatness (Wiener entropy proxy) | 0.011535 |
| Participation ratio (effective components) | 5.58 |
| Noise floor $\sigma_n$ | 0.03035 |

---

## 4. Shannon-Hartley Per-Component Capacity

For each detected component with amplitude $A_k$ in additive white noise
of variance $\sigma_n^2$, the per-component encoding capacity is:

$$C_k = \log_2\!\left(1 + \frac{A_k^2/2}{\sigma_n^2/N}\right) \text{ bits}$$

Total capacity across 5 matched components:
$C_{\rm total} = 79.7088$ bits.

---

## 5. Spectral Flatness

The **Wiener entropy** (spectral flatness measure):

$$\mathrm{SFM} = \frac{\exp\bigl(\langle \ln S(f) \rangle\bigr)}{\langle S(f) \rangle}
  = 0.011535$$

$\mathrm{SFM} \to 1$: white noise (maximally flat).
$\mathrm{SFM} \to 0$: tonal / highly structured signal.
The value $0.0115$ indicates a
highly structured signal with clear tonal components.

---
*IA-2026-329-T6 · 2026-11-25 · seed 20261125*
