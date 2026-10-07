--
session_id: IA-2026-280-T6
date: 2026-10-07
topic: Spectral Encoding Capacity
seed: 20261007
N_signal: 1024
fs_hz: 1000.0
n_components: 5
---

# Invention Archive — Daily Session 2026-10-07

**Session ID:** `IA-2026-280-T6`
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
plus Gaussian noise ($\sigma_n = 0.0633$):

$$x(t) = \sum_{k=1}^{5} A_k \cos(2\pi f_k t + \phi_k) + \eta(t)$$

True components: $f \in \{32.24, 51.50, 76.95, 136.62, 146.23\}$ Hz,
$A \in \{0.899, 0.486, 0.343, 0.317, 0.187\}$.

---

## 3. SpectraNova FFT Decomposition

Frequency resolution: $\Delta f = f_s / N = 0.977$ Hz.

### 3.1 Component Recovery

| $f_{\rm true}$ (Hz) | $A_{\rm true}$ | $f_{\rm det}$ (Hz) | $A_{\rm det}$ | $|f_{\rm err}|$ (Hz) | $C_k$ (bits) |
|---:|---:|---:|---:|---:|---:|
| 32.24 | 0.8987 | 32.23 | 0.8959 | 0.015 | 16.645 |
| 51.50 | 0.4861 | 51.76 | 0.4337 | 0.261 | 14.551 |
| 76.95 | 0.3427 | 77.15 | 0.3133 | 0.203 | 13.613 |
| 136.62 | 0.3169 | 136.72 | 0.3071 | 0.102 | 13.556 |
| 146.23 | 0.1866 | 146.48 | 0.1652 | 0.258 | 11.766 |

### 3.2 System-Level Statistics

| Metric | Value |
|---|---|
| Total signal SNR | 22.09 dB |
| Total FRAE encoding capacity $\sum_k C_k$ | **70.1314 bits** |
| Spectral flatness (Wiener entropy proxy) | 0.007641 |
| Participation ratio (effective components) | 2.40 |
| Noise floor $\sigma_n$ | 0.06333 |

---

## 4. Shannon-Hartley Per-Component Capacity

For each detected component with amplitude $A_k$ in additive white noise
of variance $\sigma_n^2$, the per-component encoding capacity is:

$$C_k = \log_2\!\left(1 + \frac{A_k^2/2}{\sigma_n^2/N}\right) \text{ bits}$$

Total capacity across 5 matched components:
$C_{\rm total} = 70.1314$ bits.

---

## 5. Spectral Flatness

The **Wiener entropy** (spectral flatness measure):

$$\mathrm{SFM} = \frac{\exp\bigl(\langle \ln S(f) \rangle\bigr)}{\langle S(f) \rangle}
  = 0.007641$$

$\mathrm{SFM} \to 1$: white noise (maximally flat).
$\mathrm{SFM} \to 0$: tonal / highly structured signal.
The value $0.0076$ indicates a
highly structured signal with clear tonal components.

---
*IA-2026-280-T6 · 2026-10-07 · seed 20261007*
