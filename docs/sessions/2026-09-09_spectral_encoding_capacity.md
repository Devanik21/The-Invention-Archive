--
session_id: IA-2026-252-T6
date: 2026-09-09
topic: Spectral Encoding Capacity
seed: 20260909
N_signal: 1024
fs_hz: 1000.0
n_components: 5
---

# Invention Archive — Daily Session 2026-09-09

**Session ID:** `IA-2026-252-T6`
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
plus Gaussian noise ($\sigma_n = 0.0458$):

$$x(t) = \sum_{k=1}^{5} A_k \cos(2\pi f_k t + \phi_k) + \eta(t)$$

True components: $f \in \{18.20, 42.14, 71.35, 114.90, 149.86\}$ Hz,
$A \in \{0.611, 0.323, 0.223, 0.146, 0.145\}$.

---

## 3. SpectraNova FFT Decomposition

Frequency resolution: $\Delta f = f_s / N = 0.977$ Hz.

### 3.1 Component Recovery

| $f_{\rm true}$ (Hz) | $A_{\rm true}$ | $f_{\rm det}$ (Hz) | $A_{\rm det}$ | $|f_{\rm err}|$ (Hz) | $C_k$ (bits) |
|---:|---:|---:|---:|---:|---:|
| 18.20 | 0.6108 | 18.55 | 0.4920 | 0.352 | 15.851 |
| 42.14 | 0.3231 | 41.99 | 0.3183 | 0.147 | 14.595 |
| 71.35 | 0.2232 | 71.29 | 0.2182 | 0.059 | 13.505 |
| 114.90 | 0.1458 | 115.23 | 0.1215 | 0.332 | 11.816 |
| 149.86 | 0.1447 | 149.41 | 0.0964 | 0.442 | 11.149 |

### 3.2 System-Level Statistics

| Metric | Value |
|---|---|
| Total signal SNR | 21.33 dB |
| Total FRAE encoding capacity $\sum_k C_k$ | **66.9148 bits** |
| Spectral flatness (Wiener entropy proxy) | 0.012330 |
| Participation ratio (effective components) | 4.30 |
| Noise floor $\sigma_n$ | 0.04579 |

---

## 4. Shannon-Hartley Per-Component Capacity

For each detected component with amplitude $A_k$ in additive white noise
of variance $\sigma_n^2$, the per-component encoding capacity is:

$$C_k = \log_2\!\left(1 + \frac{A_k^2/2}{\sigma_n^2/N}\right) \text{ bits}$$

Total capacity across 5 matched components:
$C_{\rm total} = 66.9148$ bits.

---

## 5. Spectral Flatness

The **Wiener entropy** (spectral flatness measure):

$$\mathrm{SFM} = \frac{\exp\bigl(\langle \ln S(f) \rangle\bigr)}{\langle S(f) \rangle}
  = 0.012330$$

$\mathrm{SFM} \to 1$: white noise (maximally flat).
$\mathrm{SFM} \to 0$: tonal / highly structured signal.
The value $0.0123$ indicates a
highly structured signal with clear tonal components.

---
*IA-2026-252-T6 · 2026-09-09 · seed 20260909*
