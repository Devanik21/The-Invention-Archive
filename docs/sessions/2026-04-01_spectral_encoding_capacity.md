--
session_id: IA-2026-091-T6
date: 2026-04-01
topic: Spectral Encoding Capacity
seed: 20260401
N_signal: 1024
fs_hz: 1000.0
n_components: 5
---

# Invention Archive — Daily Session 2026-04-01

**Session ID:** `IA-2026-091-T6`
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
plus Gaussian noise ($\sigma_n = 0.0573$):

$$x(t) = \sum_{k=1}^{5} A_k \cos(2\pi f_k t + \phi_k) + \eta(t)$$

True components: $f \in \{14.16, 36.08, 74.54, 101.46, 116.70\}$ Hz,
$A \in \{0.856, 0.647, 0.586, 0.480, 0.260\}$.

---

## 3. SpectraNova FFT Decomposition

Frequency resolution: $\Delta f = f_s / N = 0.977$ Hz.

### 3.1 Component Recovery

| $f_{\rm true}$ (Hz) | $A_{\rm true}$ | $f_{\rm det}$ (Hz) | $A_{\rm det}$ | $|f_{\rm err}|$ (Hz) | $C_k$ (bits) |
|---:|---:|---:|---:|---:|---:|
| 14.16 | 0.8562 | 13.67 | 0.5671 | 0.484 | 15.612 |
| 36.08 | 0.6474 | 36.13 | 0.6512 | 0.055 | 16.011 |
| 74.54 | 0.5860 | 74.22 | 0.4814 | 0.325 | 15.139 |
| 101.46 | 0.4802 | 101.56 | 0.4680 | 0.106 | 15.058 |
| 116.70 | 0.2597 | 116.21 | 0.1718 | 0.486 | 12.166 |

### 3.2 System-Level Statistics

| Metric | Value |
|---|---|
| Total signal SNR | 24.36 dB |
| Total FRAE encoding capacity $\sum_k C_k$ | **73.9850 bits** |
| Spectral flatness (Wiener entropy proxy) | 0.005510 |
| Participation ratio (effective components) | 6.95 |
| Noise floor $\sigma_n$ | 0.05734 |

---

## 4. Shannon-Hartley Per-Component Capacity

For each detected component with amplitude $A_k$ in additive white noise
of variance $\sigma_n^2$, the per-component encoding capacity is:

$$C_k = \log_2\!\left(1 + \frac{A_k^2/2}{\sigma_n^2/N}\right) \text{ bits}$$

Total capacity across 5 matched components:
$C_{\rm total} = 73.9850$ bits.

---

## 5. Spectral Flatness

The **Wiener entropy** (spectral flatness measure):

$$\mathrm{SFM} = \frac{\exp\bigl(\langle \ln S(f) \rangle\bigr)}{\langle S(f) \rangle}
  = 0.005510$$

$\mathrm{SFM} \to 1$: white noise (maximally flat).
$\mathrm{SFM} \to 0$: tonal / highly structured signal.
The value $0.0055$ indicates a
highly structured signal with clear tonal components.

---
*IA-2026-091-T6 · 2026-04-01 · seed 20260401*
