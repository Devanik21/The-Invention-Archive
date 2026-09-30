--
session_id: IA-2026-273-T6
date: 2026-09-30
topic: Spectral Encoding Capacity
seed: 20260930
N_signal: 1024
fs_hz: 1000.0
n_components: 5
---

# Invention Archive — Daily Session 2026-09-30

**Session ID:** `IA-2026-273-T6`
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
plus Gaussian noise ($\sigma_n = 0.0358$):

$$x(t) = \sum_{k=1}^{5} A_k \cos(2\pi f_k t + \phi_k) + \eta(t)$$

True components: $f \in \{26.12, 47.54, 51.46, 56.89, 64.35\}$ Hz,
$A \in \{0.832, 0.773, 0.723, 0.634, 0.449\}$.

---

## 3. SpectraNova FFT Decomposition

Frequency resolution: $\Delta f = f_s / N = 0.977$ Hz.

### 3.1 Component Recovery

| $f_{\rm true}$ (Hz) | $A_{\rm true}$ | $f_{\rm det}$ (Hz) | $A_{\rm det}$ | $|f_{\rm err}|$ (Hz) | $C_k$ (bits) |
|---:|---:|---:|---:|---:|---:|
| 26.12 | 0.8318 | 26.37 | 0.7519 | 0.250 | 17.783 |
| 47.54 | 0.7728 | 47.85 | 0.5863 | 0.309 | 17.065 |
| 51.46 | 0.7228 | 51.76 | 0.6320 | 0.299 | 17.282 |
| 56.89 | 0.6337 | 56.64 | 0.5169 | 0.249 | 16.702 |
| 64.35 | 0.4487 | 64.45 | 0.4671 | 0.100 | 16.409 |

### 3.2 System-Level Statistics

| Metric | Value |
|---|---|
| Total signal SNR | 29.73 dB |
| Total FRAE encoding capacity $\sum_k C_k$ | **85.2411 bits** |
| Spectral flatness (Wiener entropy proxy) | 0.008639 |
| Participation ratio (effective components) | 7.51 |
| Noise floor $\sigma_n$ | 0.03583 |

---

## 4. Shannon-Hartley Per-Component Capacity

For each detected component with amplitude $A_k$ in additive white noise
of variance $\sigma_n^2$, the per-component encoding capacity is:

$$C_k = \log_2\!\left(1 + \frac{A_k^2/2}{\sigma_n^2/N}\right) \text{ bits}$$

Total capacity across 5 matched components:
$C_{\rm total} = 85.2411$ bits.

---

## 5. Spectral Flatness

The **Wiener entropy** (spectral flatness measure):

$$\mathrm{SFM} = \frac{\exp\bigl(\langle \ln S(f) \rangle\bigr)}{\langle S(f) \rangle}
  = 0.008639$$

$\mathrm{SFM} \to 1$: white noise (maximally flat).
$\mathrm{SFM} \to 0$: tonal / highly structured signal.
The value $0.0086$ indicates a
highly structured signal with clear tonal components.

---
*IA-2026-273-T6 · 2026-09-30 · seed 20260930*
