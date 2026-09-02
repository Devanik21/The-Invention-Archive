--
session_id: IA-2026-245-T6
date: 2026-09-02
topic: Spectral Encoding Capacity
seed: 20260902
N_signal: 1024
fs_hz: 1000.0
n_components: 5
---

# Invention Archive — Daily Session 2026-09-02

**Session ID:** `IA-2026-245-T6`
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
plus Gaussian noise ($\sigma_n = 0.0519$):

$$x(t) = \sum_{k=1}^{5} A_k \cos(2\pi f_k t + \phi_k) + \eta(t)$$

True components: $f \in \{29.75, 94.96, 123.42, 138.68, 142.93\}$ Hz,
$A \in \{0.906, 0.733, 0.628, 0.568, 0.291\}$.

---

## 3. SpectraNova FFT Decomposition

Frequency resolution: $\Delta f = f_s / N = 0.977$ Hz.

### 3.1 Component Recovery

| $f_{\rm true}$ (Hz) | $A_{\rm true}$ | $f_{\rm det}$ (Hz) | $A_{\rm det}$ | $|f_{\rm err}|$ (Hz) | $C_k$ (bits) |
|---:|---:|---:|---:|---:|---:|
| 29.75 | 0.9065 | 29.30 | 0.6132 | 0.457 | 16.124 |
| 94.96 | 0.7331 | 94.73 | 0.6635 | 0.233 | 16.352 |
| 123.42 | 0.6281 | 123.05 | 0.4876 | 0.377 | 15.463 |
| 138.68 | 0.5683 | 138.67 | 0.5485 | 0.004 | 15.802 |
| 142.93 | 0.2908 | 142.58 | 0.2343 | 0.347 | 13.348 |

### 3.2 System-Level Statistics

| Metric | Value |
|---|---|
| Total signal SNR | 26.03 dB |
| Total FRAE encoding capacity $\sum_k C_k$ | **77.0890 bits** |
| Spectral flatness (Wiener entropy proxy) | 0.009250 |
| Participation ratio (effective components) | 7.85 |
| Noise floor $\sigma_n$ | 0.05192 |

---

## 4. Shannon-Hartley Per-Component Capacity

For each detected component with amplitude $A_k$ in additive white noise
of variance $\sigma_n^2$, the per-component encoding capacity is:

$$C_k = \log_2\!\left(1 + \frac{A_k^2/2}{\sigma_n^2/N}\right) \text{ bits}$$

Total capacity across 5 matched components:
$C_{\rm total} = 77.0890$ bits.

---

## 5. Spectral Flatness

The **Wiener entropy** (spectral flatness measure):

$$\mathrm{SFM} = \frac{\exp\bigl(\langle \ln S(f) \rangle\bigr)}{\langle S(f) \rangle}
  = 0.009250$$

$\mathrm{SFM} \to 1$: white noise (maximally flat).
$\mathrm{SFM} \to 0$: tonal / highly structured signal.
The value $0.0093$ indicates a
highly structured signal with clear tonal components.

---
*IA-2026-245-T6 · 2026-09-02 · seed 20260902*
