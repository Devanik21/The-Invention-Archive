--
session_id: IA-2026-189-T6
date: 2026-07-08
topic: Spectral Encoding Capacity
seed: 20260708
N_signal: 1024
fs_hz: 1000.0
n_components: 5
---

# Invention Archive — Daily Session 2026-07-08

**Session ID:** `IA-2026-189-T6`
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
plus Gaussian noise ($\sigma_n = 0.0554$):

$$x(t) = \sum_{k=1}^{5} A_k \cos(2\pi f_k t + \phi_k) + \eta(t)$$

True components: $f \in \{12.87, 73.67, 101.25, 115.46, 117.13\}$ Hz,
$A \in \{0.923, 0.822, 0.688, 0.595, 0.198\}$.

---

## 3. SpectraNova FFT Decomposition

Frequency resolution: $\Delta f = f_s / N = 0.977$ Hz.

### 3.1 Component Recovery

| $f_{\rm true}$ (Hz) | $A_{\rm true}$ | $f_{\rm det}$ (Hz) | $A_{\rm det}$ | $|f_{\rm err}|$ (Hz) | $C_k$ (bits) |
|---:|---:|---:|---:|---:|---:|
| 12.87 | 0.9231 | 12.70 | 0.8733 | 0.176 | 16.959 |
| 73.67 | 0.8224 | 73.24 | 0.5765 | 0.433 | 15.761 |
| 101.25 | 0.6879 | 101.56 | 0.5774 | 0.310 | 15.765 |
| 115.46 | 0.5946 | 115.23 | 0.5445 | 0.224 | 15.596 |

### 3.2 System-Level Statistics

| Metric | Value |
|---|---|
| Total signal SNR | 25.92 dB |
| Total FRAE encoding capacity $\sum_k C_k$ | **64.0812 bits** |
| Spectral flatness (Wiener entropy proxy) | 0.005442 |
| Participation ratio (effective components) | 6.01 |
| Noise floor $\sigma_n$ | 0.05536 |

---

## 4. Shannon-Hartley Per-Component Capacity

For each detected component with amplitude $A_k$ in additive white noise
of variance $\sigma_n^2$, the per-component encoding capacity is:

$$C_k = \log_2\!\left(1 + \frac{A_k^2/2}{\sigma_n^2/N}\right) \text{ bits}$$

Total capacity across 4 matched components:
$C_{\rm total} = 64.0812$ bits.

---

## 5. Spectral Flatness

The **Wiener entropy** (spectral flatness measure):

$$\mathrm{SFM} = \frac{\exp\bigl(\langle \ln S(f) \rangle\bigr)}{\langle S(f) \rangle}
  = 0.005442$$

$\mathrm{SFM} \to 1$: white noise (maximally flat).
$\mathrm{SFM} \to 0$: tonal / highly structured signal.
The value $0.0054$ indicates a
highly structured signal with clear tonal components.

---
*IA-2026-189-T6 · 2026-07-08 · seed 20260708*
