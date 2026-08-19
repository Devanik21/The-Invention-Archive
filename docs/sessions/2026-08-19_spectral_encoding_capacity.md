--
session_id: IA-2026-231-T6
date: 2026-08-19
topic: Spectral Encoding Capacity
seed: 20260819
N_signal: 1024
fs_hz: 1000.0
n_components: 5
---

# Invention Archive — Daily Session 2026-08-19

**Session ID:** `IA-2026-231-T6`
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
plus Gaussian noise ($\sigma_n = 0.0482$):

$$x(t) = \sum_{k=1}^{5} A_k \cos(2\pi f_k t + \phi_k) + \eta(t)$$

True components: $f \in \{57.29, 73.05, 98.89, 108.73, 116.66\}$ Hz,
$A \in \{0.978, 0.819, 0.724, 0.596, 0.566\}$.

---

## 3. SpectraNova FFT Decomposition

Frequency resolution: $\Delta f = f_s / N = 0.977$ Hz.

### 3.1 Component Recovery

| $f_{\rm true}$ (Hz) | $A_{\rm true}$ | $f_{\rm det}$ (Hz) | $A_{\rm det}$ | $|f_{\rm err}|$ (Hz) | $C_k$ (bits) |
|---:|---:|---:|---:|---:|---:|
| 57.29 | 0.9785 | 57.62 | 0.8144 | 0.331 | 17.156 |
| 73.05 | 0.8192 | 73.24 | 0.7468 | 0.189 | 16.906 |
| 98.89 | 0.7239 | 98.63 | 0.6381 | 0.258 | 16.452 |
| 108.73 | 0.5963 | 108.40 | 0.4803 | 0.336 | 15.632 |
| 116.66 | 0.5665 | 116.21 | 0.4073 | 0.449 | 15.156 |

### 3.2 System-Level Statistics

| Metric | Value |
|---|---|
| Total signal SNR | 27.84 dB |
| Total FRAE encoding capacity $\sum_k C_k$ | **81.3020 bits** |
| Spectral flatness (Wiener entropy proxy) | 0.003395 |
| Participation ratio (effective components) | 7.52 |
| Noise floor $\sigma_n$ | 0.04822 |

---

## 4. Shannon-Hartley Per-Component Capacity

For each detected component with amplitude $A_k$ in additive white noise
of variance $\sigma_n^2$, the per-component encoding capacity is:

$$C_k = \log_2\!\left(1 + \frac{A_k^2/2}{\sigma_n^2/N}\right) \text{ bits}$$

Total capacity across 5 matched components:
$C_{\rm total} = 81.3020$ bits.

---

## 5. Spectral Flatness

The **Wiener entropy** (spectral flatness measure):

$$\mathrm{SFM} = \frac{\exp\bigl(\langle \ln S(f) \rangle\bigr)}{\langle S(f) \rangle}
  = 0.003395$$

$\mathrm{SFM} \to 1$: white noise (maximally flat).
$\mathrm{SFM} \to 0$: tonal / highly structured signal.
The value $0.0034$ indicates a
highly structured signal with clear tonal components.

---
*IA-2026-231-T6 · 2026-08-19 · seed 20260819*
