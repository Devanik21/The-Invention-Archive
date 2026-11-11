--
session_id: IA-2026-315-T6
date: 2026-11-11
topic: Spectral Encoding Capacity
seed: 20261111
N_signal: 1024
fs_hz: 1000.0
n_components: 5
---

# Invention Archive — Daily Session 2026-11-11

**Session ID:** `IA-2026-315-T6`
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
plus Gaussian noise ($\sigma_n = 0.0322$):

$$x(t) = \sum_{k=1}^{5} A_k \cos(2\pi f_k t + \phi_k) + \eta(t)$$

True components: $f \in \{12.25, 75.35, 104.49, 126.87, 148.09\}$ Hz,
$A \in \{0.911, 0.443, 0.343, 0.342, 0.230\}$.

---

## 3. SpectraNova FFT Decomposition

Frequency resolution: $\Delta f = f_s / N = 0.977$ Hz.

### 3.1 Component Recovery

| $f_{\rm true}$ (Hz) | $A_{\rm true}$ | $f_{\rm det}$ (Hz) | $A_{\rm det}$ | $|f_{\rm err}|$ (Hz) | $C_k$ (bits) |
|---:|---:|---:|---:|---:|---:|
| 12.25 | 0.9115 | 12.70 | 0.6362 | 0.449 | 17.610 |
| 75.35 | 0.4435 | 75.20 | 0.4337 | 0.151 | 16.504 |
| 104.49 | 0.3427 | 104.49 | 0.3412 | 0.006 | 15.812 |
| 126.87 | 0.3419 | 126.95 | 0.3419 | 0.082 | 15.818 |
| 148.09 | 0.2301 | 148.44 | 0.1868 | 0.343 | 14.074 |

### 3.2 System-Level Statistics

| Metric | Value |
|---|---|
| Total signal SNR | 28.02 dB |
| Total FRAE encoding capacity $\sum_k C_k$ | **79.8179 bits** |
| Spectral flatness (Wiener entropy proxy) | 0.005381 |
| Participation ratio (effective components) | 5.74 |
| Noise floor $\sigma_n$ | 0.03219 |

---

## 4. Shannon-Hartley Per-Component Capacity

For each detected component with amplitude $A_k$ in additive white noise
of variance $\sigma_n^2$, the per-component encoding capacity is:

$$C_k = \log_2\!\left(1 + \frac{A_k^2/2}{\sigma_n^2/N}\right) \text{ bits}$$

Total capacity across 5 matched components:
$C_{\rm total} = 79.8179$ bits.

---

## 5. Spectral Flatness

The **Wiener entropy** (spectral flatness measure):

$$\mathrm{SFM} = \frac{\exp\bigl(\langle \ln S(f) \rangle\bigr)}{\langle S(f) \rangle}
  = 0.005381$$

$\mathrm{SFM} \to 1$: white noise (maximally flat).
$\mathrm{SFM} \to 0$: tonal / highly structured signal.
The value $0.0054$ indicates a
highly structured signal with clear tonal components.

---
*IA-2026-315-T6 · 2026-11-11 · seed 20261111*
