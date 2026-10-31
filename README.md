# Audio-Reactive Phase-Space Engine: Emergent Geometry Toy Model

![3D Spacetime Trajectory Animation](true_fft_universe_3d.gif)

This repository houses a self-contained, audio-reactive Python simulation engine. It acts as a pre-geometric toy model, utilizing non-commutative matrix operator fields to explore how macroscopic, 3D network topologies and gravitational warping analogies can emerge from background-independent information frameworks.

## Core Software Architecture

The Python pipeline (`fourier_spacetime_engine.py`) takes any local `.mp3` audio track as an input stream and processes it through a three-tiered simulation layer:

1. **Digital Signal Processing (FFT):** The engine decodes MP3 files via `miniaudio` and executes a Real Fast Fourier Transform (RFFT) to extract localized frequency voltage arrays (Bass, Mids, Highs) frame-by-frame.
2. **Phase-Space Matrix Dynamics:** Frequency components are injected as structural defects into coordinate matrices ($X_1, X_2$). The script resolves their physical evolution using a Symplectic Leapfrog Integrator paired with an algebraic force-clamping and dissipative velocity damping loop to maintain mathematical stability.
3. **Graph Network Geometry:** Real-time diagonal parameters from the Position ($X$) and Momentum ($P$) operators drive the visual $X, Y, Z$ positions of a `networkx` graph. The system calculates geometric scaling properties via a native unweighted edge curvature scale and logs frame-by-frame telemetry directly to a spreadsheet (`matrix_energy_metrics.csv`).

## Telemetry Database Output

Every run automatically generates a localized metrics log tracking the system behavior across the audio timeline:
* **Potential Energy (PE):** Calculated from the non-commutative matrix action commutator trace.
* **Kinetic Energy (KE):** Calculated from the momentum matrix trace flux.
* **Mean Curvature Scale:** Approximated unweighted edge structural metrics.

## Quick Start Setup

This framework is optimized to run out-of-the-box on stable Python 3.12+ environments without requiring external C-compilers.

```powershell
# Setup and activate your environment
python -m venv stable_quantum_env
.\stable_quantum_env\Scripts\Activate.ps1

# Install required mathematical and signal processing libraries
pip install torch networkx numpy matplotlib miniaudio
```

## Running Playback

Ensure your target audio track is verified at your local path, and launch the file compiler pass:

```powershell
python fourier_spacetime_engine.py
```
