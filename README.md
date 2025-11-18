
# Diffusion Equation with Time-Decaying Source — PINN Project

**Repository:** Physics-Informed Neural Network (PINN) for the diffusion PDE with a time-decaying sinusoidal source.  
**Course / Project:** MA515 — Project 7 (Foundation of Data Science)

---

## Overview

This project implements a Physics-Informed Neural Network (PINN) to solve the PDE:

\[
u_t = D\,u_{xx} + A e^{-t} \sin\!\left(\frac{\pi x}{L}\right)
\]

with boundary conditions \(u(0,t)=u(L,t)=0\) and initial condition \(u(x,0)=0\).  
Parameters used in the implementation:

- Diffusion coefficient: `D = 0.1`
- Source amplitude: `A = 1.0`
- Domain length: `L = 1.0`

The network architecture, training setup and analytical solution are included in `model.py` and the project report `Ma515_project7.pdf`.

---

## Repository structure

```
.
├── model.py           # PINN implementation, training loop and plotting
├── Ma515_project7.pdf # Project report (theory, methodology, results)
└── README.md          # (this file)
```

---

## Requirements

Recommended Python environment (tested conceptually):

- Python 3.8+
- PyTorch (cpu or cuda) — e.g. `pip install torch`
- numpy
- matplotlib

You can create a virtual environment and install packages:

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows

pip install --upgrade pip
pip install torch numpy matplotlib
```

> Note: Installing `torch` may require choosing a package that matches your OS and CUDA version. See https://pytorch.org/ for platform-specific commands.

---

## How to run

The repository includes `model.py`, which runs training and produces a comparison plot between the trained PINN and the analytical solution for `t = 1.0`.

To run:

```bash
python model.py
```

What the script does:

- Builds a small fully-connected neural network with Tanh activations: layers `[2, 64, 64, 64, 1]`.
- Samples collocation points inside the domain to enforce the PDE residual using automatic differentiation.
- Enforces boundary and initial conditions via dedicated loss terms.
- Trains with Adam (`lr=1e-3`) for 5000 epochs (loss is printed every 500 epochs).
- After training it computes and plots the analytical solution vs the PINN output at `t = 1.0`.

---

## Analytical solution

The closed-form analytical solution used for comparison is:

\[
u(x,t) = A\sin\!\left(\frac{\pi x}{L}\right)
\frac{1 - \exp\big(-\big(1 - D(\pi/L)^2\big)t\big)}{1 - D(\pi/L)^2}.
\]

This is computed inside `model.py` for verification.

---

## Notes & Tips for improving accuracy

If the PINN underestimates amplitude or has accuracy issues (commonly observed):

- Increase the number of collocation points (sample more `x_f` / `t_f`).
- Train for more epochs, or use a second-phase optimizer (L-BFGS) after Adam.
- Add loss weighting to balance PDE residual against BC/IC losses.
- Introduce Fourier features to help the network represent oscillatory solutions.
- Consider architecture changes (wider or deeper network) or different activation functions (Tanh generally works well for PINNs).

---

## Results (summary)

- Network learns the spatial sine shape (sin(πx/L)) well but may underestimate amplitude with the baseline settings.
- Loss typically decreases steadily with Adam; using L-BFGS as a fine-tuning stage improves match to the analytical solution.

Refer to `Ma515_project7.pdf` for plots, training curves, discussion and suggested future work.

---

## License

This repository is provided for educational purposes. You can add a license file if you want to open-source it (e.g., MIT License). Example:

```
MIT License
```

---

## Contact

If you want the README customized (add badges, CI instructions, requirements.txt, Dockerfile, or GitHub Actions workflow), tell me what you'd like and I will update the README accordingly.

---

**Files included**
- `model.py` — PINN code (training + plotting).
- `Ma515_project7.pdf` — Project report and methodology.

