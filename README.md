# Diffusion Equation PINN — Project 7

This project implements a Physics-Informed Neural Network (PINN) to solve the 1D diffusion equation with a time-decaying sinusoidal source. The model enforces the governing PDE, boundary conditions, and initial condition directly in the loss function using automatic differentiation. The project compares the learned PINN solution with the analytical solution.

---

## Problem Statement

We solve the PDE:

\[
u_t = D\,u_{xx} + A e^{-t}\sin\left(\frac{\pi x}{L}\right)
\]

with:

- **Initial condition:** \(u(x,0) = 0\)
- **Boundary conditions:**  
  \[
  u(0,t)=0, \quad u(L,t)=0
  \]

**Domain:**

- \(x \in [0,1]\)
- \(t \in [0,5]\)

---

## PINN Approach

The PINN approximates \(u(x,t)\) using a fully-connected neural network:

- **Architecture:** [2, 64, 64, 64, 1]  
- **Activation:** tanh  
- **Inputs:** (x, t)  
- **Output:** u(x, t)

**Loss components:**

1. PDE residual loss  
2. Boundary condition loss  
3. Initial condition loss  

Automatic differentiation is used to compute:

\[
u_t, \quad u_x, \quad u_{xx}
\]

---

## Repository Structure

```
model.py               # PINN implementation and training script
Ma515_project7.pdf     # Project report
README.md              # This file
```

---

## How to Run

Install dependencies:

```bash
pip install torch numpy matplotlib
```

Run the model:

```bash
python model.py
```

This trains the PINN for 5000 epochs using Adam and plots the analytical vs PINN solution at t = 1.0.

---

## Analytical Solution

The analytical solution used for comparison is:

\[
u(x,t) =
A\sin\left(\frac{\pi x}{L}\right)
\frac{1 - \exp\!\big(-(1 - D(\pi/L)^2)t\big)}
     {1 - D(\pi/L)^2}
\]

---

## Results Summary

- The PINN learns the sinusoidal spatial structure accurately.  
- The amplitude is slightly underestimated due to:
  - Limited collocation points near \(t = 0\)
  - Equal loss weighting
  - No L-BFGS refinement stage

The project report contains full plots, loss curves, and discussion.

---

## Possible Improvements

- Adam → L-BFGS two-stage optimization  
- Increased or adaptive collocation sampling  
- Dynamic loss weighting  
- Fourier feature embeddings  
- Deeper or wider network architectures  

---

## Authors (Group 26 – MA515 Project)

- Mudit Gupta  
- Aryan Arora  
- Divyanshu Sharma  
- Ishant Jindal  
- Garvit Bhalla  
- Srijan Kumar  

