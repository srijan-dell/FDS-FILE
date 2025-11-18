# Diffusion Equation PINN — Project 7

This project implements a Physics-Informed Neural Network (PINN) to solve the 1D diffusion equation with a time-decaying sinusoidal source. The model enforces the PDE, boundary conditions, and initial condition directly in the loss function through automatic differentiation and compares the PINN solution with the analytical solution.

---

## Problem Statement

We solve the PDE:

$$
u_t = D\u_{xx} + A e^{-t}\sin\!\left(\frac{\pi x}{L}\right)
$$

with:

- **Initial condition:**

$$
u(x,0) = 0
$$

- **Boundary conditions:**

$$
u(0,t) = 0, \quad u(L,t) = 0
$$

**Domain:**

- \( x \in [0,1] \)  
- \( t \in [0,5] \)

---

## PINN Approach

The PINN approximates the solution \( u(x,t) \) using a fully-connected neural network:

- Architecture: **[2, 64, 64, 64, 1]**
- Activation: **tanh**
- Inputs: **(x, t)**
- Output: **u(x, t)**

**Loss function components:**

1. PDE residual  
2. Boundary condition enforcement  
3. Initial condition enforcement  

Automatic differentiation is used to compute:

$$
u_t,\; u_x,\; u_{xx}
$$

---

## Repository Structure

```
model.py               # PINN model and training script
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

This trains the PINN for 5000 epochs using Adam and plots the analytical vs PINN solution at \( t = 1.0 \).

---

## Analytical Solution

The analytical reference solution is:
$$
u(x,t)= A \sin\!\left(\frac{\pi x}{L}\right)\frac{1 - \exp\!\Bigl(-(1 - D(\tfrac{\pi}{L})^{2})\,t\Bigr)}{1 - D(\tfrac{\pi}{L})^{2}}
$$



---

## Results Summary

- The PINN learns the spatial sinusoidal structure well.  
- The amplitude is slightly underestimated due to:
  - Limited collocation points near \( t = 0 \)
  - Equal weighting of PDE/BC/IC losses
  - Absence of an L-BFGS refinement stage

Further plots, error analysis, and discussions are in the project PDF.

---

## Possible Improvements

- Two-stage optimization (Adam → L-BFGS)  
- Adaptive collocation point sampling  
- Dynamic or weighted loss balancing  
- Fourier feature embeddings  
- Deeper or wider neural network  

---

## Authors (Group 26 — MA515 Project)

- Mudit Gupta  
- Aryan Arora  
- Divyanshu Sharma  
- Ishant Jindal  
- Garvit Bhalla  
- Srijan Kumar  
