
# PINN for Diffusion Equation with Time-Decaying Source

This repository solves the PDE using a Physics-Informed Neural Network (PINN):

$$
u_t = D\,u_{xx} + A e^{-t} \sin\!\left(\frac{\pi x}{L}\right)
$$

with boundary conditions:

$$
u(0,t)=0,\qquad u(L,t)=0
$$

and initial condition:

$$
u(x,0)=0.
$$

---

## **Analytical Solution**

The closed‑form analytical solution is:

$$
u(x,t)
= A \sin\!\left(\frac{\pi x}{L}\right)
\frac{1 - \exp\!\Big(-\big(1 - D(\pi/L)^2\big)t\Big)}
{1 - D(\pi/L)^2}.
$$

---

## **Files in this repository**

```
model.py               # PINN implementation
Ma515_project7.pdf     # Full report
README.md              # This file
```

---

## **How to run**

```bash
python model.py
```

---

## **Requirements**

- Python 3.8+
- PyTorch  
- numpy  
- matplotlib  

---

## **Notes**

GitHub *does not render LaTeX equations natively*, but it still displays the raw Markdown cleanly.  
If you want GitHub to render equations visually, you must enable a documentation site (GitHub Pages) with MathJax.

