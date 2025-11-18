"""
train_pinn.py
Physics-Informed Neural Network for:
u_t = D u_xx + A e^{-t} sin(pi x / L)

Loads fixed dataset from pinn_dataset.npz (generated separately).
Trains with Adam + L-BFGS for high accuracy.
Plots:
- PINN vs Analytical snapshots
- Error heatmap
- L2 error vs time
- Animation of time evolution
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import time

torch.set_default_dtype(torch.float32)

# ===============================
# Constants
# ===============================
D = 0.1
A = 1.0
L = 1.0
device = torch.device("cpu")   # change to "cuda" if you installed CUDA torch

# ===============================
# Load Dataset
# ===============================
data = np.load("pinn_dataset.npz")

x_f = torch.tensor(data["x_f"], dtype=torch.float32, device=device)
t_f = torch.tensor(data["t_f"], dtype=torch.float32, device=device)

x_b = torch.tensor(data["x_b"], dtype=torch.float32, device=device)
t_b = torch.tensor(data["t_b"], dtype=torch.float32, device=device)

x_i = torch.tensor(data["x_i"], dtype=torch.float32, device=device)
t_i = torch.tensor(data["t_i"], dtype=torch.float32, device=device)


# ===============================
# Analytical Solution (torch + numpy)
# ===============================
def analytical_np(x, t):
    denom = 1.0 - D * (math.pi / L)**2
    return A * np.sin(math.pi * x / L) * (1 - np.exp(-(1 - D*(math.pi/L)**2)*t)) / denom

def analytical_torch(x, t):
    denom = 1.0 - D * (torch.pi / L)**2
    return A * torch.sin(torch.pi * x / L) * (1 - torch.exp(-(1 - D*(torch.pi/L)**2)*t)) / denom


# ===============================
# PINN Model
# ===============================
class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        net_layers = []
        for i in range(len(layers)-1):
            net_layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers)-2:
                net_layers.append(nn.Tanh())
        self.net = nn.Sequential(*net_layers)

        # Xavier initialization
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        return self.net(torch.cat([x,t], dim=1))


layers = [2, 128, 128, 128, 64, 1]
model = PINN(layers).to(device)

print("Total trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))


# ===============================
# Loss Functions
# ===============================
def pde_residual(model, x, t):
    x_req = x.clone().detach().requires_grad_(True)
    t_req = t.clone().detach().requires_grad_(True)

    u = model(x_req, t_req)

    u_t = torch.autograd.grad(u, t_req, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x_req, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_req, torch.ones_like(u_x), create_graph=True)[0]

    source = A * torch.exp(-t_req) * torch.sin(torch.pi * x_req / L)

    return u_t - D*u_xx - source

def boundary_loss(model):
    return torch.mean(model(x_b, t_b)**2)

def initial_loss(model):
    return torch.mean(model(x_i, t_i)**2)


# Loss weights
λ_pde = 1.0
λ_bc  = 20.0
λ_ic  = 20.0

# ===============================
# Training: Adam → L-BFGS
# ===============================
opt_adam = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 3000

print("Starting Adam training...")

for epoch in range(1, epochs+1):
    opt_adam.zero_grad()

    loss_pde = torch.mean(pde_residual(model, x_f, t_f)**2)
    loss_bc  = boundary_loss(model)
    loss_ic  = initial_loss(model)

    loss = λ_pde*loss_pde + λ_bc*loss_bc + λ_ic*loss_ic
    loss.backward()
    opt_adam.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item():.6e}")

# L-BFGS refinement
print("Starting L-BFGS refinement...")

optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(),
    max_iter=500,
    max_eval=500,
    history_size=50,
    tolerance_grad=1e-9,
    tolerance_change=1e-9,
    line_search_fn="strong_wolfe"
)

def closure():
    optimizer_lbfgs.zero_grad()

    loss_pde = torch.mean(pde_residual(model, x_f, t_f)**2)
    loss_bc  = boundary_loss(model)
    loss_ic  = initial_loss(model)

    loss = λ_pde*loss_pde + λ_bc*loss_bc + λ_ic*loss_ic
    loss.backward()

    return loss

optimizer_lbfgs.step(closure)
print("L-BFGS completed.")


# ===============================
# Evaluation & Plots
# ===============================
model.eval()

# --- Snapshot Comparisons ---
times = [0.1, 0.5, 1, 2, 5]
x_plot = np.linspace(0, L, 300)
x_t = torch.tensor(x_plot.reshape(-1,1), dtype=torch.float32, device=device)

plt.figure(figsize=(10,6))
for tval in times:
    t_t = torch.ones_like(x_t) * tval
    u_pred = model(x_t, t_t).detach().cpu().numpy().flatten()
    u_true = analytical_np(x_plot, tval)

    plt.plot(x_plot, u_true, label=f"Analytical t={tval}")
    plt.plot(x_plot, u_pred, '--', label=f"PINN t={tval}")

plt.title("PINN vs Analytical Solution - Time Snapshots")
plt.xlabel("x"); plt.ylabel("u(x,t)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# --- Error Heatmap ---
nx, nt = 200, 200
xg = np.linspace(0, L, nx)
tg = np.linspace(0, 5, nt)
X, T = np.meshgrid(xg, tg)

with torch.no_grad():
    xs = torch.tensor(X.reshape(-1,1), dtype=torch.float32, device=device)
    ts = torch.tensor(T.reshape(-1,1), dtype=torch.float32, device=device)
    u_pred = model(xs, ts).cpu().numpy().reshape(nt, nx)

u_true = analytical_np(X, T)
err = np.abs(u_pred - u_true)

plt.figure(figsize=(9,5))
plt.pcolormesh(xg, tg, err, shading="auto", cmap="viridis")
plt.colorbar(label="Absolute Error")
plt.xlabel("x"); plt.ylabel("t")
plt.title("Error Heatmap |u_pred - u_true|")
plt.tight_layout()
plt.show()


# --- L2 Error vs Time ---
l2_err = np.sqrt(np.mean((u_pred - u_true)**2, axis=1))
plt.figure(figsize=(8,4))
plt.plot(tg, l2_err)
plt.xlabel("t"); plt.ylabel("L2 Error")
plt.title("L2 Error vs Time")
plt.grid()
plt.tight_layout()
plt.show()


# --- Animation ---
fig, ax = plt.subplots(figsize=(10,5))
line_pinn, = ax.plot([], [], '--', label="PINN")
line_true, = ax.plot([], [], label="Analytical")

ax.set_xlim(0, L)
ax.set_ylim(-0.1, 1.2 * A/(1-D*(math.pi/L)**2))
ax.set_xlabel("x"); ax.set_ylabel("u(x,t)")
ax.legend()

time_vals = np.linspace(0,5,150)
x_anim = x_plot
x_t_anim = torch.tensor(x_anim.reshape(-1,1), dtype=torch.float32, device=device)

def update(frame):
    tval = time_vals[frame]
    t_tensor = torch.ones_like(x_t_anim) * tval
    with torch.no_grad():
        u_p = model(x_t_anim, t_tensor).cpu().numpy().flatten()
    u_t = analytical_np(x_anim, tval)

    line_pinn.set_data(x_anim, u_p)
    line_true.set_data(x_anim, u_t)
    ax.set_title(f"Time Evolution t = {tval:.2f}")
    return line_pinn, line_true

anim = FuncAnimation(fig, update, frames=len(time_vals), interval=60)
plt.show()

# Save model
torch.save(model.state_dict(), "pinn_trained_model.pth")
print("Model saved to pinn_trained_model.pth")
