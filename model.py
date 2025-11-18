import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Constants
D = 0.1      # Diffusion coefficient
A = 1.0      # Source amplitude
L = 1.0      # Domain length

class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.net = nn.Sequential()
        for i in range(len(layers)-1):
            self.net.add_module(f"layer_{i}", nn.Linear(layers[i], layers[i+1]))
            if i < len(layers)-2:
                self.net.add_module(f"tanh_{i}", nn.Tanh())

    def forward(self, x, t):
        inputs = torch.cat((x, t), dim=1)
        return self.net(inputs)

def pde_residual(model, x, t):
    x.requires_grad = True
    t.requires_grad = True
    u = model(x, t)

    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    source = A * torch.exp(-t) * torch.sin(np.pi * x / L)
    residual = u_t - D * u_xx - source
    return residual

def boundary_loss(model, t):
    x0 = torch.zeros_like(t)
    xL = L * torch.ones_like(t)
    u0 = model(x0, t)
    uL = model(xL, t)
    return torch.mean(u0**2) + torch.mean(uL**2)

def initial_loss(model, x):
    t0 = torch.zeros_like(x)
    u0 = model(x, t0)
    return torch.mean(u0**2)

layers = [2, 64, 64, 64, 1]
model = PINN(layers)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5000):
    try:
        optimizer.zero_grad()

        # Random points in domain
        x_f = torch.rand(100,1) * L
        t_f = torch.rand(100,1) * 5  # up to t=5

        # Boundary/Initial points
        x_b = torch.rand(100,1) * L
        t_b = torch.rand(100,1) * 5

        # Compute losses
        loss_pde = torch.mean(pde_residual(model, x_f, t_f)**2)
        loss_bc = boundary_loss(model, t_b)
        loss_ic = initial_loss(model, x_b)

        loss = loss_pde + loss_bc + loss_ic

        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    except Exception as e:
        print("\n=== ERROR OCCURRED ===")
        print(f"Epoch: {epoch}")
        print("Error type:", type(e).__name__)
        print("Error message:", e)
        print("======================\n")
        raise   # re-throws to stop program unless you remove this line


def analytical_solution(x, t):
    return A * np.sin(np.pi*x/L) * (1 - np.exp(-(1 - D*(np.pi/L)**2)*t)) / (1 - D*(np.pi/L)**2)

x_test = np.linspace(0, L, 100)
t_test = 1.0
x_tensor = torch.tensor(x_test.reshape(-1,1), dtype=torch.float32)
t_tensor = torch.ones_like(x_tensor) * t_test

u_pred = model(x_tensor, t_tensor).detach().numpy().flatten()
u_true = analytical_solution(x_test, t_test)

plt.plot(x_test, u_true, label='Analytical')
plt.plot(x_test, u_pred, '--', label='PINN')
plt.legend(); plt.xlabel('x'); plt.ylabel('u(x, t)')
plt.title(f"Comparison at t={t_test}")
plt.show()
