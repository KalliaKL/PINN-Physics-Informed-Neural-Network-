"""
Physics-Informed Neural Network: Simple Harmonic Oscillator
============================================================
The physics:
    d²x/dt² + ω²x = 0

    A mass on a spring. If you displace it and let go, it oscillates.
    The exact solution is:
        x(t) = A * cos(ωt) + B * sin(ωt)

    With initial conditions x(0) = 1, dx/dt(0) = 0:
        x(t) = cos(ωt)

The idea:
    Instead of telling the network the solution, we tell it the *law*.
    The network learns a function x(t) that satisfies:
        1. The differential equation everywhere
        2. The initial conditions at t=0

    PyTorch's autograd computes dx/dt and d²x/dt² automatically.
    We use those derivatives directly as part of the loss function.

Run with:
    pip install torch numpy matplotlib
    python pinn_harmonic_oscillator.py

I built a neural network that learns to satisfy a differential equation without
being given the solution. The physics law, d²x/dt² + ω²x = 0, is encoded
directly into the loss function. I used Tanh instead of ReLU
because I need a smooth second derivative to compute the acceleration term.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# STEP 1: The neural network
# =============================================================================
# This is just a standard feedforward network.
# Input:  t (time), shape (N, 1)
# Output: x (position), shape (N, 1)
#
# Nothing special here yet. The physics comes in the loss function.

class PINN(nn.Module):
    def __init__(self, hidden_size=32, n_layers=4):
        super().__init__()

        # Build layers: input(1) -> hidden -> hidden -> ... -> output(1)
        layers = []
        in_size = 1
        for _ in range(n_layers):
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.Tanh())   # Tanh works better than ReLU here.
            in_size = hidden_size      # Why: ReLU has zero second derivative almost
        layers.append(nn.Linear(hidden_size, 1))  # everywhere, which breaks the physics loss.

        self.net = nn.Sequential(*layers)

    def forward(self, t):
        return self.net(t)


# =============================================================================
# STEP 2: Computing derivatives with autograd
# =============================================================================
# This is the key PyTorch concept.
# torch.autograd.grad computes exact derivatives of the network output
# with respect to its input. No finite differences, no approximation.
# The gradient flows through the network weights automatically.

def derivatives(model, t):
    """
    Given a model and time points t, returns:
        x      - position predicted by network
        dx_dt  - first derivative  (velocity)
        d2x_dt2 - second derivative (acceleration)
    """
    # requires_grad=True tells PyTorch: track operations on this tensor
    # so we can differentiate through them later.
    t = t.requires_grad_(True)

    x = model(t)

    # First derivative: dx/dt
    # create_graph=True means we can differentiate *again* to get d²x/dt²
    dx_dt = torch.autograd.grad(
        outputs=x,
        inputs=t,
        grad_outputs=torch.ones_like(x),
        create_graph=True
    )[0]

    # Second derivative: d²x/dt²
    d2x_dt2 = torch.autograd.grad(
        outputs=dx_dt,
        inputs=t,
        grad_outputs=torch.ones_like(dx_dt),
        create_graph=True
    )[0]

    return x, dx_dt, d2x_dt2


# =============================================================================
# STEP 3: The loss function
# =============================================================================
# This is where the physics lives.
# The total loss has three parts:
#
#   L_physics = (d²x/dt² + ω²x)²     averaged over many time points
#               ^^^^^^^^^^^^^^^^^^
#               This should be zero everywhere if the ODE is satisfied.
#
#   L_ic_pos  = (x(0) - 1)²          initial position = 1
#
#   L_ic_vel  = (dx/dt(0) - 0)²      initial velocity = 0
#
# The network has no idea what the answer is at the start.
# Gradient descent pushes it toward satisfying all three simultaneously.

def loss_fn(model, t_physics, t0, omega=2.0):
    """
    t_physics : random time points where we enforce the ODE  (N, 1)
    t0        : the single point t=0 for initial conditions   (1, 1)
    omega     : angular frequency ω
    """

    # --- Physics loss ---
    x, _, d2x_dt2 = derivatives(model, t_physics)
    residual = d2x_dt2 + omega**2 * x      # should be zero
    L_physics = torch.mean(residual**2)

    # --- Initial condition loss ---
    x0, dx0_dt, _ = derivatives(model, t0)
    L_ic_pos = (x0 - 1.0)**2              # x(0) = 1
    L_ic_vel = (dx0_dt - 0.0)**2          # dx/dt(0) = 0

    # Total loss: sum of all three terms
    # You could weight them differently (e.g. 10 * L_ic) if one dominates.
    return L_physics + L_ic_pos + L_ic_vel


# =============================================================================
# STEP 4: Training
# =============================================================================

def train():
    omega = 2.0          # angular frequency
    T = 2 * np.pi        # one full period

    model = PINN(hidden_size=32, n_layers=4)

    # Adam is standard. Learning rate 1e-3 is a safe default for PINNs.
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    # t=0 for initial conditions
    t0 = torch.tensor([[0.0]], requires_grad=True)

    print(f"Training PINN for x'' + {omega}²x = 0,  x(0)=1,  x'(0)=0")
    print(f"Exact solution: x(t) = cos({omega}t)\n")

    losses = []
    for epoch in range(5000):

        # Sample fresh random time points each epoch.
        # This is important: we are not training on a fixed dataset,
        # we are enforcing the ODE at randomly sampled collocation points.
        t_physics = torch.FloatTensor(200, 1).uniform_(0, T)

        optimiser.zero_grad()           # clear gradients from last step
        loss = loss_fn(model, t_physics, t0, omega)
        loss.backward()                 # compute gradients
        optimiser.step()                # update weights

        losses.append(loss.item())
        if epoch % 500 == 0:
            print(f"  Epoch {epoch:5d}  |  Loss: {loss.item():.6f}")

    print("\nTraining complete.\n")
    return model, losses, omega, T


# =============================================================================
# STEP 5: Evaluate and plot
# =============================================================================

def evaluate(model, omega, T):
    t_test = torch.linspace(0, T, 500).reshape(-1, 1)

    with torch.no_grad():
        x_pred = model(t_test).numpy().flatten()

    t_np = t_test.numpy().flatten()
    x_exact = np.cos(omega * t_np)

    mae = np.mean(np.abs(x_pred - x_exact))
    print(f"Mean absolute error vs exact solution: {mae:.6f}")

    plt.figure(figsize=(9, 4))
    plt.plot(t_np, x_exact, 'k-',  linewidth=2,   label='Exact: cos(ωt)')
    plt.plot(t_np, x_pred,  'r--', linewidth=1.5, label='PINN prediction')
    plt.xlabel('Time t')
    plt.ylabel('Position x(t)')
    plt.title('Physics-Informed Neural Network: Simple Harmonic Oscillator')
    plt.legend()
    plt.tight_layout()
    plt.savefig('pinn_result.png', dpi=150)
    plt.show()
    print("Plot saved to pinn_result.png")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    model, losses, omega, T = train()
    evaluate(model, omega, T)

    # Plot training loss
    plt.figure(figsize=(7, 3))
    plt.semilogy(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Training loss')
    plt.tight_layout()
    plt.savefig('pinn_loss.png', dpi=150)
    plt.show()
