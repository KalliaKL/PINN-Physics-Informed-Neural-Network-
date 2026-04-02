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
