"""
1D heat equation u_t = alpha * u_xx solved with two finite difference schemes:
  FTCS (forward time, centred space): explicit, stable only if r <= 1/2
  BTCS (backward time, centred space): implicit, unconditionally stable
where r = alpha * dt / dx^2.

Run:  python heat_equation.py
Output: snapshots.png, heat_animation.gif, ftcs_unstable.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


# Parameters
L, Nx = 1.0, 50                 # domain length, number of grid points
dx = L / (Nx - 1)
x = np.linspace(0, L, Nx)
alpha = 0.01                    # thermal diffusivity
T_total, dt = 2.0, 0.01         # total time, time step
Nt = int(round(T_total / dt))

u0 = np.exp(-100 * (x - 0.5) ** 2)   # initial condition: heat pulse in the middle
# Boundary values are held fixed at their initial values (Dirichlet conditions).


def solve_ftcs(u0, Nt, r):
    """Explicit FTCS scheme: u_i^{n+1} = u_i^n + r (u_{i+1}^n - 2u_i^n + u_{i-1}^n)."""
    u = u0.copy(); history = [u.copy()]
    for _ in range(Nt):
        u[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
        history.append(u.copy())
    return np.array(history)


def solve_btcs(u0, Nt, r):
    """Implicit BTCS scheme: solve (1 + 2r) u_i - r u_{i-1} - r u_{i+1} = u_i^n each step."""
    u = u0.copy(); history = [u.copy()]
    n = len(u) - 2
    A = diags([-r * np.ones(n - 1), (1 + 2 * r) * np.ones(n), -r * np.ones(n - 1)], [-1, 0, 1]).tocsr()
    for _ in range(Nt):
        b = u[1:-1].copy()
        b[0] += r * u[0]; b[-1] += r * u[-1]     # boundary contributions
        u[1:-1] = spsolve(A, b)
        history.append(u.copy())
    return np.array(history)


r = alpha * dt / dx ** 2
print(f"Stability parameter r = {r:.4f} (FTCS needs r <= 0.5)")
ftcs, btcs = solve_ftcs(u0, Nt, r), solve_btcs(u0, Nt, r)
print(f"Max difference between FTCS and BTCS at t = {T_total}: {np.abs(ftcs[-1] - btcs[-1]).max():.2e}")

# Snapshots at selected times
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for k, tk in enumerate([0, 0.25, 0.5, 1.0, 2.0]):
    n = int(round(tk / dt)); c = plt.cm.viridis(k / 4)
    ax[0].plot(x, ftcs[n], color=c, lw=2, label=f"t = {tk}")
    ax[1].plot(x, btcs[n], color=c, lw=2, label=f"t = {tk}")
ax[0].set_title(f"FTCS (explicit), r = {r:.2f}"); ax[1].set_title(f"BTCS (implicit), r = {r:.2f}")
for a in ax:
    a.set_xlabel("x"); a.grid(alpha=0.3); a.legend(fontsize=9)
ax[0].set_ylabel("Temperature u(x, t)")
plt.tight_layout(); plt.savefig("snapshots.png", dpi=150); plt.close()

# Animation
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
l1, = a1.plot(x, u0, "r-", lw=2, label="FTCS (explicit)")
l2, = a2.plot(x, u0, "b-", lw=2, label="BTCS (implicit)")
for a in (a1, a2):
    a.set_ylim(-0.1, 1.1); a.set_xlabel("x"); a.set_ylabel("Temperature"); a.legend(); a.grid(alpha=0.3)
def update(n):
    l1.set_ydata(ftcs[n]); l2.set_ydata(btcs[n]); a1.set_title(f"t = {n * dt:.2f}")
    return l1, l2
FuncAnimation(fig, update, frames=range(0, Nt + 1, 4), interval=60).save("heat_animation.gif", writer=PillowWriter(fps=15))
plt.close()

# Stability demonstration: a larger time step makes FTCS blow up while BTCS stays smooth
dt_big = 0.03
r_big = alpha * dt_big / dx ** 2
steps = 100
f_big, b_big = solve_ftcs(u0, steps, r_big), solve_btcs(u0, steps, r_big)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(x, f_big[-1], "r-", lw=1.5); ax[0].set_title(f"FTCS, r = {r_big:.2f} > 0.5: unstable")
ax[1].plot(x, b_big[-1], "b-", lw=2); ax[1].set_title(f"BTCS, r = {r_big:.2f}: stable")
for a in ax:
    a.set_xlabel("x"); a.set_ylabel(f"u(x, t = {steps * dt_big:.1f})"); a.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("ftcs_unstable.png", dpi=150); plt.close()
print(f"Stability demo: r = {r_big:.2f}, FTCS max |u| = {np.abs(f_big[-1]).max():.2e}, BTCS max |u| = {np.abs(b_big[-1]).max():.3f}")
print("Figures saved")
