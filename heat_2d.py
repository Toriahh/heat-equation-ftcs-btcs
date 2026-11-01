"""
2D heat equation u_t = alpha (u_xx + u_yy) on the unit square, u = 0 on the boundary,
u(x, y, 0) = sin(pi x) sin(pi y), alpha = 1.
Exact solution: u = exp(-2 pi^2 alpha t) sin(pi x) sin(pi y).

Compares the explicit Forward Euler (FTCS) scheme with the implicit Crank-Nicolson scheme:
solution evolution, centreline comparison, error surfaces, a grid refinement (convergence)
study and the stability of each scheme.

Run:  python heat_2d.py      (the convergence study takes 1 to 2 minutes)
"""



import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from matplotlib import cm
import os

SAVE_DIR = "."

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "figure.dpi":     110,    # on-screen size
    "savefig.dpi":    150,    # resolution of the saved PNG
    "savefig.bbox":   "tight",
})
CMAP = cm.viridis

ALPHA = 1.0      # thermal diffusivity
L     = 1.0      # side of the square domain
T_END = 0.1      # final time


def exact(X, Y, t, alpha=ALPHA):
    """Exact solution: exp(-2 pi^2 alpha t) sin(pi x) sin(pi y)."""
    return np.exp(-2.0*np.pi**2*alpha*t)*np.sin(np.pi*X)*np.sin(np.pi*Y)


def grid(N):
    """Uniform (N+1)x(N+1) grid on the unit square."""
    x = np.linspace(0.0, L, N+1)
    y = np.linspace(0.0, L, N+1)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return x, y, X, Y


def laplacian(N):
    """Five point discrete Laplacian on the (N-1)^2 interior nodes."""
    m   = N - 1
    T   = sp.diags([1.0, -4.0, 1.0], [-1, 0, 1], shape=(m, m), format="csr")
    I   = sp.identity(m, format="csr")
    Off = sp.diags([1.0, 1.0], [-1, 1], shape=(m, m), format="csr")
    return (sp.kron(I, T) + sp.kron(Off, I)).tocsr()


def explicit(N, dt, T_final, alpha=ALPHA):
    """Explicit Forward Euler (FTCS). Stable only when r <= 1/4."""
    x, y, X, Y = grid(N)
    h = L/N
    r = alpha*dt/h**2
    u = exact(X, Y, 0.0)
    u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0.0
    nt = int(round(T_final/dt))
    for _ in range(nt):
        un = u.copy()
        u[1:-1, 1:-1] = ((1.0 - 4.0*r)*un[1:-1, 1:-1]
                         + r*(un[2:, 1:-1] + un[:-2, 1:-1]
                              + un[1:-1, 2:] + un[1:-1, :-2]))
        u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0.0
    return X, Y, u, r, nt


def crank_nicolson(N, dt, T_final, alpha=ALPHA):
    """Implicit Crank-Nicolson. Unconditionally stable."""
    x, y, X, Y = grid(N)
    h = L/N
    r = alpha*dt/h**2
    m = N - 1
    Lap = laplacian(N)
    I   = sp.identity(m*m, format="csr")
    A1  = (I - (r/2.0)*Lap).tocsc()
    A2  = (I + (r/2.0)*Lap).tocsr()
    solve = spla.factorized(A1)          # factorise once, outside the loop
    u0 = exact(X, Y, 0.0)
    v  = u0[1:-1, 1:-1].reshape(-1, order="F").copy()
    nt = int(round(T_final/dt))
    for _ in range(nt):
        v = solve(A2.dot(v))
    u = np.zeros_like(u0)
    u[1:-1, 1:-1] = v.reshape((m, m), order="F")
    return X, Y, u, r, nt


def maxerr(X, Y, u, t):
    """Maximum absolute error over the interior nodes."""
    return np.max(np.abs(u[1:-1, 1:-1] - exact(X, Y, t)[1:-1, 1:-1]))


def surf(ax, X, Y, Z, title, zlabel="$u$"):
    """Standard 3D surface panel."""
    ax.plot_surface(X, Y, Z, cmap=CMAP, linewidth=0,
                    antialiased=True, rstride=1, cstride=1)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.view_init(elev=26, azim=-135)




# ---------------------------------------------------------------------
# FIGURE 4.1 -- initial condition and exact solution at t = 0.1
# ---------------------------------------------------------------------
N = 40                                    # grid resolution for the plot

x, y, X, Y = grid(N)

fig = plt.figure(figsize=(9, 3.8))

ax1 = fig.add_subplot(121, projection="3d")
surf(ax1, X, Y, exact(X, Y, 0.0), "(a) Initial condition, $t=0$")
ax1.set_zlim(0, 1)

ax2 = fig.add_subplot(122, projection="3d")
surf(ax2, X, Y, exact(X, Y, T_END), "(b) Exact solution, $t=0.1$")
ax2.set_zlim(0, 1)

plt.savefig(f"{SAVE_DIR}/heat2d_fig1_exact.png")
plt.close()

print("Peak amplitude at t = 0.1 :  %.6f" % np.exp(-2*np.pi**2*ALPHA*T_END))

# ---------------------------------------------------------------------
# FIGURE 4.2 -- explicit scheme, solution at four times
# ---------------------------------------------------------------------
N     = 40                                # grid resolution
h     = 1.0/N
dt    = 0.2*h**2                          # r = 0.2, inside the stability limit
times = [0.0, 0.01, 0.03, 0.1]            # times to display

fig = plt.figure(figsize=(9, 7.2))
for k, t in enumerate(times):
    if t == 0.0:
        x, y, X, Y = grid(N)
        u = exact(X, Y, 0.0)
    else:
        X, Y, u, r, nt = explicit(N, dt, t)
    ax = fig.add_subplot(2, 2, k+1, projection="3d")
    surf(ax, X, Y, u, f"({chr(97+k)}) $t={t}$")
    ax.set_zlim(0, 1)

plt.savefig(f"{SAVE_DIR}/heat2d_fig2_explicit_evolution.png")
plt.close()

print("N = %d,  h = %.6f,  dt = %.4e,  r = %.3f"
      % (N, h, dt, ALPHA*dt/h**2))

# ---------------------------------------------------------------------
# FIGURE 4.3 -- Crank-Nicolson scheme, solution at four times
# ---------------------------------------------------------------------
N     = 40                                # grid resolution
h     = 1.0/N
dt    = h/10.0                            # r = 4.0, well beyond explicit limit
times = [0.0, 0.01, 0.03, 0.1]

fig = plt.figure(figsize=(9, 7.2))
for k, t in enumerate(times):
    x, y, X, Y = grid(N)
    if t == 0.0:
        u = exact(X, Y, 0.0)
    else:
        X, Y, u, r, nt = crank_nicolson(N, dt, t)
    ax = fig.add_subplot(2, 2, k+1, projection="3d")
    surf(ax, X, Y, u, f"({chr(97+k)}) $t={t}$")
    ax.set_zlim(0, 1)

plt.savefig(f"{SAVE_DIR}/heat2d_fig3_cn_evolution.png")
plt.close()

print("N = %d,  h = %.6f,  dt = %.4e,  r = %.3f"
      % (N, h, dt, ALPHA*dt/h**2))
print("Explicit limit is r = 0.25, so this is %.0f times that."
      % ((ALPHA*dt/h**2)/0.25))

# ---------------------------------------------------------------------
# FIGURE 4.4 -- centreline profile at y = 0.5, t = 0.1
# ---------------------------------------------------------------------
N   = 20                                  # coarse grid, so error is visible
h   = 1.0/N
mid = N//2                                # index of the line y = 0.5

X, Y, u_exp, r_exp, _ = explicit(N, 0.2*h**2, T_END)
X, Y, u_cn,  r_cn,  _ = crank_nicolson(N, h/10.0, T_END)

xs   = np.linspace(0, 1, N+1)             # grid points
xf   = np.linspace(0, 1, 400)             # fine grid for the exact curve
u_ex = np.exp(-2*np.pi**2*ALPHA*T_END)*np.sin(np.pi*xf)*np.sin(np.pi*0.5)

fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.plot(xf, u_ex, "k-", lw=1.6, label="Exact")
ax.plot(xs, u_exp[:, mid], "o", ms=5, mfc="none", mec="C0",
        label=f"Explicit ($r={r_exp:.1f}$)")
ax.plot(xs, u_cn[:, mid], "s", ms=5, mfc="none", mec="C3",
        label=f"Crank--Nicolson ($r={r_cn:.1f}$)")

ax.set_xlabel("$x$")
ax.set_ylabel("$u(x,\\,0.5,\\,0.1)$")
ax.set_title(f"Centreline profile at $y=0.5$, $t=0.1$, $N={N}$")
ax.legend(frameon=False)
ax.grid(alpha=0.3)

plt.savefig(f"{SAVE_DIR}/heat2d_fig4_centreline.png")
plt.close()

print("max error, explicit       : %.6e" % maxerr(X, Y, u_exp, T_END))
print("max error, Crank-Nicolson : %.6e" % maxerr(X, Y, u_cn,  T_END))

# ---------------------------------------------------------------------
# FIGURE 4.5 -- absolute error surfaces at t = 0.1
# ---------------------------------------------------------------------
N = 40                                    # grid resolution
h = 1.0/N

X, Y, u_exp, _, _ = explicit(N, 0.2*h**2, T_END)
X, Y, u_cn,  _, _ = crank_nicolson(N, h/10.0, T_END)

E_exp = np.abs(u_exp - exact(X, Y, T_END))
E_cn  = np.abs(u_cn  - exact(X, Y, T_END))

fig = plt.figure(figsize=(9, 3.8))

ax1 = fig.add_subplot(121, projection="3d")
surf(ax1, X, Y, E_exp, "(a) Explicit scheme", "$|$error$|$")

ax2 = fig.add_subplot(122, projection="3d")
surf(ax2, X, Y, E_cn, "(b) Crank--Nicolson scheme", "$|$error$|$")

plt.savefig(f"{SAVE_DIR}/heat2d_fig5_error_surfaces.png")
plt.close()

print("peak error, explicit       : %.6e" % E_exp.max())
print("peak error, Crank-Nicolson : %.6e" % E_cn.max())
print("ratio                      : %.4f" % (E_exp.max()/E_cn.max()))

# ---------------------------------------------------------------------
# FIGURE 4.6 -- grid refinement, log-log error plot
# ---------------------------------------------------------------------
Ns = [8, 16, 32, 64, 128]                 # remove 128 for a faster run

hs, E_exp, E_cn = [], [], []
for N in Ns:
    h = 1.0/N
    X, Y, u, _, _ = explicit(N, 0.2*h**2, T_END)      # r = 0.2 held fixed
    E_exp.append(maxerr(X, Y, u, T_END))
    X, Y, u, _, _ = crank_nicolson(N, h/10.0, T_END)  # dt proportional to h
    E_cn.append(maxerr(X, Y, u, T_END))
    hs.append(h)
    print("N = %4d done" % N)

hs, E_exp, E_cn = np.array(hs), np.array(E_exp), np.array(E_cn)

fig, ax = plt.subplots(figsize=(6.2, 4.6))
ax.loglog(hs, E_exp, "o-", color="C0", label="Explicit ($r=0.2$)")
ax.loglog(hs, E_cn,  "s-", color="C3",
          label="Crank--Nicolson ($\\Delta t=h/10$)")
ax.loglog(hs, E_exp[0]*(hs/hs[0])**2, "k--", lw=1, label="Slope 2 reference")

ax.set_xlabel("$h$")
ax.set_ylabel("$E_\\infty(h)$")
ax.set_title("Grid refinement at $t=0.1$")
ax.legend(frameon=False)
ax.grid(alpha=0.3, which="both")

plt.savefig(f"{SAVE_DIR}/heat2d_fig6_convergence.png")
plt.close()

# ---- observed orders of convergence: p = log2( E(h) / E(h/2) ) ----
print("\nObserved orders of convergence")
print("   N      h        E_explicit       p        E_CN             p")
for k, N in enumerate(Ns):
    if k == 0:
        print("%4d  %8.6f  %.6e     n/a      %.6e     n/a"
              % (N, hs[k], E_exp[k], E_cn[k]))
    else:
        pe = np.log2(E_exp[k-1]/E_exp[k])
        pc = np.log2(E_cn[k-1]/E_cn[k])
        print("%4d  %8.6f  %.6e  %7.4f  %.6e  %7.4f"
              % (N, hs[k], E_exp[k], pe, E_cn[k], pc))

print("\nError constants  E / h^2  (should settle to a constant):")
for k, N in enumerate(Ns):
    print("  N = %4d :  explicit %.5f    CN %.5f    ratio %.4f"
          % (N, E_exp[k]/hs[k]**2, E_cn[k]/hs[k]**2, E_exp[k]/E_cn[k]))

# ---------------------------------------------------------------------
# FIGURE 4.7 -- explicit scheme either side of the stability limit
# ---------------------------------------------------------------------
N     = 16                                # grid resolution
h     = 1.0/N
NSTEP = 400                               # number of time levels to plot

fig, ax = plt.subplots(figsize=(6.4, 4.6))

for r, c in [(0.20, "C0"), (0.25, "C2"), (0.30, "C3")]:
    x, y, X, Y = grid(N)
    u = exact(X, Y, 0.0)
    u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0.0

    hist = []
    for n in range(NSTEP+1):
        hist.append(np.max(np.abs(u)))
        un = u.copy()
        u[1:-1, 1:-1] = ((1.0 - 4.0*r)*un[1:-1, 1:-1]
                         + r*(un[2:, 1:-1] + un[:-2, 1:-1]
                              + un[1:-1, 2:] + un[1:-1, :-2]))
        u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0.0

    ax.semilogy(hist, "-", color=c, lw=1.4, label=f"$r={r:.2f}$")

ax.axhline(1.0, color="k", lw=0.8, ls=":")
ax.set_xlabel("Time level $n$")
ax.set_ylabel(r"$\max_{i,j}|u_{i,j}^{\,n}|$")
ax.set_title(f"Explicit scheme, $N={N}$: behaviour either side of $r=1/4$")
ax.legend(frameon=False)
ax.grid(alpha=0.3, which="both")
ax.set_ylim(1e-6, 1e40)

plt.savefig(f"{SAVE_DIR}/heat2d_fig7_stability.png")
plt.close()

# ---- theory check: the exact discrete threshold on this grid ----
s = np.sin((N-1)*np.pi/(2*N))**2       # highest representable grid mode
print("\nHighest grid mode:  sin^2(beta h / 2) = %.6f" % s)
print("Discrete threshold: r* = 1/(4 sin^2)  = %.6f" % (1.0/(4*s)))
print("Predicted |G| at r = 0.30            : %.6f" % abs(1 - 8*0.30*s))
print("(the measured growth per step in the unstable run matches this)")

# ---------------------------------------------------------------------
# FIGURE 4.8 -- Crank-Nicolson stability at large mesh ratio
# ---------------------------------------------------------------------
N  = 20                                   # grid resolution
h  = 1.0/N
TF = 0.2                                  # final time for this figure
m  = N - 1

Lap = laplacian(N)
I   = sp.identity(m*m, format="csr")

fig, ax = plt.subplots(figsize=(6.4, 4.6))

# exact decay envelope
tt = np.linspace(0, TF, 300)
ax.semilogy(tt, np.exp(-2*np.pi**2*ALPHA*tt), "k-", lw=1.6,
            label="Exact decay")

# three time steps, giving r = 2, 10 and 40
for dt, c, mk in [(h/10.0, "C0", "o"), (h/2.0, "C2", "s"), (2.0*h, "C3", "^")]:
    r  = ALPHA*dt/h**2
    A1 = (I - (r/2.0)*Lap).tocsc()
    A2 = (I + (r/2.0)*Lap).tocsr()
    solve = spla.factorized(A1)

    x, y, X, Y = grid(N)
    v = exact(X, Y, 0.0)[1:-1, 1:-1].reshape(-1, order="F").copy()

    nt = int(round(TF/dt))
    ts, vals = [], []
    for n in range(nt+1):
        ts.append(n*dt)
        vals.append(np.max(np.abs(v)))
        v = solve(A2.dot(v))

    ax.semilogy(ts, vals, mk, color=c, ms=4, mfc="none",
                label=f"$r={r:.1f}$")

ax.set_xlabel("$t$")
ax.set_ylabel(r"$\max_{i,j}|u_{i,j}^{\,n}|$")
ax.set_title(f"Crank--Nicolson, $N={N}$: stability at large mesh ratio")
ax.legend(frameon=False)
ax.grid(alpha=0.3, which="both")

plt.savefig(f"{SAVE_DIR}/heat2d_fig8_cn_stability.png")
plt.close()

print("Explicit stability limit is r = 0.25,")
print("so r = 40 is 160 times that, and the scheme is still bounded.")