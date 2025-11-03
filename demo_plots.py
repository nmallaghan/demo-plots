import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Time array
time = np.linspace(-1.5, 1.5, 400)

# Symmetric transit (for 3% case)
def symmetric_transit(depth, sigma):
    return 1.0 - depth * np.exp(-0.5 * (time / sigma) ** 2)

# Asymmetric transit controlled by asymmetry_factor
def asymmetric_transit(depth, sigma, asymmetry_factor=2.0):
    """Asymmetric Gaussian transit.
    asymmetry_factor > 1 => slower egress
    asymmetry_factor < 1 => slower ingress
    """
    flux = np.ones_like(time)
    for i, t in enumerate(time):
        if t < 0:
            # Ingress width (shorter for steeper ingress)
            flux[i] = 1.0 - depth * np.exp(-0.5 * (t / sigma) ** 2)
        else:
            # Egress width stretched by asymmetry_factor
            flux[i] = 1.0 - depth * np.exp(-0.5 * (t / (sigma * asymmetry_factor)) ** 2)
    return flux

# Ringed transit (adds substructure to ingress/egress)
def ringed_transit(depth, sigma, asymmetry_factor=2.0, ring_amp=0.01, ring_freq=25):
    """
    Simulate a ringed planet transit with substructure in ingress/egress.

    Parameters
    ----------
    depth : float
        Maximum depth of the transit (planet + rings).
    sigma : float
        Width of the planet’s main Gaussian transit.
    asymmetry_factor : float
        Ratio of egress width to ingress width (>1 => slower egress).
    ring_amp : float
        Amplitude of flux variations due to ring gaps (fractional).
    ring_freq : float
        Frequency of the ring pattern (number of substructures).
    """
    flux = np.ones_like(time)
    for i, t in enumerate(time):
        if t < 0:
            base = 1.0 - depth * np.exp(-0.5 * (t / sigma) ** 2)
            ring_mod = ring_amp * np.sin(ring_freq * t) * np.exp(-abs(t) / 0.3)
            flux[i] = base + ring_mod
        else:
            base = 1.0 - depth * np.exp(-0.5 * (t / (sigma * asymmetry_factor)) ** 2)
            ring_mod = ring_amp * np.sin(ring_freq * t) * np.exp(-abs(t) / 0.3)
            flux[i] = base + ring_mod
    return flux

# Generate light curves
flux_3 = symmetric_transit(0.03, sigma=0.1)
flux_20 = ringed_transit(
    depth=0.20,
    sigma=0.3,
    asymmetry_factor=0.4,
    ring_amp=0.03,   # amplitude of ring modulation
    ring_freq=20     # number of wiggles (ring gaps)
)

# Set up figure
fig, ax = plt.subplots(figsize=(6, 4))
line1, = ax.plot([], [], color='tab:blue', lw=2, label='Giant planet')
line2, = ax.plot([], [], color='tab:red', lw=2, label='Giant Exorings')

ax.set_xlim(time.min(), time.max())
ax.set_ylim(0.75, 1.05)
ax.set_xlabel('Time (arbitrary units)')
ax.set_ylabel('Normalized Flux')
ax.legend(loc='lower center')
ax.set_title('Simulated Real-Time Transit Observation (Ringed)')

# Animation functions
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2

def update(frame):
    line1.set_data(time[:frame], flux_3[:frame])
    line2.set_data(time[:frame], flux_20[:frame])
    return line1, line2

# Create animation
anim = FuncAnimation(
    fig, update, frames=len(time), init_func=init,
    interval=15, blit=True, repeat=False
)

# Save GIF
anim.save('/Users/niamhmallaghan/Documents/transit_ringed.gif', writer=PillowWriter(fps=40))
plt.close()

print("✅ Saved animation as transit_asymmetric.gif")

