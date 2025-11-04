import meep as mp
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import PowerNorm
import os

# ----------- Laser Parameters in MEEP Units -----------
lambda0 = 0.8   # Laser wavelength in um
fcen    = 1.0 / lambda0  # Center Frequency (um^-1)
FWHM_fs = 45.0  # Pulse FWHM (fs)
fs_per_unit = 3.3356409519815204  # Convert 1 µm/c to fs
sigma_t_meep = (FWHM_fs/(2.0*math.sqrt(math.log(2.0))))/fs_per_unit
fwidth  = 1.0/(4.0*math.pi*sigma_t_meep)

# ----------- Power and Intensity Calculations -----------
P_avg = 40e-3  # Average power in Watts
rep_rate = 1e3  # Repetition rate in Hz (1 kHz)

E_pulse = P_avg / rep_rate  # Joules per pulse
tau_FWHM = FWHM_fs * 1e-15  # Convert to seconds
tau_pulse = tau_FWHM / (2 * np.sqrt(np.log(2)))  # 1/e width
P_peak = E_pulse / (tau_pulse * np.sqrt(np.pi))

beam_w0 = 56.0  # µm
A_focus = np.pi * (beam_w0 * 1e-6)**2  # Beam area in m²
I_peak = P_peak / A_focus  # W/m²

# Convert to MEEP units
E0_meep = np.sqrt(I_peak / 1e16)  # Empirical scaling factor

print(f"Pulse energy: {E_pulse*1e6:.1f} µJ")
print(f"Peak power: {P_peak*1e-6:.2f} MW")
print(f"Peak intensity: {I_peak*1e-9:.2f} GW/cm²")
print(f"MEEP amplitude: {E0_meep:.3f}\n")

# ----------- Materials -----------
silica   = mp.Medium(index=1.45)
sapphire = mp.Medium(index=1.75)
air      = mp.Medium(index=1.0)

# ----------- Single Sphere Geometry -----------
d = 4.3  # sphere diameter
r = d/2

# Cell size - just enough for one sphere plus boundaries
sx = 3 * d      # Width in x
sy = 3 * d      # Width in y (keeping it 3D even though we'll slice at y=0)
pml = 1.2       # PML thickness
top_air = 4.0   # air above sphere
sapp_thk = 10.0 # sapphire thickness
sz = top_air + 2*r + sapp_thk + 2*pml

cell = mp.Vector3(sx, sy, sz)

# Single sphere at origin
sphere = mp.Sphere(material=silica, center=mp.Vector3(0, 0, 0), radius=r)

# Sapphire substrate
z_top = -r
z_bot = -0.5*sz + pml + 1e-3
substrate = mp.Block(
    material=sapphire,
    size=mp.Vector3(mp.inf, mp.inf, z_top - z_bot),
    center=mp.Vector3(0, 0, 0.5*(z_top + z_bot))
)

geometry = [sphere, substrate]

# ----------- Laser Source -----------
z_src = 0.5*sz - pml - 0.8
focus = mp.Vector3(0, 0, 0)

def make_gaussian_beam(theta_deg):
    """Create angled Gaussian beam source"""
    th = math.radians(theta_deg)
    kdir = mp.Vector3(math.sin(th), 0, -math.cos(th))

    return mp.GaussianBeamSource(
        src=mp.GaussianSource(frequency=fcen, fwidth=fwidth),
        center=mp.Vector3(0, 0, z_src),
        size=mp.Vector3(sx, sy, 0),
        beam_x0=focus - mp.Vector3(0, 0, z_src),
        beam_kdir=kdir,
        beam_w0=beam_w0,
        beam_E0=mp.Vector3(0, E0_meep, 0)  # Ey polarization
    )

# ----------- Boundaries -----------
boundary_layers = [
    mp.PML(pml, direction=mp.Z, side=mp.High),
    mp.PML(pml, direction=mp.Z, side=mp.Low),
    mp.PML(pml, direction=mp.X),
    mp.PML(pml, direction=mp.Y),
]

# ----------- Animation Settings -----------
resolution = 10  # Spatial resolution
angle = 23.5     # Incident angle in degrees
T_RUN = 120      # Total simulation time
fps = 15         # Frames per second for animation
n_frames = 120   # Total number of frames

# Calculate time step between frames
dt_frame = T_RUN / n_frames

print(f"Creating animation with {n_frames} frames at {fps} FPS")
print(f"Incident angle: {angle}°")
print(f"Resolution: {resolution} pixels/µm")
print(f"Time step between frames: {dt_frame:.2f} MEEP time units\n")

# ----------- Field Capture Function -----------
class FieldCapture:
    """Captures field data at specified time intervals"""
    def __init__(self, sim, n_frames, dt_frame):
        self.sim = sim
        self.n_frames = n_frames
        self.dt_frame = dt_frame
        self.frames = []
        self.times = []
        self.frame_count = 0
        self.next_capture_time = 0

    def capture_frame(self):
        """Capture current field state"""
        # Get XZ slice at y=0
        size = mp.Vector3(sx, 0, sz)
        center = mp.Vector3(0, 0, 0)

        Ex = self.sim.get_array(center=center, size=size, component=mp.Ex)
        Ey = self.sim.get_array(center=center, size=size, component=mp.Ey)
        Ez = self.sim.get_array(center=center, size=size, component=mp.Ez)

        # Calculate intensity
        I = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2

        self.frames.append(I)
        self.times.append(self.sim.meep_time())
        self.frame_count += 1

        print(f"  Captured frame {self.frame_count}/{self.n_frames} at t={self.sim.meep_time():.2f}")

    def __call__(self, sim, todo):
        """Called by MEEP at each timestep"""
        if sim.meep_time() >= self.next_capture_time:
            self.capture_frame()
            self.next_capture_time += self.dt_frame

# ----------- Create and Run Simulation -----------
print("[Starting simulation...]")

sim = mp.Simulation(
    cell_size=cell,
    geometry=geometry,
    boundary_layers=boundary_layers,
    resolution=resolution,
    default_material=air,
)

sim.sources = [make_gaussian_beam(angle)]

# Create field capture object
field_capture = FieldCapture(sim, n_frames, dt_frame)

# Run simulation with field capture
sim.run(mp.at_every(dt_frame/2, field_capture), until=T_RUN)

print(f"\n[Simulation complete] Captured {len(field_capture.frames)} frames")

# ----------- Create Animation -----------
print("\n[Generating animation...]")

# Get coordinate arrays
nx, nz = field_capture.frames[0].shape
xs = np.linspace(-0.5*sx, 0.5*sx, nx)
zs = np.linspace(-0.5*sz, 0.5*sz, nz)

# Crop to region of interest
z_top_crop = +3.5
z_bot_crop = -9.0
iz = np.where((zs >= z_bot_crop) & (zs <= z_top_crop))[0]
zs_crop = zs[iz]

# Crop all frames
frames_cropped = [frame[:, iz] for frame in field_capture.frames]

# Determine global intensity range for consistent colormap
all_intensities = np.concatenate([frame.flatten() for frame in frames_cropped])
vmax = np.percentile(all_intensities, 99.5)  # Use 99.5th percentile to avoid outliers
vmin = 0

print(f"Intensity range: {vmin:.2e} to {vmax:.2e}")

# Create figure
fig, ax = plt.subplots(figsize=(12, 8), dpi=120)

# Initial frame
im = ax.imshow(
    np.rot90(frames_cropped[0]),
    extent=[xs[0], xs[-1], zs_crop[0], zs_crop[-1]],
    origin="lower",
    vmin=vmin,
    vmax=vmax,
    cmap="hot",
    aspect="equal",
    interpolation='bilinear'
)

# Draw sphere outline
theta_circle = np.linspace(0, 2*np.pi, 200)
sphere_x = r * np.cos(theta_circle)
sphere_z = r * np.sin(theta_circle)
sphere_outline, = ax.plot(sphere_x, sphere_z, 'cyan', linewidth=2, alpha=0.8)

# Sapphire surface
substrate_line, = ax.plot([xs[0], xs[-1]], [-r, -r],
                          color='#ffaa00', linewidth=2, label='Sapphire')

# Substrate shading
ax.fill_between([xs[0], xs[-1]], [z_bot_crop, z_bot_crop], [-r, -r],
                color='gray', alpha=0.15)

# Labels and formatting
ax.set_xlabel('x (µm)', fontsize=14, fontweight='bold')
ax.set_ylabel('z (µm)', fontsize=14, fontweight='bold')
time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                    fontsize=13, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontweight='bold')

ax.set_xlim(xs[0], xs[-1])
ax.set_ylim(zs_crop[0], zs_crop[-1])
ax.grid(True, color='white', alpha=0.2, linewidth=0.5, linestyle=':')
ax.legend(loc='upper right', fontsize=11)

# Colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Intensity |E|² (MEEP units)', fontsize=12, fontweight='bold')

plt.tight_layout()

# Animation update function
def update(frame_idx):
    """Update animation frame"""
    im.set_array(np.rot90(frames_cropped[frame_idx]))
    time_text.set_text(f'Time: {field_capture.times[frame_idx]:.1f} (MEEP units)\nFrame: {frame_idx+1}/{n_frames}')
    return [im, time_text]

# Create animation
anim = animation.FuncAnimation(
    fig,
    update,
    frames=len(frames_cropped),
    interval=1000/fps,  # milliseconds per frame
    blit=True,
    repeat=True
)

# Save animation
output_file = f'pnj_wavefront_animation_theta{angle:.1f}_res{resolution}.mp4'
print(f"\n[Saving animation to {output_file}...]")
print("(This may take a few minutes...)")

# Try to save as MP4 (requires ffmpeg)
try:
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='MEEP'), bitrate=3000)
    anim.save(output_file, writer=writer)
    print(f"[Success] Animation saved as {output_file}")
except:
    # Fallback: save as GIF
    output_file = output_file.replace('.mp4', '.gif')
    print(f"[FFmpeg not found] Saving as GIF instead: {output_file}")
    anim.save(output_file, writer='pillow', fps=fps)
    print(f"[Success] Animation saved as {output_file}")

plt.close()

# ----------- Also save individual frames as images -----------
print("\n[Saving individual frames...]")
frames_dir = f"frames_theta{angle:.1f}_res{resolution}"
os.makedirs(frames_dir, exist_ok=True)

# Save every Nth frame to avoid too many files
save_every = max(1, n_frames // 20)  # Save ~20 key frames

for i in range(0, len(frames_cropped), save_every):
    fig_frame, ax_frame = plt.subplots(figsize=(12, 8), dpi=150)

    ax_frame.imshow(
        np.rot90(frames_cropped[i]),
        extent=[xs[0], xs[-1], zs_crop[0], zs_crop[-1]],
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        cmap="hot",
        aspect="equal"
    )

    # Sphere and substrate
    ax_frame.plot(sphere_x, sphere_z, 'cyan', linewidth=2, alpha=0.8)
    ax_frame.plot([xs[0], xs[-1]], [-r, -r], color='#ffaa00', linewidth=2)
    ax_frame.fill_between([xs[0], xs[-1]], [z_bot_crop, z_bot_crop], [-r, -r],
                          color='gray', alpha=0.15)

    ax_frame.set_xlabel('x (µm)', fontsize=13, fontweight='bold')
    ax_frame.set_ylabel('z (µm)', fontsize=13, fontweight='bold')
    ax_frame.set_title(f'PNJ Wavefront - t={field_capture.times[i]:.1f} MEEP units',
                      fontsize=14, fontweight='bold')
    ax_frame.grid(True, color='white', alpha=0.2, linewidth=0.5, linestyle=':')

    plt.tight_layout()
    frame_file = f"{frames_dir}/frame_{i:04d}.png"
    plt.savefig(frame_file, dpi=150, bbox_inches='tight')
    plt.close()

print(f"[Success] Saved {len(range(0, len(frames_cropped), save_every))} key frames to {frames_dir}/")

print("\n" + "="*60)
print("ANIMATION COMPLETE!")
print("="*60)
print(f"Video file: {output_file}")
print(f"Frame images: {frames_dir}/")
print(f"\nTo view: open {output_file}")
print("="*60)
