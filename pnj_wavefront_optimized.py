import meep as mp
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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

beam_w0 = 56.0  # µm (beam covers many spheres in periodic array)
A_focus = np.pi * (beam_w0 * 1e-6)**2  # Beam area in m²
I_peak = P_peak / A_focus  # W/m²

# Convert to MEEP units
E0_meep = np.sqrt(I_peak / 1e16)  # Empirical scaling factor

print(f"Pulse energy: {E_pulse*1e6:.1f} µJ")
print(f"Peak power: {P_peak*1e-6:.2f} MW")
print(f"Beam waist: {beam_w0:.1f} µm")
print(f"Peak intensity: {I_peak*1e-9:.2f} GW/cm²")
print(f"MEEP amplitude: {E0_meep:.3f}\n")

# ----------- Materials -----------
silica   = mp.Medium(index=1.45)
sapphire = mp.Medium(index=1.75)
air      = mp.Medium(index=1.0)

# ----------- HCP Unit Cell Geometry -----------
d = 4.3  # sphere diameter / pitch
r = d/2

# Rectangular unit cell for hexagonal tiling
sx = d                        # 4.3 µm
sy = math.sqrt(3) * d        # 7.45 µm

# Vertical dimensions - OPTIMIZED
pml_bottom = 1.5   # Thicker PML on bottom to absorb light in substrate
pml_top = 1.2      # PML on top to absorb any back-reflections
top_air = 6.0      # Air above sphere (enough to see incoming beam)
sapp_thk = 6.0     # REDUCED sapphire thickness (was 10.0) - still thick enough but less dispersion
sz = top_air + 2*r + sapp_thk + pml_top + pml_bottom

cell = mp.Vector3(sx, sy, sz)

print(f"Unit cell: {sx:.2f} × {sy:.2f} × {sz:.2f} µm³")
print(f"Sapphire thickness: {sapp_thk:.1f} µm")
print(f"Top air layer: {top_air:.1f} µm")
print(f"PML: top={pml_top:.1f} µm, bottom={pml_bottom:.1f} µm")
print(f"Periodic in X and Y (Bloch boundaries)\n")

# HCP unit cell with two spheres
spheres = [
    mp.Sphere(material=silica, center=mp.Vector3(0, 0, 0), radius=r),
    mp.Sphere(material=silica, center=mp.Vector3(0.5*d, 0.5*sy, 0), radius=r),
]

# Sapphire substrate - positioned relative to sphere
z_top = -r  # Top of sapphire touches bottom of sphere
z_bot = -0.5*sz + pml_bottom + 0.1  # Leave small gap above PML
substrate = mp.Block(
    material=sapphire,
    size=mp.Vector3(mp.inf, mp.inf, z_top - z_bot),
    center=mp.Vector3(0, 0, 0.5*(z_top + z_bot))
)

geometry = spheres + [substrate]

# ----------- Laser Source -----------
z_src = 0.5*sz - pml_top - 0.8
focus = mp.Vector3(0, 0, 0)

print(f"Source position: z = {z_src:.2f} µm")
print(f"Sphere: z = [{-r:.2f}, {r:.2f}] µm")
print(f"Sapphire top: z = {z_top:.2f} µm")
print(f"Sapphire bottom: z = {z_bot:.2f} µm\n")

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

# ----------- Boundaries: Periodic XY + Absorbing Z -----------
boundary_layers = [
    mp.PML(pml_top, direction=mp.Z, side=mp.High),      # Top absorber
    mp.PML(pml_bottom, direction=mp.Z, side=mp.Low),    # Bottom absorber (substrate)
    # X and Y are PERIODIC (Bloch boundaries with k-point)
]

# Bloch k-point matching beam angle
def make_k_point(theta_deg):
    th = math.radians(theta_deg)
    kx = math.sin(th)
    return mp.Vector3(fcen*kx, 0.0, 0.0)

# ----------- Animation Settings -----------
resolution = 16  # HIGH resolution to minimize numerical dispersion in sapphire
angle = 23.5     # Incident angle in degrees
T_RUN = 100      # Total simulation time (slightly reduced since less substrate)
fps = 15         # Frames per second for animation
n_frames = 120   # Total number of frames

# Calculate time step between frames
dt_frame = T_RUN / n_frames

# Courant factor check (for stability in high-index materials)
courant = 0.5  # MEEP default
dt_courant = courant / math.sqrt(1.75**2)  # Check for sapphire (n=1.75)
print(f"Creating animation with {n_frames} frames at {fps} FPS")
print(f"Incident angle: {angle}°")
print(f"Resolution: {resolution} pixels/µm")
print(f"  (High resolution reduces numerical dispersion in sapphire)")
print(f"Bloch k-point: kx = {math.sin(math.radians(angle)) * fcen:.4f}")
print(f"Time step between frames: {dt_frame:.2f} MEEP time units")
print(f"Courant stability factor for sapphire: {dt_courant:.3f}\n")

# ----------- Field Capture Function -----------
def make_field_capture():
    """Factory function to create field capture callback"""
    frames = []
    times = []
    frame_count = [0]

    def capture_fields(sim):
        """Capture current field state"""
        # Get XZ slice at y=0 (through sphere center)
        size = mp.Vector3(sx, 0, sz)
        center = mp.Vector3(0, 0, 0)

        Ex = sim.get_array(center=center, size=size, component=mp.Ex)
        Ey = sim.get_array(center=center, size=size, component=mp.Ey)
        Ez = sim.get_array(center=center, size=size, component=mp.Ez)

        # Calculate intensity
        I = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2

        frames.append(I)
        times.append(sim.meep_time())
        frame_count[0] += 1

        print(f"  Captured frame {frame_count[0]}/{n_frames} at t={sim.meep_time():.2f}")

    capture_fields.frames = frames
    capture_fields.times = times
    capture_fields.frame_count = frame_count

    return capture_fields

# ----------- Create and Run Simulation -----------
print("[Starting simulation with PERIODIC boundaries...]")
print("This may take 10-20 minutes due to high resolution...\n")

sim = mp.Simulation(
    cell_size=cell,
    geometry=geometry,
    boundary_layers=boundary_layers,
    resolution=resolution,
    default_material=air,
    k_point=make_k_point(angle),
    # Force better numerical stability
    force_complex_fields=False,  # Use real fields when possible (faster)
)

sim.sources = [make_gaussian_beam(angle)]

print(f"Simulation setup complete:")
print(f"  Bloch k-point: ({sim.k_point.x:.4f}, {sim.k_point.y:.4f}, {sim.k_point.z:.4f})")
print(f"  Grid points: {int(resolution*sx)} × {int(resolution*sy)} × {int(resolution*sz)}")
print(f"  Total grid points: {int(resolution*sx * resolution*sy * resolution*sz):,}")
print(f"  Periodic array physics with single unit cell view\n")

# Create field capture function
field_capture = make_field_capture()

# Run simulation with field capture
sim.run(mp.at_every(dt_frame, field_capture), until=T_RUN)

print(f"\n[Simulation complete] Captured {len(field_capture.frames)} frames")

# ----------- Create Animation -----------
print("\n[Generating animation...]")

# Get coordinate arrays for single unit cell
nx, nz = field_capture.frames[0].shape
xs = np.linspace(-0.5*sx, 0.5*sx, nx)
zs = np.linspace(-0.5*sz, 0.5*sz, nz)

# Viewing window - show from source down through most of sapphire
z_top_crop = +5.0   # Show incoming beam
z_bot_crop = -5.0   # Show PNJ penetration into sapphire (but not noisy PML region)

iz = np.where((zs >= z_bot_crop) & (zs <= z_top_crop))[0]
zs_crop = zs[iz]

print(f"Viewing window: z ∈ [{z_bot_crop:.1f}, {z_top_crop:.1f}] µm")
print(f"  Sphere: z ∈ [{-r:.1f}, {r:.1f}] µm")
print(f"  Sapphire: z ∈ [{z_top:.1f}, {z_bot:.1f}] µm")
print(f"  (Excluding noisy PML regions)")

# Crop all frames
frames_cropped = [frame[:, iz] for frame in field_capture.frames]

# Determine global intensity range
all_intensities = np.concatenate([frame.flatten() for frame in frames_cropped])
# Use slightly lower percentile to reduce noise visibility
vmax = np.percentile(all_intensities, 99.0)  # Was 99.5, now 99.0
vmin = 0

print(f"Intensity range: {vmin:.2e} to {vmax:.2e}\n")

# Calculate aspect ratio
x_range = xs[-1] - xs[0]
z_range = zs_crop[-1] - zs_crop[0]
aspect_ratio = z_range / x_range
fig_width = 8
fig_height = fig_width * aspect_ratio

# Create figure
fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=130)

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
ax.plot(sphere_x, sphere_z, 'cyan', linewidth=2.5, alpha=0.95, label='Silica sphere')

# Sapphire surface
ax.plot([xs[0], xs[-1]], [-r, -r], color='#ffaa00', linewidth=2.5,
        label='Sapphire surface', linestyle='-')

# Material regions with subtle shading
ax.fill_between([xs[0], xs[-1]], [z_bot_crop, z_bot_crop], [-r, -r],
                color='#d4a574', alpha=0.12, label='Sapphire (n=1.75)')
ax.fill_between([xs[0], xs[-1]], [-r, -r], [z_top_crop, z_top_crop],
                color='lightblue', alpha=0.04)

# Reference lines
ax.axhline(y=0, color='white', linestyle=':', alpha=0.25, linewidth=1)  # Sphere center
ax.axvline(x=0, color='white', linestyle=':', alpha=0.2, linewidth=0.8)  # x=0 axis

# Labels
ax.set_xlabel('x (µm)', fontsize=13, fontweight='bold')
ax.set_ylabel('z (µm)', fontsize=13, fontweight='bold')

# Info box
info_text = (f"θ = {angle}°\n"
             f"Periodic (X,Y)\n"
             f"Res = {resolution} px/µm")
ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
        fontsize=9, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

# Time text
time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.90),
                    fontweight='bold')

ax.set_xlim(xs[0], xs[-1])
ax.set_ylim(zs_crop[0], zs_crop[-1])
ax.grid(True, color='white', alpha=0.12, linewidth=0.4, linestyle=':')
ax.legend(loc='upper left', fontsize=8.5, framealpha=0.88)

ax.set_title(f'PNJ Wavefront - Periodic HCP Array (Unit Cell View)',
             fontsize=12, fontweight='bold', pad=8)

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Intensity |E|²', fontsize=11, fontweight='bold')
cbar.ax.tick_params(labelsize=9)

plt.tight_layout()

# Animation update function
def update(frame_idx):
    """Update animation frame"""
    im.set_array(np.rot90(frames_cropped[frame_idx]))
    time_text.set_text(f't = {field_capture.times[frame_idx]:.1f}\n#{frame_idx+1}/{n_frames}')
    return [im, time_text]

# Create animation
anim = animation.FuncAnimation(
    fig,
    update,
    frames=len(frames_cropped),
    interval=1000/fps,
    blit=True,
    repeat=True
)

# Save animation
output_file = f'pnj_periodic_clean_theta{angle:.1f}_res{resolution}.mp4'
print(f"[Saving animation to {output_file}...]")
print("(This may take a few minutes...)")

# Try to save as MP4
try:
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='MEEP'), bitrate=4000)
    anim.save(output_file, writer=writer)
    print(f"[Success] Animation saved as {output_file}")
except Exception as e:
    # Fallback: save as GIF
    output_file = output_file.replace('.mp4', '.gif')
    print(f"[FFmpeg not available] Saving as GIF: {output_file}")
    anim.save(output_file, writer='pillow', fps=fps)
    print(f"[Success] Animation saved as {output_file}")

plt.close()

# ----------- Save key frames -----------
print("\n[Saving key frames...]")
frames_dir = f"frames_periodic_clean_theta{angle:.1f}"
os.makedirs(frames_dir, exist_ok=True)

save_every = max(1, n_frames // 20)

for i in range(0, len(frames_cropped), save_every):
    fig_frame, ax_frame = plt.subplots(figsize=(fig_width, fig_height), dpi=150)

    ax_frame.imshow(
        np.rot90(frames_cropped[i]),
        extent=[xs[0], xs[-1], zs_crop[0], zs_crop[-1]],
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        cmap="hot",
        aspect="equal",
        interpolation='bilinear'
    )

    # Annotations
    ax_frame.plot(sphere_x, sphere_z, 'cyan', linewidth=2.5, alpha=0.95)
    ax_frame.plot([xs[0], xs[-1]], [-r, -r], color='#ffaa00', linewidth=2.5)
    ax_frame.fill_between([xs[0], xs[-1]], [z_bot_crop, z_bot_crop], [-r, -r],
                          color='#d4a574', alpha=0.12)
    ax_frame.axhline(y=0, color='white', linestyle=':', alpha=0.25, linewidth=1)

    ax_frame.set_xlabel('x (µm)', fontsize=12, fontweight='bold')
    ax_frame.set_ylabel('z (µm)', fontsize=12, fontweight='bold')
    ax_frame.set_title(f'PNJ Periodic - t={field_capture.times[i]:.1f} - θ={angle}°',
                      fontsize=12, fontweight='bold')
    ax_frame.grid(True, color='white', alpha=0.12, linewidth=0.4, linestyle=':')

    plt.tight_layout()
    frame_file = f"{frames_dir}/frame_{i:04d}.png"
    plt.savefig(frame_file, dpi=150, bbox_inches='tight')
    plt.close()

print(f"[Success] Saved {len(range(0, len(frames_cropped), save_every))} frames to {frames_dir}/\n")

print("="*70)
print("OPTIMIZED PERIODIC WAVEFRONT ANIMATION COMPLETE!")
print("="*70)
print(f"Video: {output_file}")
print(f"Frames: {frames_dir}/")
print(f"\nOptimizations applied:")
print(f"  ✓ High resolution ({resolution} px/µm) → reduced numerical dispersion")
print(f"  ✓ Thinner sapphire ({sapp_thk} µm) → less propagation distance")
print(f"  ✓ Stronger bottom PML ({pml_bottom} µm) → better absorption")
print(f"  ✓ Bloch-periodic boundaries → correct array physics")
print(f"  ✓ Viewing window excludes noisy PML regions")
print(f"  ✓ Single unit cell display with periodic coupling")
print("="*70)
