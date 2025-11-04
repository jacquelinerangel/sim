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

beam_w0 = 56.0  # µm (back to original - beam covers many spheres in array)
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

# ----------- HCP Unit Cell Geometry (SAME AS ANGLE COMPARISON CODE) -----------
d = 4.3  # sphere diameter / pitch
r = d/2

# Rectangular unit cell for hexagonal tiling
sx = d                        # 4.3 µm (one sphere per row)
sy = math.sqrt(3) * d        # 7.45 µm
pml = 1.2       # PML thickness (only in Z!)
top_air = 8.0   # Air above sphere (increased to see incoming beam)
sapp_thk = 10.0 # Sapphire thickness
sz = top_air + 2*r + sapp_thk + 2*pml

cell = mp.Vector3(sx, sy, sz)

print(f"Unit cell: {sx:.2f} × {sy:.2f} × {sz:.2f} µm³")
print(f"Periodic in X and Y (Bloch boundaries)")
print(f"PML only in Z direction\n")

# HCP unit cell with two spheres (one at origin for y=0 slice)
spheres = [
    mp.Sphere(material=silica, center=mp.Vector3(0, 0, 0), radius=r),
    mp.Sphere(material=silica, center=mp.Vector3(0.5*d, 0.5*sy, 0), radius=r),
]

# Sapphire substrate
z_top = -r
z_bot = -0.5*sz + pml + 1e-3
substrate = mp.Block(
    material=sapphire,
    size=mp.Vector3(mp.inf, mp.inf, z_top - z_bot),
    center=mp.Vector3(0, 0, 0.5*(z_top + z_bot))
)

geometry = spheres + [substrate]

# ----------- Laser Source -----------
z_src = 0.5*sz - pml - 0.8
focus = mp.Vector3(0, 0, 0)

print(f"Source position: z = {z_src:.2f} µm")
print(f"Sphere top: z = {r:.2f} µm\n")

def make_gaussian_beam(theta_deg):
    """Create angled Gaussian beam source"""
    th = math.radians(theta_deg)
    kdir = mp.Vector3(math.sin(th), 0, -math.cos(th))  # Positive x component

    return mp.GaussianBeamSource(
        src=mp.GaussianSource(frequency=fcen, fwidth=fwidth),
        center=mp.Vector3(0, 0, z_src),
        size=mp.Vector3(sx, sy, 0),
        beam_x0=focus - mp.Vector3(0, 0, z_src),
        beam_kdir=kdir,
        beam_w0=beam_w0,
        beam_E0=mp.Vector3(0, E0_meep, 0)  # Ey polarization
    )

# ----------- Boundaries: PERIODIC in X,Y with Bloch k-point -----------
boundary_layers = [
    mp.PML(pml, direction=mp.Z, side=mp.High),
    mp.PML(pml, direction=mp.Z, side=mp.Low),
    # NO PML in X or Y - using Bloch-periodic boundaries!
]

# Bloch k-point (matches beam angle)
def make_k_point(theta_deg):
    th = math.radians(theta_deg)
    kx = math.sin(th)  # Positive to match kdir.x
    return mp.Vector3(fcen*kx, 0.0, 0.0)

# ----------- Animation Settings -----------
resolution = 12  # INCREASED resolution to reduce numerical dispersion (was 8-10)
angle = 23.5     # Incident angle in degrees
T_RUN = 120      # Total simulation time
fps = 15         # Frames per second for animation
n_frames = 120   # Total number of frames

# Calculate time step between frames
dt_frame = T_RUN / n_frames

print(f"Creating animation with {n_frames} frames at {fps} FPS")
print(f"Incident angle: {angle}°")
print(f"Resolution: {resolution} pixels/µm (increased to reduce dispersion)")
print(f"Bloch k-point: kx = {math.sin(math.radians(angle)) * fcen:.4f}")
print(f"Time step between frames: {dt_frame:.2f} MEEP time units\n")

# ----------- Field Capture Function -----------
def make_field_capture():
    """Factory function to create field capture callback"""
    frames = []
    times = []
    frame_count = [0]

    def capture_fields(sim):
        """Capture current field state"""
        # Get XZ slice at y=0 (through sphere center in front row)
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

    # Attach data storage to the function
    capture_fields.frames = frames
    capture_fields.times = times
    capture_fields.frame_count = frame_count

    return capture_fields

# ----------- Create and Run Simulation -----------
print("[Starting simulation with PERIODIC boundaries...]")

sim = mp.Simulation(
    cell_size=cell,
    geometry=geometry,
    boundary_layers=boundary_layers,
    resolution=resolution,
    default_material=air,
    k_point=make_k_point(angle)  # CRITICAL: Bloch k-point for periodic boundaries
)

sim.sources = [make_gaussian_beam(angle)]

print(f"Bloch k-point: ({sim.k_point.x:.4f}, {sim.k_point.y:.4f}, {sim.k_point.z:.4f})")
print(f"This accounts for interference from neighboring spheres in the array!\n")

# Create field capture function
field_capture = make_field_capture()

# Run simulation with field capture
sim.run(mp.at_every(dt_frame, field_capture), until=T_RUN)

print(f"\n[Simulation complete] Captured {len(field_capture.frames)} frames")

# ----------- Create Animation -----------
print("\n[Generating animation...]")

# Get coordinate arrays for SINGLE unit cell
nx, nz = field_capture.frames[0].shape
xs = np.linspace(-0.5*sx, 0.5*sx, nx)
zs = np.linspace(-0.5*sz, 0.5*sz, nz)

# Viewing window - extended to show incoming beam
z_top_crop = +6.0
z_bot_crop = -9.0
iz = np.where((zs >= z_bot_crop) & (zs <= z_top_crop))[0]
zs_crop = zs[iz]

print(f"Viewing window: z = [{z_bot_crop:.1f}, {z_top_crop:.1f}] µm")
print(f"  (Sphere: z = [{-r:.1f}, {r:.1f}] µm)")
print(f"  (Source: z = {z_src:.1f} µm)")

# Crop all frames
frames_cropped = [frame[:, iz] for frame in field_capture.frames]

# Determine global intensity range for consistent colormap
all_intensities = np.concatenate([frame.flatten() for frame in frames_cropped])
vmax = np.percentile(all_intensities, 99.5)
vmin = 0

print(f"Intensity range: {vmin:.2e} to {vmax:.2e}")

# Calculate aspect ratio
x_range = xs[-1] - xs[0]
z_range = zs_crop[-1] - zs_crop[0]
aspect_ratio = z_range / x_range
fig_width = 10
fig_height = fig_width * aspect_ratio * 1.1  # Slight adjustment for labels

# Create figure
fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=120)

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

# Draw sphere outline (at origin - first sphere in unit cell)
theta_circle = np.linspace(0, 2*np.pi, 200)
sphere_x = r * np.cos(theta_circle)
sphere_z = r * np.sin(theta_circle)
sphere_outline, = ax.plot(sphere_x, sphere_z, 'cyan', linewidth=2.5, alpha=0.95,
                          label='Silica sphere (one unit cell)')

# Sapphire surface
substrate_line, = ax.plot([xs[0], xs[-1]], [-r, -r],
                          color='#ffaa00', linewidth=2.5, label='Sapphire surface')

# Substrate shading
ax.fill_between([xs[0], xs[-1]], [z_bot_crop, z_bot_crop], [-r, -r],
                color='gray', alpha=0.15)

# Air region - very subtle
ax.fill_between([xs[0], xs[-1]], [r, r], [z_top_crop, z_top_crop],
                color='lightblue', alpha=0.03)

# Reference lines
ax.axhline(y=0, color='white', linestyle=':', alpha=0.25, linewidth=1)  # Sphere center
ax.axhline(y=r, color='cyan', linestyle=':', alpha=0.2, linewidth=0.8)  # Sphere top

# Labels and formatting
ax.set_xlabel('x (µm)', fontsize=14, fontweight='bold')
ax.set_ylabel('z (µm)', fontsize=14, fontweight='bold')

# Info text box
info_text = (f"Angle: {angle}°\n"
             f"Periodic boundaries (X,Y)\n"
             f"Unit cell view")
ax.text(0.98, 0.98, info_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.85),
                    fontweight='bold')

ax.set_xlim(xs[0], xs[-1])
ax.set_ylim(zs_crop[0], zs_crop[-1])
ax.grid(True, color='white', alpha=0.15, linewidth=0.5, linestyle=':')
ax.legend(loc='upper left', fontsize=9, framealpha=0.85)

ax.set_title(f'PNJ Wavefront in Periodic Array (Single Unit Cell View)',
             fontsize=13, fontweight='bold', pad=10)

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Intensity |E|²', fontsize=11, fontweight='bold')

plt.tight_layout()

# Animation update function
def update(frame_idx):
    """Update animation frame"""
    im.set_array(np.rot90(frames_cropped[frame_idx]))
    time_text.set_text(f't = {field_capture.times[frame_idx]:.1f}\nFrame {frame_idx+1}/{n_frames}')
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
output_file = f'pnj_wavefront_periodic_theta{angle:.1f}_res{resolution}.mp4'
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
frames_dir = f"frames_periodic_theta{angle:.1f}_res{resolution}"
os.makedirs(frames_dir, exist_ok=True)

# Save every Nth frame
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
        aspect="equal"
    )

    # Sphere and substrate
    ax_frame.plot(sphere_x, sphere_z, 'cyan', linewidth=2.5, alpha=0.95)
    ax_frame.plot([xs[0], xs[-1]], [-r, -r], color='#ffaa00', linewidth=2.5)
    ax_frame.fill_between([xs[0], xs[-1]], [z_bot_crop, z_bot_crop], [-r, -r],
                          color='gray', alpha=0.15)
    ax_frame.axhline(y=0, color='white', linestyle=':', alpha=0.25, linewidth=1)

    ax_frame.set_xlabel('x (µm)', fontsize=13, fontweight='bold')
    ax_frame.set_ylabel('z (µm)', fontsize=13, fontweight='bold')
    ax_frame.set_title(f'PNJ Periodic Array - t={field_capture.times[i]:.1f} - θ={angle}°',
                      fontsize=13, fontweight='bold')
    ax_frame.grid(True, color='white', alpha=0.15, linewidth=0.5, linestyle=':')

    plt.tight_layout()
    frame_file = f"{frames_dir}/frame_{i:04d}.png"
    plt.savefig(frame_file, dpi=150, bbox_inches='tight')
    plt.close()

print(f"[Success] Saved {len(range(0, len(frames_cropped), save_every))} key frames to {frames_dir}/")

print("\n" + "="*70)
print("PERIODIC WAVEFRONT ANIMATION COMPLETE!")
print("="*70)
print(f"Video file: {output_file}")
print(f"Frame images: {frames_dir}/")
print(f"\nKey features:")
print(f"  ✓ Bloch-periodic boundaries in X and Y (kx = {sim.k_point.x:.4f})")
print(f"  ✓ Accounts for neighboring sphere interference")
print(f"  ✓ Shows single unit cell (one sphere at y=0)")
print(f"  ✓ Higher resolution (res={resolution}) to reduce numerical dispersion")
print(f"  ✓ Same geometry as HCP angle comparison simulations")
print("="*70)
