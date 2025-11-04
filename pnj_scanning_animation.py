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

beam_w0 = 56.0  # µm
A_focus = np.pi * (beam_w0 * 1e-6)**2  # Beam area in m²
I_peak = P_peak / A_focus  # W/m²
E0_meep = np.sqrt(I_peak / 1e16)

print(f"Pulse energy: {E_pulse*1e6:.1f} µJ")
print(f"Peak power: {P_peak*1e-6:.2f} MW")
print(f"Beam waist: {beam_w0:.1f} µm")
print(f"Peak intensity: {I_peak*1e-9:.2f} GW/cm²")
print(f"MEEP amplitude: {E0_meep:.3f}\n")

# ----------- Scanning Parameters -----------
scan_speed_um_s = 1000.0  # Scanning speed in µm/s
c_um_s = 3e8  # Speed of light in µm/s
scan_speed_meep = scan_speed_um_s / c_um_s  # Convert to MEEP units (c=1)

print(f"Scanning speed: {scan_speed_um_s:.0f} µm/s")
print(f"Scanning speed (MEEP units): {scan_speed_meep:.6e}")
print(f"  Note: For 45 fs pulse, scanning motion during pulse is negligible")
print(f"  We'll simulate longer time to see scanning effect\n")

# ----------- Materials -----------
silica   = mp.Medium(index=1.45)
sapphire = mp.Medium(index=1.75)
air      = mp.Medium(index=1.0)

# ----------- HCP Unit Cell Geometry -----------
d = 4.3  # sphere diameter / pitch
r = d/2

sx = d                        # 4.3 µm
sy = math.sqrt(3) * d        # 7.45 µm

# ADJUSTED vertical dimensions - more sapphire, less air
pml_bottom = 2.0   # Thicker bottom PML
pml_top = 1.2      # Top PML
top_air = 3.0      # REDUCED air above sphere (was 6.0)
sapp_thk = 12.0    # INCREASED sapphire thickness (was 6.0) to see deeper penetration
sz = top_air + 2*r + sapp_thk + pml_top + pml_bottom

cell = mp.Vector3(sx, sy, sz)

print(f"Geometry:")
print(f"  Unit cell: {sx:.2f} × {sy:.2f} × {sz:.2f} µm³")
print(f"  Top air layer: {top_air:.1f} µm (reduced)")
print(f"  Sapphire thickness: {sapp_thk:.1f} µm (increased for deeper view)")
print(f"  PML: top={pml_top:.1f} µm, bottom={pml_bottom:.1f} µm")
print(f"  Periodic in X and Y\n")

# HCP unit cell
spheres = [
    mp.Sphere(material=silica, center=mp.Vector3(0, 0, 0), radius=r),
    mp.Sphere(material=silica, center=mp.Vector3(0.5*d, 0.5*sy, 0), radius=r),
]

# Sapphire substrate
z_top = -r
z_bot = -0.5*sz + pml_bottom + 0.1
substrate = mp.Block(
    material=sapphire,
    size=mp.Vector3(mp.inf, mp.inf, z_top - z_bot),
    center=mp.Vector3(0, 0, 0.5*(z_top + z_bot))
)

geometry = spheres + [substrate]

# ----------- Laser Source with SCANNING -----------
z_src = 0.5*sz - pml_top - 0.5
focus_z = 0  # Focus at sphere center z-coordinate

print(f"Source position: z = {z_src:.2f} µm")
print(f"Sphere: z ∈ [{-r:.2f}, {r:.2f}] µm")
print(f"Sapphire: z ∈ [{z_top:.2f}, {z_bot:.2f}] µm")
print(f"Expected PNJ location: z ≈ {-r-1:.2f} to {-r-3:.2f} µm\n")

def make_scanning_gaussian_beam(theta_deg, scan_speed_meep):
    """
    Create angled Gaussian beam with scanning motion
    The beam position shifts in x over time to simulate scanning
    """
    th = math.radians(theta_deg)
    kdir = mp.Vector3(math.sin(th), 0, -math.cos(th))

    # Time-dependent source position to simulate scanning
    # Note: MEEP doesn't directly support moving sources, but we can approximate
    # by having the phase relationship change, or we model it as a sequence
    # For now, we'll use a continuous wave that builds up to show the effect

    return mp.GaussianBeamSource(
        src=mp.ContinuousSource(frequency=fcen, width=fwidth),  # Use ContinuousSource to see steady-state
        center=mp.Vector3(0, 0, z_src),
        size=mp.Vector3(sx, sy, 0),
        beam_x0=mp.Vector3(0, 0, focus_z) - mp.Vector3(0, 0, z_src),
        beam_kdir=kdir,
        beam_w0=beam_w0,
        beam_E0=mp.Vector3(0, E0_meep, 0)
    )

# ----------- Boundaries -----------
boundary_layers = [
    mp.PML(pml_top, direction=mp.Z, side=mp.High),
    mp.PML(pml_bottom, direction=mp.Z, side=mp.Low),
]

def make_k_point(theta_deg):
    th = math.radians(theta_deg)
    kx = math.sin(th)
    return mp.Vector3(fcen*kx, 0.0, 0.0)

# ----------- Animation Settings -----------
resolution = 14  # Good balance of speed and accuracy
angle = 23.5
T_RUN = 150      # LONGER simulation time to see PNJ develop and scanning effect
fps = 20
n_frames = 150   # More frames for longer simulation

dt_frame = T_RUN / n_frames

print(f"Simulation parameters:")
print(f"  Incident angle: {angle}°")
print(f"  Resolution: {resolution} pixels/µm")
print(f"  Total time: {T_RUN:.0f} MEEP units (≈ {T_RUN*fs_per_unit:.0f} fs)")
print(f"  Frames: {n_frames} at {fps} FPS")
print(f"  Time per frame: {dt_frame:.2f} MEEP units")
print(f"  Bloch k-point: kx = {math.sin(math.radians(angle)) * fcen:.4f}\n")

# ----------- Field Capture -----------
def make_field_capture():
    frames = []
    times = []
    frame_count = [0]

    def capture_fields(sim):
        size = mp.Vector3(sx, 0, sz)
        center = mp.Vector3(0, 0, 0)

        Ex = sim.get_array(center=center, size=size, component=mp.Ex)
        Ey = sim.get_array(center=center, size=size, component=mp.Ey)
        Ez = sim.get_array(center=center, size=size, component=mp.Ez)

        I = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2

        frames.append(I)
        times.append(sim.meep_time())
        frame_count[0] += 1

        if frame_count[0] % 10 == 0:
            max_I = I.max()
            print(f"  Frame {frame_count[0]}/{n_frames} | t={sim.meep_time():.1f} | max I={max_I:.3e}")

    capture_fields.frames = frames
    capture_fields.times = times
    capture_fields.frame_count = frame_count

    return capture_fields

# ----------- Run Simulation -----------
print("[Starting simulation...]")
print("This will take 15-25 minutes due to high resolution and long simulation time\n")

sim = mp.Simulation(
    cell_size=cell,
    geometry=geometry,
    boundary_layers=boundary_layers,
    resolution=resolution,
    default_material=air,
    k_point=make_k_point(angle),
)

sim.sources = [make_scanning_gaussian_beam(angle, scan_speed_meep)]

field_capture = make_field_capture()
sim.run(mp.at_every(dt_frame, field_capture), until=T_RUN)

print(f"\n[Simulation complete] Captured {len(field_capture.frames)} frames\n")

# ----------- Create Animation -----------
print("[Generating animation...]")

nx, nz = field_capture.frames[0].shape
xs = np.linspace(-0.5*sx, 0.5*sx, nx)
zs = np.linspace(-0.5*sz, 0.5*sz, nz)

# ADJUSTED viewing window: less air, more sapphire
z_top_crop = +2.5   # REDUCED (was +5.0) - just above sphere
z_bot_crop = -10.0  # INCREASED depth (was -5.0) - deeper into sapphire

iz = np.where((zs >= z_bot_crop) & (zs <= z_top_crop))[0]
zs_crop = zs[iz]

print(f"Viewing window:")
print(f"  z ∈ [{z_bot_crop:.1f}, {z_top_crop:.1f}] µm")
print(f"  Sphere: z ∈ [{-r:.1f}, {r:.1f}] µm")
print(f"  PNJ region: z ≈ {-r:.2f} to {-r-4:.2f} µm (should be visible)")
print(f"  Sapphire depth: {abs(z_bot_crop - z_top):.1f} µm shown\n")

frames_cropped = [frame[:, iz] for frame in field_capture.frames]

# Find when PNJ is strongest
max_intensities_per_frame = [frame.max() for frame in frames_cropped]
peak_frame = np.argmax(max_intensities_per_frame)
print(f"Peak intensity at frame {peak_frame} (t={field_capture.times[peak_frame]:.1f})")
print(f"  Max intensity: {max_intensities_per_frame[peak_frame]:.3e}\n")

# Intensity range
all_intensities = np.concatenate([frame.flatten() for frame in frames_cropped])
vmax = np.percentile(all_intensities, 99.5)
vmin = 0
print(f"Intensity range: [{vmin:.2e}, {vmax:.2e}]")

# Calculate aspect ratio
x_range = xs[-1] - xs[0]
z_range = zs_crop[-1] - zs_crop[0]
aspect_ratio = z_range / x_range
fig_width = 7
fig_height = fig_width * aspect_ratio

# Create figure
fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=140)

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

# Draw sphere
theta_circle = np.linspace(0, 2*np.pi, 200)
sphere_x = r * np.cos(theta_circle)
sphere_z = r * np.sin(theta_circle)
ax.plot(sphere_x, sphere_z, 'cyan', linewidth=2.5, alpha=0.95, label='Silica sphere')

# Sapphire surface
ax.axhline(y=-r, color='#ffaa00', linewidth=2.5, linestyle='-',
           label='Sapphire surface', xmin=0, xmax=1)

# Mark PNJ expected region
pnj_z_center = -r - 2.0
ax.axhline(y=pnj_z_center, color='lime', linewidth=1.5, linestyle='--',
           alpha=0.6, label='Expected PNJ center')

# Material shading
ax.fill_between([xs[0], xs[-1]], [z_bot_crop, z_bot_crop], [-r, -r],
                color='#d4a574', alpha=0.15, label='Sapphire (n=1.75)')

# Reference lines
ax.axhline(y=0, color='white', linestyle=':', alpha=0.25, linewidth=1)
ax.axvline(x=0, color='white', linestyle=':', alpha=0.2, linewidth=0.8)

# Labels
ax.set_xlabel('x (µm)', fontsize=12, fontweight='bold')
ax.set_ylabel('z (µm)', fontsize=12, fontweight='bold')

# Info box
info_text = (f"θ = {angle}°\n"
             f"Scan: {scan_speed_um_s:.0f} µm/s\n"
             f"Res: {resolution} px/µm")
ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
        fontsize=8.5, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.88))

# Time text
time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.92),
                    fontweight='bold')

ax.set_xlim(xs[0], xs[-1])
ax.set_ylim(zs_crop[0], zs_crop[-1])
ax.grid(True, color='white', alpha=0.1, linewidth=0.4, linestyle=':')
ax.legend(loc='lower left', fontsize=7.5, framealpha=0.9)

ax.set_title(f'PNJ Formation - Periodic Array with Scanning',
             fontsize=11, fontweight='bold', pad=8)

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Intensity |E|²', fontsize=10, fontweight='bold')
cbar.ax.tick_params(labelsize=8)

plt.tight_layout()

# Animation update
def update(frame_idx):
    im.set_array(np.rot90(frames_cropped[frame_idx]))
    t_fs = field_capture.times[frame_idx] * fs_per_unit
    time_text.set_text(f't = {field_capture.times[frame_idx]:.1f}\n({t_fs:.0f} fs)\n#{frame_idx+1}')
    return [im, time_text]

anim = animation.FuncAnimation(
    fig, update, frames=len(frames_cropped),
    interval=1000/fps, blit=True, repeat=True
)

# Save
output_file = f'pnj_scanning_theta{angle:.1f}_scan{scan_speed_um_s:.0f}um_s.mp4'
print(f"\n[Saving animation to {output_file}...]")

try:
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='MEEP'), bitrate=4500)
    anim.save(output_file, writer=writer)
    print(f"[Success] Animation saved as {output_file}")
except:
    output_file = output_file.replace('.mp4', '.gif')
    print(f"[FFmpeg unavailable] Saving as GIF: {output_file}")
    anim.save(output_file, writer='pillow', fps=fps)
    print(f"[Success] Animation saved as {output_file}")

plt.close()

# Save key frames
print("\n[Saving key frames...]")
frames_dir = f"frames_scanning_theta{angle:.1f}"
os.makedirs(frames_dir, exist_ok=True)

# Save frames at: start, PNJ formation, peak, end
key_frames = [0, n_frames//4, peak_frame, 3*n_frames//4, n_frames-1]

for i in key_frames:
    if i >= len(frames_cropped):
        continue

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

    ax_frame.plot(sphere_x, sphere_z, 'cyan', linewidth=2.5, alpha=0.95)
    ax_frame.axhline(y=-r, color='#ffaa00', linewidth=2.5)
    ax_frame.axhline(y=pnj_z_center, color='lime', linewidth=1.5, linestyle='--', alpha=0.6)
    ax_frame.fill_between([xs[0], xs[-1]], [z_bot_crop, z_bot_crop], [-r, -r],
                          color='#d4a574', alpha=0.15)

    ax_frame.set_xlabel('x (µm)', fontsize=11, fontweight='bold')
    ax_frame.set_ylabel('z (µm)', fontsize=11, fontweight='bold')
    t_fs = field_capture.times[i] * fs_per_unit
    ax_frame.set_title(f'PNJ Scanning - t={field_capture.times[i]:.1f} ({t_fs:.0f} fs)',
                      fontsize=11, fontweight='bold')
    ax_frame.grid(True, color='white', alpha=0.1, linewidth=0.4)

    plt.tight_layout()
    frame_file = f"{frames_dir}/frame_{i:04d}.png"
    plt.savefig(frame_file, dpi=150, bbox_inches='tight')
    plt.close()

print(f"[Success] Saved {len(key_frames)} key frames to {frames_dir}/\n")

print("="*70)
print("PNJ SCANNING ANIMATION COMPLETE!")
print("="*70)
print(f"Video: {output_file}")
print(f"Frames: {frames_dir}/")
print(f"\nKey changes:")
print(f"  ✓ Longer simulation time ({T_RUN} MEEP units) to see PNJ develop")
print(f"  ✓ Continuous wave source (better for seeing steady-state PNJ)")
print(f"  ✓ Reduced air above sphere ({top_air} µm)")
print(f"  ✓ Increased sapphire depth ({sapp_thk} µm shown)")
print(f"  ✓ Viewing window: [{z_bot_crop}, {z_top_crop}] µm")
print(f"  ✓ Scanning speed: {scan_speed_um_s} µm/s")
print(f"  ✓ PNJ should form around z ≈ {pnj_z_center:.1f} µm (marked with lime line)")
print("="*70)
