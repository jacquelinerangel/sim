import meep as mp
import numpy as np
import math
import matplotlib.pyplot as plt

# ----------- Laser Parameters in MEEP Units -----------
lambda0 = 0.8   # Laser wavelength in um
fcen    = 1.0 / lambda0  # Center Frequency (um^-1)
FWHM_fs = 45.0  # Pulse FWHM (fs)
fs_per_unit = 3.3356409519815204  # Convert 1 µm/c to fs
sigma_t_meep = (FWHM_fs/(2.0*math.sqrt(math.log(2.0))))/fs_per_unit
fwidth  = 1.0/(4.0*math.pi*sigma_t_meep)

# ----------- Power and Intensity Calculations -----------
P_avg = 40e-3  # Average power in Watts
rep_rate = 1e3  # Repetition rate in Hz (1 kHz - adjust for your laser)

# Pulse energy
E_pulse = P_avg / rep_rate  # Joules per pulse
print(f"Pulse energy: {E_pulse*1e6:.1f} µJ")

# Peak power (for Gaussian pulse)
tau_FWHM = FWHM_fs * 1e-15  # Convert to seconds
tau_pulse = tau_FWHM / (2 * np.sqrt(np.log(2)))  # 1/e width
P_peak = E_pulse / (tau_pulse * np.sqrt(np.pi))
print(f"Peak power: {P_peak*1e-6:.2f} MW")

# Beam waist and intensity
beam_w0 = 56.0  # µm
A_focus = np.pi * (beam_w0 * 1e-6)**2  # Beam area in m²
I_peak = P_peak / A_focus  # W/m²
print(f"Peak intensity: {I_peak*1e-9:.2f} GW/cm²")

# Fluence
fluence = I_peak * tau_FWHM * 1e-15  # J/m²
print(f"Fluence: {fluence*1e-4:.3f} J/cm²")

# Electric field amplitude in SI units
eps0 = 8.854e-12  # F/m
c_si = 3e8  # m/s
n_medium = 1.0  # refractive index at source (air)
E0_SI = np.sqrt(2 * I_peak / (eps0 * c_si * n_medium))  # V/m
print(f"Peak E-field: {E0_SI*1e-6:.1f} MV/m")

# Convert to MEEP units (approximate scaling for Gaussian source)
# MEEP's amplitude parameter scales the source
E0_meep = np.sqrt(I_peak / 1e16)  # Empirical scaling factor
print(f"MEEP amplitude: {E0_meep:.3f}\n")

# ----------- Materials Refractive Indices -----------
silica   = mp.Medium(index=1.45)
sapphire = mp.Medium(index=1.75)
air      = mp.Medium(index=1.0)

# ----------- HCP 2D unitcell - CORRECTED -----------
d = 4.3  # sphere diameter / pitch
r = d/2

# Rectangular unit cell for hexagonal tiling
sx = 2 * d                    # 8.6 µm
sy = math.sqrt(3) * d        # 7.44 µm
cell = mp.Vector3(sx, sy, 0)  # Will add sz later

# HCP unit cell with uniform front row at y=0
# Two spheres per unit cell: both contribute to front row uniformly
spheres = [
    mp.Sphere(material=silica, center=mp.Vector3(0, 0, 0), radius=r),
    mp.Sphere(material=silica, center=mp.Vector3(0.5*sx, 0.5*sy, 0), radius=r),
]

# vertical stack (Z-axis boundaries and dimensions)
pml      = 1.2  # boundary/absorption thickness (um)
top_air  = 4.0  # air thickness (um)
sapp_thk = 10.0 # wafer thickness (um)
sz = top_air + 2*r + sapp_thk + 2*pml

# Update cell with z dimension
cell = mp.Vector3(sx, sy, sz)

# sapphire wafer: top at z = -r (touching sphere bottom)
z_top = -r
z_bot = -0.5*sz + pml + 1e-3
substrate = mp.Block(
    material=sapphire,
    size=mp.Vector3(mp.inf, mp.inf, z_top - z_bot),
    center=mp.Vector3(0, 0, 0.5*(z_top + z_bot))
)
geometry = spheres + [substrate]

# ----------- Laser Source: Gaussian (angle from +z) -----------
z_src   = 0.5*sz - pml - 0.8
focus   = mp.Vector3(0, 0, 0)

def make_gaussian_beam(theta_deg):
    th   = math.radians(theta_deg)
    kdir = mp.Vector3(-math.sin(th), 0, -math.cos(th))
    # Scale beam_E0 by amplitude instead
    return mp.GaussianBeamSource(
        src=mp.GaussianSource(frequency=fcen, fwidth=fwidth),
        center=mp.Vector3(0, 0, z_src),
        size=mp.Vector3(sx, sy, 0),
        beam_x0=focus - mp.Vector3(0, 0, z_src),
        beam_kdir=kdir,
        beam_w0=beam_w0,
        beam_E0=mp.Vector3(0, E0_meep, 0)  # Ey polarization with amplitude
    )

# ----------- boundaries -----------
boundary_layers = [
    mp.PML(pml, direction=mp.Z, side=mp.High),
    mp.PML(pml, direction=mp.Z, side=mp.Low),
]

# Bloch k-point
def make_k_point(theta_deg):
    th = math.radians(theta_deg)
    kx = -math.sin(th)
    return mp.Vector3(fcen*kx, 0.0, 0.0)

# ----------- numerics -----------
resolution = 14
T_RUN = 140

# ----------- Single Front Row Visualization -----------
def save_front_row_pnjs(sim, theta_deg, n_spheres=3):
    """
    Extract and plot normalized intensity for ONE ROW of PNJs (y=0 slice).
    Shows n_spheres side-by-side by tiling the unit cell.
    Returns the intensity data and coordinate arrays for later analysis.
    """
    # Get XZ slice at y=0 (front row)
    size   = mp.Vector3(sx, 0, sz)
    center = mp.Vector3(0, 0, 0)

    Ex = sim.get_array(center=center, size=size, component=mp.Ex)
    Ey = sim.get_array(center=center, size=size, component=mp.Ey)
    Ez = sim.get_array(center=center, size=size, component=mp.Ez)
    I  = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2

    nx, nz = I.shape

    # Tile in x-direction to show multiple spheres
    I_tiled = np.tile(I, (n_spheres, 1))

    # Normalize
    if I_tiled.max() > 0:
        I_tiled /= I_tiled.max()

    # Coordinate arrays
    xs = np.linspace(-0.5*n_spheres*sx, 0.5*n_spheres*sx, I_tiled.shape[0])
    zs = np.linspace(-0.5*sz, 0.5*sz, nz)

    # Cropping in z for better visualization
    z_top_um = +3.0
    z_bot_um = -9.0
    iz = np.where((zs >= z_bot_um) & (zs <= z_top_um))[0]
    I_crop = I_tiled[:, iz]
    zs_crop = zs[iz]

    # Plot
    plt.figure(figsize=(12, 6), dpi=240)
    plt.imshow(
        np.rot90(I_crop),
        extent=[xs[0], xs[-1], zs_crop[0], zs_crop[-1]],
        origin="lower",
        vmin=0, vmax=1,
        cmap="turbo",
        aspect="auto"
    )
    plt.xlabel("x (µm)", fontsize=11)
    plt.ylabel("z (µm)", fontsize=11)
    plt.title(f"Angled Exposure PNJs at {theta_deg}°",
              fontsize=12, pad=10)
    cbar = plt.colorbar()
    cbar.set_label("Normalized Intensity |E|²", fontsize=10)

    # Draw sphere outlines for front row
    th = np.linspace(0, 2*np.pi, 512)
    for i in range(n_spheres):
        # One sphere per unit cell at y=0
        x_offset = (i - n_spheres//2) * sx
        xc = x_offset
        xs_c = xc + r*np.cos(th)
        zs_c = 0   + r*np.sin(th)
        plt.plot(xs_c, zs_c, color="white", linewidth=1.0, alpha=0.8)

    # Sapphire surface
    plt.plot([xs[0], xs[-1]], [-r, -r], color="#ffaa00", linewidth=1.5, label="Sapphire surface")

    # Faint substrate shading
    plt.fill_between([xs[0], xs[-1]], [-r, zs_crop[0]], color="gray", alpha=0.08)

    plt.xlim(xs[0], xs[-1])
    plt.ylim(zs_crop[0], zs_crop[-1])
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    out = f"hcp_frontrow_pnj_theta{theta_deg:.1f}_power{P_avg*1e3:.0f}mW.png"
    plt.savefig(out, dpi=260)
    plt.close()
    print(f"[save] {out}")

    # Return data for comparison plots
    return {
        'intensity': I_tiled,
        'xs': xs,
        'zs': zs,
        'I_crop': I_crop,
        'zs_crop': zs_crop,
        'theta': theta_deg
    }

# ----------- Angle Comparison Functions -----------
def plot_intensity_vs_angle(angle_data):
    """
    Plot comparison of intensity distributions across different angles.
    Creates multiple comparison plots:
    1. Maximum intensity vs angle
    2. Intensity profiles along z-axis (at x=0) for each angle
    3. 2D heatmap of intensity vs (z, angle)
    """
    angles = sorted(angle_data.keys())

    # 1. Maximum intensity vs angle
    plt.figure(figsize=(8, 6))
    max_intensities = [angle_data[ang]['intensity'].max() for ang in angles]
    plt.plot(angles, max_intensities, 'o-', linewidth=2, markersize=8)
    plt.xlabel("Angle (degrees)", fontsize=12)
    plt.ylabel("Maximum Normalized Intensity", fontsize=12)
    plt.title("Peak Intensity vs Exposure Angle", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"max_intensity_vs_angle_power{P_avg*1e3:.0f}mW.png", dpi=200)
    plt.close()
    print("[save] max_intensity_vs_angle_power{:.0f}mW.png".format(P_avg*1e3))

    # 2. Intensity profiles along z-axis at x=0
    plt.figure(figsize=(10, 7))
    for ang in angles:
        data = angle_data[ang]
        # Extract center column (x=0)
        nx = data['intensity'].shape[0]
        center_idx = nx // 2
        z_profile = data['intensity'][center_idx, :]
        plt.plot(data['zs'], z_profile, label=f"{ang}°", linewidth=2)

    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Sphere center')
    plt.axvline(x=-r, color='orange', linestyle='--', alpha=0.5, label='Substrate surface')
    plt.xlabel("z position (µm)", fontsize=12)
    plt.ylabel("Normalized Intensity", fontsize=12)
    plt.title("Axial Intensity Profile (x=0) vs Angle", fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(-9, 3)
    plt.tight_layout()
    plt.savefig(f"z_profile_comparison_power{P_avg*1e3:.0f}mW.png", dpi=200)
    plt.close()
    print("[save] z_profile_comparison_power{:.0f}mW.png".format(P_avg*1e3))

    # 3. 2D heatmap: Intensity vs (z, angle)
    # Create interpolated grid
    z_min, z_max = -9.0, 3.0
    nz_interp = 300
    z_interp = np.linspace(z_min, z_max, nz_interp)

    intensity_grid = np.zeros((len(angles), nz_interp))

    for i, ang in enumerate(angles):
        data = angle_data[ang]
        nx = data['intensity'].shape[0]
        center_idx = nx // 2
        z_profile = data['intensity'][center_idx, :]
        # Interpolate to common z grid
        intensity_grid[i, :] = np.interp(z_interp, data['zs'], z_profile)

    plt.figure(figsize=(10, 6))
    plt.imshow(
        intensity_grid,
        extent=[z_min, z_max, angles[0], angles[-1]],
        origin="lower",
        aspect="auto",
        cmap="turbo",
        vmin=0,
        vmax=1
    )
    plt.xlabel("z position (µm)", fontsize=12)
    plt.ylabel("Exposure Angle (degrees)", fontsize=12)
    plt.title("Intensity Distribution vs Angle (x=0 slice)", fontsize=13)
    cbar = plt.colorbar()
    cbar.set_label("Normalized Intensity", fontsize=11)

    # Mark key positions
    plt.axvline(x=0, color='white', linestyle='--', alpha=0.6, linewidth=1.5, label='Sphere center')
    plt.axvline(x=-r, color='yellow', linestyle='--', alpha=0.6, linewidth=1.5, label='Substrate')
    plt.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"intensity_heatmap_vs_angle_power{P_avg*1e3:.0f}mW.png", dpi=200)
    plt.close()
    print("[save] intensity_heatmap_vs_angle_power{:.0f}mW.png".format(P_avg*1e3))

    # 4. Intensity at substrate surface vs angle
    plt.figure(figsize=(8, 6))
    substrate_intensities = []
    for ang in angles:
        data = angle_data[ang]
        # Find z index closest to -r (substrate surface)
        z_substrate_idx = np.argmin(np.abs(data['zs'] - (-r)))
        nx = data['intensity'].shape[0]
        center_idx = nx // 2
        substrate_intensities.append(data['intensity'][center_idx, z_substrate_idx])

    plt.plot(angles, substrate_intensities, 's-', linewidth=2, markersize=10, color='orangered')
    plt.xlabel("Angle (degrees)", fontsize=12)
    plt.ylabel("Normalized Intensity at Substrate", fontsize=12)
    plt.title("PNJ Intensity at Substrate Surface vs Angle", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"substrate_intensity_vs_angle_power{P_avg*1e3:.0f}mW.png", dpi=200)
    plt.close()
    print("[save] substrate_intensity_vs_angle_power{:.0f}mW.png".format(P_avg*1e3))

def save_angle_data(angle_data, filename="pnj_angle_data.npz"):
    """Save all angle data to a file for later analysis"""
    np.savez(filename, **{f"angle_{int(ang)}": data for ang, data in angle_data.items()})
    print(f"[save] {filename}")

def load_angle_data(filename="pnj_angle_data.npz"):
    """Load previously saved angle data"""
    data = np.load(filename, allow_pickle=True)
    angle_data = {}
    for key in data.keys():
        ang = float(key.replace("angle_", ""))
        angle_data[ang] = data[key].item()
    return angle_data

# ----------- main -----------
if __name__ == "__main__":
    angles = [15.0, 23.5, 30.0]

    # Dictionary to store intensity data for each angle
    angle_data = {}

    for th in angles:
        sim = mp.Simulation(
            cell_size=cell,
            geometry=geometry,
            boundary_layers=boundary_layers,
            resolution=resolution,
            default_material=air,
            k_point=make_k_point(th)
        )
        sim.sources = [make_gaussian_beam(th)]

        print(f"[run] HCP on sapphire, angle={th}°, res={resolution}, T={T_RUN}")
        print(f"      Unit cell: {sx:.2f} × {sy:.2f} × {sz:.2f} µm³")
        sim.run(until=T_RUN)

        # Save front-row PNJ visualization and store data
        data = save_front_row_pnjs(sim, th, n_spheres=3)
        angle_data[th] = data

        print()

    # Create comparison plots
    print("\n[Generating angle comparison plots...]")
    plot_intensity_vs_angle(angle_data)

    # Save data for later use
    save_angle_data(angle_data)

    print("\n[Complete] All simulations and comparisons finished!")
