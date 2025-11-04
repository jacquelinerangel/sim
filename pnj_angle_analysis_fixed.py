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
E0_meep = np.sqrt(I_peak / 1e16)  # Empirical scaling factor
print(f"MEEP amplitude: {E0_meep:.3f}\n")

# ----------- Materials Refractive Indices -----------
silica   = mp.Medium(index=1.45)
sapphire = mp.Medium(index=1.75)
air      = mp.Medium(index=1.0)

# ----------- HCP 2D unitcell - CORRECTED -----------
d = 4.3  # sphere diameter / pitch
r = d/2

# Rectangular unit cell for hexagonal tiling - CORRECTED
sx = d                        # 4.3 µm (NOT 2*d!)
sy = math.sqrt(3) * d        # 7.45 µm
cell = mp.Vector3(sx, sy, 0)  # Will add sz later

# 2-sphere HCP unit cell with touching spheres
spheres = [
    mp.Sphere(material=silica, center=mp.Vector3(0, 0, 0), radius=r),
    mp.Sphere(material=silica, center=mp.Vector3(0.5*d, 0.5*sy, 0), radius=r),
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
    # Beam from top-LEFT to bottom-RIGHT (positive x, negative z)
    kdir = mp.Vector3(math.sin(th), 0, -math.cos(th))  # POSITIVE x component
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

# Bloch k-point (FIXED: matches beam direction)
def make_k_point(theta_deg):
    th = math.radians(theta_deg)
    kx = math.sin(th)  # ✅ POSITIVE to match kdir.x = +sin(th)
    return mp.Vector3(fcen*kx, 0.0, 0.0)

# ----------- numerics -----------
resolution = 8     # Lower for speed (was 14)
T_RUN = 80         # Shorter run time (was 140)

# ----------- Single Front Row Visualization -----------
def save_front_row_pnjs(sim, theta_deg, n_spheres=3):
    """
    Extract and plot normalized intensity for ONE ROW of PNJs (y=0 slice).
    Shows n_spheres side-by-side by tiling the unit cell.
    Returns the intensity data for comparison.
    """
    # Get XZ slice at y=0 (front row)
    size   = mp.Vector3(sx, 0, sz)
    center = mp.Vector3(0, 0, 0)

    Ex = sim.get_array(center=center, size=size, component=mp.Ex)
    Ey = sim.get_array(center=center, size=size, component=mp.Ey)
    Ez = sim.get_array(center=center, size=size, component=mp.Ez)
    I  = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2

    # Also get dielectric structure to see where spheres actually are
    eps = sim.get_array(center=center, size=size, component=mp.Dielectric)

    nx, nz = I.shape

    print(f"  Single cell dimensions: nx={nx}, nz={nz}, sx={sx:.2f} µm, sz={sz:.2f} µm")

    # Coordinate arrays for SINGLE unit cell
    xs_single = np.linspace(-0.5*sx, 0.5*sx, nx)
    zs = np.linspace(-0.5*sz, 0.5*sz, nz)

    # Tile
    I_tiled = np.tile(I, (n_spheres, 1))
    eps_tiled = np.tile(eps, (n_spheres, 1))

    # Coordinate arrays for tiled region
    xs = np.linspace(-0.5*n_spheres*sx, 0.5*n_spheres*sx, I_tiled.shape[0])

    # Cropping in z for better visualization
    z_top_um = +3.0
    z_bot_um = -9.0
    iz = np.where((zs >= z_bot_um) & (zs <= z_top_um))[0]
    I_crop = I_tiled[:, iz]
    eps_crop = eps_tiled[:, iz]
    zs_crop = zs[iz]

    # Store max intensity from cropped region
    max_intensity = float(I_crop.max()) if I_crop.size > 0 and I_crop.max() > 0 else 1.0

    # Normalize for display
    if max_intensity > 0:
        I_crop_normalized = I_crop / max_intensity
    else:
        I_crop_normalized = I_crop

    # Calculate aspect ratio to make axes equal scale
    x_range = xs[-1] - xs[0]
    z_range = zs_crop[-1] - zs_crop[0]
    aspect_ratio = z_range / x_range
    fig_width = 12
    fig_height = fig_width * aspect_ratio

    # Plot
    plt.figure(figsize=(fig_width, fig_height), dpi=240)
    plt.imshow(
        np.rot90(I_crop_normalized),
        extent=[xs[0], xs[-1], zs_crop[0], zs_crop[-1]],
        origin="lower",
        vmin=0, vmax=1,
        cmap="turbo",
        aspect="equal"
    )
    plt.xlabel("x (µm)", fontsize=14, fontweight='bold')
    plt.ylabel("z (µm)", fontsize=14, fontweight='bold')
    plt.tick_params(axis='both', labelsize=12)

    # Add grid
    plt.grid(True, color='white', alpha=0.2, linewidth=0.5, linestyle=':')

    plt.title(f"Angled Exposure PNJs at {theta_deg}° (kx={sim.k_point.x:.3f})",
              fontsize=16, fontweight='bold', pad=12)
    cbar = plt.colorbar()
    cbar.set_label("Normalized Intensity |E|²", fontsize=13, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)

    # Draw sphere outlines - one per unit cell at x = i*sx
    th_circle = np.linspace(0, 2*np.pi, 512)
    for i in range(n_spheres):
        xc = (i - n_spheres//2) * sx
        xs_c = xc + r*np.cos(th_circle)
        zs_c = 0 + r*np.sin(th_circle)
        plt.plot(xs_c, zs_c, color="white", linewidth=1.5, alpha=0.95, label='Sphere' if i == 0 else '')
        # Add small marker at center
        plt.plot(xc, 0, 'w+', markersize=8, markeredgewidth=1.5)

    # Sapphire surface
    plt.plot([xs[0], xs[-1]], [-r, -r], color="#ffaa00", linewidth=2.0, label="Sapphire surface")

    # Faint substrate shading
    plt.fill_between([xs[0], xs[-1]], [-r, zs_crop[0]], color="gray", alpha=0.08)

    plt.xlim(xs[0], xs[-1])
    plt.ylim(zs_crop[0], zs_crop[-1])
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()

    out = f"hcp_frontrow_pnj_theta{theta_deg:.1f}_power{P_avg*1e3:.0f}mW_FIXED.png"
    plt.savefig(out, dpi=260)
    plt.close()
    print(f"[save] {out}")

    # Return data for angle comparison
    return {
        'intensity': I_tiled,
        'xs': xs,
        'zs': zs,
        'I_crop': I_crop,
        'zs_crop': zs_crop,
        'theta': theta_deg,
        'max_intensity': max_intensity
    }

# ----------- Angle Comparison Functions -----------
def plot_intensity_vs_angle(angle_data):
    """
    Plot comparison of intensity distributions across different angles.
    """
    angles = sorted(angle_data.keys())

    # 1. Maximum intensity vs angle
    plt.figure(figsize=(8, 6))
    max_intensities = [angle_data[ang]['max_intensity'] for ang in angles]
    plt.plot(angles, max_intensities, 'o-', linewidth=2, markersize=8, color='#ff6b35')
    plt.xlabel("Angle (degrees)", fontsize=12)
    plt.ylabel("Maximum Normalized Intensity", fontsize=12)
    plt.title("Peak Intensity vs Exposure Angle", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"max_intensity_vs_angle_power{P_avg*1e3:.0f}mW_FIXED.png", dpi=200)
    plt.close()
    print(f"[save] max_intensity_vs_angle_power{P_avg*1e3:.0f}mW_FIXED.png")

    # 2. Intensity profiles along z-axis at x=0
    plt.figure(figsize=(10, 7))
    for ang in angles:
        data = angle_data[ang]
        # Extract center column (x=0)
        nx = data['intensity'].shape[0]
        center_idx = nx // 2
        z_profile = data['intensity'][center_idx, :]
        # Normalize each profile to its own max
        if z_profile.max() > 0:
            z_profile = z_profile / z_profile.max()
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
    plt.savefig(f"z_profile_comparison_power{P_avg*1e3:.0f}mW_FIXED.png", dpi=200)
    plt.close()
    print(f"[save] z_profile_comparison_power{P_avg*1e3:.0f}mW_FIXED.png")

    # 3. Substrate intensity vs angle
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
    plt.ylabel("Intensity at Substrate (normalized)", fontsize=12)
    plt.title("PNJ Intensity at Substrate Surface vs Angle", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"substrate_intensity_vs_angle_power{P_avg*1e3:.0f}mW_FIXED.png", dpi=200)
    plt.close()
    print(f"[save] substrate_intensity_vs_angle_power{P_avg*1e3:.0f}mW_FIXED.png")

# ----------- main -----------
if __name__ == "__main__":
    # Run all angles for comparison
    angles = [15.0, 22.5, 30.0]

    # Store data for each angle
    angle_data = {}

    for th in angles:
        sim = mp.Simulation(
            cell_size=cell,
            geometry=geometry,
            boundary_layers=boundary_layers,
            resolution=resolution,
            default_material=air,
            k_point=make_k_point(th)  # ✅ FIXED: Now using correct Bloch k-vector
        )
        sim.sources = [make_gaussian_beam(th)]

        print(f"\n[run] HCP on sapphire, angle={th}°, res={resolution}, T={T_RUN}")
        print(f"      Unit cell: {sx:.2f} × {sy:.2f} × {sz:.2f} µm³")
        print(f"      Bloch k-point: kx = {sim.k_point.x:.4f}")
        sim.run(until=T_RUN)

        # Save front-row PNJ visualization and store data
        data = save_front_row_pnjs(sim, th, n_spheres=3)
        angle_data[th] = data

        print()

    # Create comparison plots
    print("\n[Generating angle comparison plots...]")
    plot_intensity_vs_angle(angle_data)

    print("\n[Complete] All simulations and comparisons finished!")
