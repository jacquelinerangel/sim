# PNJ Angle Analysis - Enhanced Simulation

## Overview

This enhanced version of your PNJ simulation code saves normalized intensity data at each exposure angle and creates comparison plots to analyze how the photonic nanojet characteristics change with angle.

## Key Changes

### 1. Modified `save_front_row_pnjs()` Function
- Now **returns** intensity data and coordinate arrays in a dictionary
- Data structure returned:
  ```python
  {
      'intensity': I_tiled,      # 2D array of normalized intensity
      'xs': xs,                   # x-coordinates
      'zs': zs,                   # z-coordinates
      'I_crop': I_crop,           # Cropped intensity for visualization
      'zs_crop': zs_crop,         # Cropped z-coordinates
      'theta': theta_deg          # Exposure angle
  }
  ```

### 2. New Function: `plot_intensity_vs_angle()`
Creates 4 different comparison plots:

1. **Maximum Intensity vs Angle** (`max_intensity_vs_angle_*.png`)
   - Simple line plot showing how peak intensity varies with angle
   - Useful for finding optimal exposure angle

2. **Axial Intensity Profiles** (`z_profile_comparison_*.png`)
   - Overlaid z-axis profiles (at x=0) for all angles
   - Shows PNJ penetration depth and focus position
   - Includes markers for sphere center and substrate surface

3. **2D Intensity Heatmap** (`intensity_heatmap_vs_angle_*.png`)
   - Color map showing intensity as function of (z-position, angle)
   - Excellent for visualizing how PNJ shape changes with angle
   - White dashed line = sphere center, yellow = substrate

4. **Substrate Surface Intensity** (`substrate_intensity_vs_angle_*.png`)
   - Intensity specifically at the substrate surface (z = -r)
   - Critical for ablation/exposure applications

### 3. Data Persistence Functions

- `save_angle_data()`: Saves all simulation data to `.npz` file
- `load_angle_data()`: Loads previously saved data
- Useful for re-generating plots without re-running simulations

## Usage

### Basic Usage (Run as-is)
```bash
python pnj_angle_analysis.py
```

This will:
1. Run simulations at 15°, 23.5°, and 30°
2. Generate individual PNJ visualizations for each angle
3. Generate 4 comparison plots
4. Save all data to `pnj_angle_data.npz`

### Modify Angles
Edit line 227:
```python
angles = [15.0, 23.5, 30.0]  # Add/remove angles as needed
```

Example for finer resolution:
```python
angles = np.linspace(10, 40, 7)  # 10°, 15°, 20°, 25°, 30°, 35°, 40°
```

### Re-generate Plots from Saved Data
```python
# Load previously saved data
angle_data = load_angle_data("pnj_angle_data.npz")

# Regenerate comparison plots
plot_intensity_vs_angle(angle_data)
```

## Output Files

After running, you'll get:

### Individual Angle Visualizations
- `hcp_frontrow_pnj_theta15.0_power40mW.png`
- `hcp_frontrow_pnj_theta23.5_power40mW.png`
- `hcp_frontrow_pnj_theta30.0_power40mW.png`

### Comparison Plots
- `max_intensity_vs_angle_power40mW.png` - Peak intensity trends
- `z_profile_comparison_power40mW.png` - Axial profiles overlay
- `intensity_heatmap_vs_angle_power40mW.png` - 2D angle-depth map
- `substrate_intensity_vs_angle_power40mW.png` - Surface intensity

### Data File
- `pnj_angle_data.npz` - All simulation data for later analysis

## Analysis Tips

1. **Finding Optimal Angle**: Look at `substrate_intensity_vs_angle_*.png` to find the angle that maximizes intensity at the substrate

2. **PNJ Penetration Depth**: Use `z_profile_comparison_*.png` to see how deep the PNJ penetrates into the substrate

3. **Focus Position**: The heatmap shows where the intensity peak occurs relative to the sphere/substrate interface

4. **Uniformity**: Compare the width/shape of PNJs at different angles in the individual visualizations

## Customization

### Add More Comparison Metrics

You can add custom analysis by accessing the `angle_data` dictionary:

```python
# Example: Extract intensity at specific z-position for all angles
z_target = -1.0  # µm below sphere center
for ang, data in angle_data.items():
    z_idx = np.argmin(np.abs(data['zs'] - z_target))
    intensity_at_z = data['intensity'][:, z_idx]
    print(f"Angle {ang}°: mean intensity at z={z_target} = {intensity_at_z.mean():.3f}")
```

### Change Visualization Style

Modify the plotting functions to change:
- Colormaps (e.g., `cmap="viridis"` instead of `"turbo"`)
- Figure sizes and DPI
- Line styles and colors
- Axis ranges and labels

## Performance Notes

- Each simulation takes ~2-5 minutes depending on resolution
- Total runtime for 3 angles: ~10-15 minutes
- Saved `.npz` file is typically 5-20 MB depending on number of angles
- Comparison plots generate instantly from saved data

## Next Steps

Consider analyzing:
- Intensity at different x-positions (off-axis)
- Full width at half maximum (FWHM) of PNJ vs angle
- Aspect ratio of PNJ (width vs depth)
- Energy deposition in substrate vs angle
