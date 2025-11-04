# PNJ Wavefront Animation

This script creates a video animation showing how a laser pulse interacts with a single silica microsphere to create a photonic nanojet (PNJ).

## Features

- **Real-time wavefront visualization**: Watch the laser pulse approach, interact with, and pass through the sphere
- **Single sphere geometry**: Simplified from the HCP array to focus on one particle
- **Video output**: Generates MP4 (if ffmpeg available) or GIF animation
- **Key frame extraction**: Saves individual frames as PNG images for presentations
- **Same physics**: Uses identical laser parameters and boundary conditions as the main simulation

## Quick Start

```bash
python pnj_wavefront_animation.py
```

**Runtime:** ~5-10 minutes depending on resolution

## Output Files

1. **Video animation**: `pnj_wavefront_animation_theta23.5_res10.mp4` (or `.gif`)
   - Shows full time evolution of the electromagnetic field
   - 120 frames at 15 FPS (~8 second video)

2. **Individual frames**: `frames_theta23.5_res10/frame_XXXX.png`
   - ~20 key frames saved as high-resolution PNGs
   - Perfect for presentations or publications

## Customization

### Change Incident Angle
```python
angle = 23.5     # Change this line (degrees)
```

### Adjust Video Quality
```python
resolution = 10  # Spatial resolution (pixels/µm)
                 # Higher = better quality but slower
                 # Recommended: 8-12

n_frames = 120   # Number of frames
                 # More frames = smoother animation

fps = 15         # Frames per second for playback
```

### Change Simulation Duration
```python
T_RUN = 120      # Total MEEP time units
                 # Increase to see longer time evolution
```

### Visualization Settings

**Colormap options** (line 198):
```python
cmap="hot"       # Try: "turbo", "plasma", "viridis", "inferno"
```

**Intensity scaling** (line 195):
```python
vmax = np.percentile(all_intensities, 99.5)  # Adjust percentile for contrast
```

**View window** (lines 181-182):
```python
z_top_crop = +3.5   # Top of viewing window (µm)
z_bot_crop = -9.0   # Bottom of viewing window (µm)
```

## What You'll See

The animation shows:

1. **Laser pulse approach** (early frames): Wavefront propagating from top
2. **Sphere interaction** (middle frames):
   - Light focusing inside the sphere
   - Refraction at sphere boundaries
   - PNJ formation below the sphere
3. **PNJ penetration** (later frames):
   - Intense focal spot below sphere
   - Light penetrating into sapphire substrate
   - Gradual decay as pulse passes

### Key Visual Elements

- **Cyan circle**: Silica sphere outline (4.3 µm diameter)
- **Orange line**: Sapphire substrate surface
- **Gray shading**: Sapphire material
- **Hot colormap**: Red = highest intensity, dark = low intensity

## Technical Details

### Geometry
- **Single sphere**: 4.3 µm diameter silica (n=1.45)
- **Substrate**: Sapphire (n=1.75), 10 µm thick
- **Cell size**: 12.9 µm × 12.9 µm × ~19.5 µm
- **Slice view**: XZ plane at y=0 (through sphere center)

### Laser Parameters
- **Wavelength**: 800 nm
- **Pulse width**: 45 fs FWHM
- **Beam waist**: 56 µm
- **Polarization**: Ey (into the page)
- **Default angle**: 23.5° from vertical

### Boundary Conditions
- **PML boundaries**: 1.2 µm thick on all sides
- **X and Y**: Absorbing (PML) - no periodicity
- **Z**: Absorbing (PML) on top and bottom

## Performance Tips

### Fast preview (low quality):
```python
resolution = 6
n_frames = 60
T_RUN = 80
```
Runtime: ~2 minutes

### High quality (publication):
```python
resolution = 14
n_frames = 200
T_RUN = 140
```
Runtime: ~20-30 minutes

### Memory usage:
Each frame stores a 2D array of size `(resolution*sx) × (resolution*sz)`
- res=10, 120 frames: ~50 MB memory
- res=14, 200 frames: ~150 MB memory

## Troubleshooting

### "FFmpeg not found"
The script automatically falls back to GIF format. To enable MP4:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# conda
conda install -c conda-forge ffmpeg
```

### Animation runs too fast/slow
Adjust the `fps` parameter (frames per second):
```python
fps = 10   # Slower playback
fps = 30   # Faster playback
```

### Can't see the PNJ
- Increase `T_RUN` to allow more time for pulse to propagate
- Check `z_bot_crop` extends low enough to see below sphere
- Adjust `vmax` percentile for better contrast

### Simulation takes too long
- Reduce `resolution` (try 8 instead of 10)
- Reduce `n_frames` (try 60 instead of 120)
- Reduce `T_RUN` (try 80 instead of 120)

## Comparison with Main Simulations

| Feature | This Script | Main HCP Simulations |
|---------|-------------|---------------------|
| Geometry | Single sphere | Periodic HCP array |
| Boundary (X,Y) | PML (absorbing) | Bloch-periodic |
| Purpose | Visualization | Quantitative analysis |
| Runtime | 5-10 min | 10-15 min per angle |
| Output | Video animation | Static field plots |

## Example Use Cases

1. **Understanding PNJ physics**: See how light focuses below the sphere in real-time
2. **Presentation videos**: Show wavefront propagation in talks
3. **Educational content**: Demonstrate electromagnetic wave behavior
4. **Angle comparison**: Generate videos at different angles to compare
5. **Parameter exploration**: Visualize effect of changing beam waist, wavelength, etc.

## Advanced: Batch Generation

Create animations for multiple angles:

```python
angles_to_test = [15, 22.5, 30, 37.5, 45]

for angle in angles_to_test:
    # Run simulation with current angle
    # ... (modify main code to loop)
```

Then compare videos side-by-side to see how PNJ changes with angle.

## Citation

If using this for publications, please cite:
- MEEP: Oskooi et al., Computer Physics Communications 181, 687 (2010)

## Notes

- The animation shows **intensity** (|E|²), not field components
- Time units are in MEEP's natural units (related to 1/frequency)
- The colormap is normalized to the maximum intensity across all frames
- Sphere appears stationary while wavefront moves through it
