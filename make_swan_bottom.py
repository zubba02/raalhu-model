#!/usr/bin/env python3
"""
Convert a scattered bathymetry CSV (x,y,bathy) into a SWAN
INPGRID BOTTOM REGULAR file + print the required SWAN commands.

Assumes:
  - CSV has a header line
  - Columns are: x, y, bathy
  - Values are elevation (negative under water)  →  use --elevation
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_2d
from scipy.ndimage import distance_transform_edt


def main():
    parser = argparse.ArgumentParser(description="Create SWAN REGULAR bottom file")
    parser.add_argument("bathy_file", help="Input CSV (x,y,bathy)")
    parser.add_argument("-o", "--output", default="bottom.bot", help="Output bottom file")
    parser.add_argument("-r", "--resolution", type=float, default=50.0,
                        help="Grid resolution in metres (default 50)")
    parser.add_argument("--elevation", action="store_true",
                        help="Input values are elevation (negative under water)")
    parser.add_argument("--exc", type=float, default=-999.0,
                        help="Exception value for dry/land points (default -999)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"Loading {args.bathy_file} ...")
    df = pd.read_csv(args.bathy_file)
    x = df.iloc[:, 0].to_numpy(dtype=float)
    y = df.iloc[:, 1].to_numpy(dtype=float)
    z = df.iloc[:, 2].to_numpy(dtype=float)
    print(f"  {len(x):,} points")

    # Convert elevation → depth (positive down) if needed
    if args.elevation:
        z = -z
        print("  Converted elevation → depth (positive down)")

    # ------------------------------------------------------------------
    # 2. Create regular grid by binning
    # ------------------------------------------------------------------
    dx = dy = args.resolution
    x_edges = np.arange(x.min(), x.max() + dx, dx)
    y_edges = np.arange(y.min(), y.max() + dy, dy)

    print(f"Binning onto {dx:.1f} m grid ...")
    Z, x_edge, y_edge, _ = binned_statistic_2d(
        x, y, z, statistic="mean", bins=[x_edges, y_edges]
    )

    # Fill empty bins with nearest neighbour
    mask = np.isnan(Z)
    if mask.any():
        ind = distance_transform_edt(mask, return_distances=False, return_indices=True)
        Z = Z[tuple(ind)]

    # Grid parameters for SWAN
    xpinp = x_edges[0]          # origin x
    ypinp = y_edges[0]          # origin y
    alpinp = 0.0                # rotation (degrees)
    mxinp = Z.shape[0] - 1      # number of meshes in x  (points-1)
    myinp = Z.shape[1] - 1      # number of meshes in y
    dxinp = dx
    dyinp = dy

    # SWAN wants the matrix with shape (myinp+1, mxinp+1)
    # and written in a specific order (idla)
    depth = Z.T.copy()          # now (ny, nx)

    # Optional: set very shallow / land points to exception value
    # (uncomment if you want)
    # depth[depth < 0.05] = args.exc

    # ------------------------------------------------------------------
    # 3. Write the bottom file (FREE format, one value per line is safest)
    # ------------------------------------------------------------------
    print(f"Writing {args.output} ...")
    with open(args.output, "w") as f:
        # Write row-by-row starting from the bottom-left (idla=1 style)
        for j in range(depth.shape[0]):
            for i in range(depth.shape[1]):
                f.write(f"{depth[j, i]:.4f}\n")

    nx = depth.shape[1]   # number of points in x
    ny = depth.shape[0]   # number of points in y

    print("\n" + "="*60)
    print("SWAN commands to put in your .swn file:")
    print("="*60)
    print()
    print(f"INPGRID BOTTOM REGULAR {xpinp:.3f} {ypinp:.3f} {alpinp:.1f} "
          f"{mxinp} {myinp} {dxinp:.3f} {dyinp:.3f}")
    print()
    print(f"READINP BOTTOM 1.0 '{args.output}' 1 0 FREE")
    print()
    print("="*60)
    print(f"Grid info:")
    print(f"  Origin          : ({xpinp:.3f}, {ypinp:.3f})")
    print(f"  Rotation        : {alpinp} deg")
    print(f"  Meshes (mx, my) : {mxinp}, {myinp}")
    print(f"  Points (nx, ny) : {nx}, {ny}")
    print(f"  Resolution      : {dxinp} m × {dyinp} m")
    print(f"  Depth range     : {depth.min():.2f} … {depth.max():.2f} m")
    print("="*60)


if __name__ == "__main__":
    main()
