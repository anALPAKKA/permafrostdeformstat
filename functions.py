import xarray as xr
import pandas as pd
import numpy as np
import numpy.ma as ma
import netCDF4 as nc
from netCDF4 import Dataset
import scipy.io as io
import scipy.stats as st
from scipy.stats import pearsonr
from scipy.stats import ks_2samp
from scipy.stats import skew
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import random
import cmocean
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyproj import Geod








# ==============================
#   1️⃣  DATA HANDLING FUNCTIONS
# ==============================



def prepare_subsidence_data(loc, year, vartype, min_value=-0.5, max_value=0.5, near_zero_threshold=1e-5):
    ''' Preparing subsidence data while ensuring matching grid sizes. '''
    filepath = filepaths[f"{loc}_{year}_{vartype}"] 
    ds = Dataset(filepath, mode='r')

    lon_var_name = 'lon' if 'lon' in ds.variables else 'longitude'
    lat_var_name = 'lat' if 'lat' in ds.variables else 'latitude'
    
    lons = ds.variables[lon_var_name][:]
    lats = ds.variables[lat_var_name][:]
    subsidence = ds.variables['Band1'][:]

    # Creating meshgrid for lons and lats
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    
    # Flatten the arrays
    lons_flat = lon_mesh.ravel()
    lats_flat = lat_mesh.ravel()
    subsidence_flat = subsidence.ravel()
    zip_data = np.column_stack((lons_flat, lats_flat, subsidence_flat))


    # Check if all arrays have the same length
    if len(lons_flat) != len(subsidence_flat):
        raise ValueError(f"Mismatch: lons_flat ({len(lons_flat)}), subsidence_flat ({len(subsidence_flat)})")

    # Create a mask for valid data
    mask = (
        (subsidence_flat >= min_value) &
        (subsidence_flat <= max_value) &
        ~((subsidence_flat > -near_zero_threshold) & (subsidence_flat < near_zero_threshold))
    )

    # Remove data value at 0 with the mask
    zip_masked = zip_data[mask]

    ds.close()
    return subsidence, lons, lats, zip_masked



def prepare_ERA_data(loc, year, vartype, varname):
    '''Prepare non-subsidence data'''

    # Load the dataset
    filepath = filepaths[f"{loc}_{year}_{vartype}"]
    ds = Dataset(filepath, mode='r')

    # Determine the correct variable names for longitude and latitude
    lon_var_name = 'lon' if 'lon' in ds.variables else 'longitude'
    lat_var_name = 'lat' if 'lat' in ds.variables else 'latitude'

    # Load longitude and latitude arrays
    lons = ds.variables[lon_var_name][:]
    lats = ds.variables[lat_var_name][:]

    # Extract the variable data
    var_data = ds.variables[varname][...]  # Load entire variable safely

    # Handle different number of dimensions
    if var_data.ndim == 3:  # Standard (time, lat, lon)
        var_data = var_data[0, :, :]  # Take the first time step
        print(f"Warning: Variable {varname} is 3D. Only the first time step is used.")
    elif var_data.ndim == 2:  # Already (lat, lon), no need to slice
        pass
    elif var_data.ndim == 1:  # 1D variable, may need reshaping
        print(f"Warning: Variable {varname} is 1D. Manual reshaping may be needed.")

    ds.close()

    return var_data, lons, lats


# ==============================
#   2️⃣  GRIDDIND AND BINNING
# ==============================

# Rregroup_and_generate_histogram is not used, can be deleted
def regroup_and_generate_histograms(grid, data, bins=50, min_val=-0.02, max_val=0.02, threshold=6000, plot=True):
    '''Regroup data into subsets based on grid and generate binned histograms.'''
    histograms = []
    grid_info = []  # Keep track of grid location and valid data points
    
    for idx, cell in enumerate(grid):
        lon_start, lon_end, lat_start, lat_end = cell
        
        # Mask data for the current grid cell
        mask = (data[:, 0] >= lon_start) & (data[:, 0] < lon_end) & \
               (data[:, 1] >= lat_start) & (data[:, 1] < lat_end)
        subset = data[mask][:, 2]
        
        # Debugging: Log grid cell details
        print(f"Grid {idx}: Lon {lon_start}-{lon_end}, Lat {lat_start}-{lat_end}, Points: {len(subset)}")
        
        if len(subset) > 0:
            # Compute histogram for the subset
            hist, _ = np.histogram(subset, bins=bins, range=(min_val, max_val), density=True)
            
            # Debugging: Log histogram details
            print(f"Histogram for Grid {idx}: {hist}")
            
            # Threshold check
            if len(subset) >= threshold:
                histograms.append(hist)  # Include histogram only if it has enough data
            else:
                histograms.append(np.full(bins, np.nan))  # Mark histograms with insufficient data as NaN
        else:
            histograms.append(np.full(bins, np.nan))  # Mark empty grids as NaN
        
        # Append grid info
        grid_info.append({
            "lon_start": lon_start,
            "lon_end": lon_end,
            "lat_start": lat_start,
            "lat_end": lat_end,
            "num_points": len(subset)
        })

        # Plot the histogram if required
        if plot and len(subset) > 0:
            plt.figure(figsize=(8, 6))
            plt.hist(subset, bins=bins, range=(min_val, max_val), edgecolor='black')
            plt.title(f'Grid Cell Histogram (Lon: {lon_start:.2f}-{lon_end:.2f}, Lat: {lat_start:.2f}-{lat_end:.2f})\nPoints: {len(subset)}')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.show()

    # Debugging: Log final histogram shape and contents
    print(f"Final histogram array shape: {np.array(histograms).shape}")
    print(f"Histogram array contents: {np.array(histograms)}")
    
    return np.array(histograms), grid_info


def create_km_grid(lon_mesh, lat_mesh, resolution_km=10, output_file='grid_description.nc'):
    '''
    Create a new grid with a specified resolution in kilometers.

    Parameters:
        lon_mesh (ndarray): Original longitude mesh.
        lat_mesh (ndarray): Original latitude mesh.
        resolution_km (float): Desired grid resolution in kilometers.

    Returns:
        grid (list of tuples): List of grid cells as (lon_start, lon_end, lat_start, lat_end).
    '''
    geod = Geod(ellps="WGS84")

    # Find the bounds of the original data
    lon_min, lon_max = lon_mesh.min(), lon_mesh.max()
    lat_min, lat_max = lat_mesh.min(), lat_mesh.max()

    # Calculate the step size in degrees for the specified resolution in km
    def calculate_step(start, end, resolution, is_latitude):
        steps = [start]
        current = start

        while current < end:
            if is_latitude:
                # Latitude distances are approximately constant
                next_point = geod.fwd(lon_mesh.mean(), current, 0, resolution * 1000)[1]
            else:
                # Longitude distances vary with latitude
                next_point = geod.fwd(current, lat_mesh.mean(), 90, resolution * 1000)[0]

            steps.append(next_point)
            current = next_point

        return np.array(steps)

    lat_steps = calculate_step(lat_min, lat_max, resolution_km, is_latitude=True)
    lon_steps = calculate_step(lon_min, lon_max, resolution_km, is_latitude=False)

    # Create the grid as a list of bounding boxes
    grid = []
    for i in range(len(lat_steps) - 1):
        for j in range(len(lon_steps) - 1):
            grid.append((lon_steps[j], lon_steps[j + 1], lat_steps[i], lat_steps[i + 1]))


    return grid



def bin_valid_data_with_stats(
    grid,
    data,
    bins=50,
    min_val=None,
    max_val=None,
    threshold=6000,
    chunk_size=5_000_000,
):
    grid_array = np.asarray(grid, dtype=np.float64)
    n_cells = len(grid_array)

    # Recover grid edges from create_km_grid().
    lon_edges = np.unique(
        np.concatenate((grid_array[:, 0], grid_array[:, 1]))
    )
    lat_edges = np.unique(
        np.concatenate((grid_array[:, 2], grid_array[:, 3]))
    )

    nx = len(lon_edges) - 1
    ny = len(lat_edges) - 1

    if nx * ny != n_cells:
        raise ValueError(
            "Grid is not a complete rectangular grid: "
            f"{nx} × {ny} != {n_cells}"
        )

    if min_val is None:
        min_val = np.nanmin(data[:, 2])

    if max_val is None:
        max_val = np.nanmax(data[:, 2])

    # Per-cell accumulators
    counts = np.zeros(n_cells, dtype=np.int64)
    sum1 = np.zeros(n_cells, dtype=np.float64)
    sum2 = np.zeros(n_cells, dtype=np.float64)
    sum3 = np.zeros(n_cells, dtype=np.float64)
    sum4 = np.zeros(n_cells, dtype=np.float64)

    histogram_counts = np.zeros(
        (n_cells, bins),
        dtype=np.int64,
    )

    histogram_width = (max_val - min_val) / bins

    # Process manageable chunks to avoid very large temporary arrays.
    for start in range(0, len(data), chunk_size):
        stop = min(start + chunk_size, len(data))
        chunk = np.asarray(data[start:stop])

        longitude = chunk[:, 0]
        latitude = chunk[:, 1]
        values = chunk[:, 2]

        ix = np.searchsorted(
            lon_edges, longitude, side="right"
        ) - 1
        iy = np.searchsorted(
            lat_edges, latitude, side="right"
        ) - 1

        spatially_valid = (
            (ix >= 0) & (ix < nx) &
            (iy >= 0) & (iy < ny) &
            np.isfinite(values)
        )

        ix = ix[spatially_valid]
        iy = iy[spatially_valid]
        values = values[spatially_valid]

        # Matches the ordering produced by create_km_grid():
        # latitude outer loop, longitude inner loop.
        cell_id = iy * nx + ix

        counts += np.bincount(
            cell_id,
            minlength=n_cells,
        )

        sum1 += np.bincount(
            cell_id, weights=values, minlength=n_cells
        )
        sum2 += np.bincount(
            cell_id, weights=values**2, minlength=n_cells
        )
        sum3 += np.bincount(
            cell_id, weights=values**3, minlength=n_cells
        )
        sum4 += np.bincount(
            cell_id, weights=values**4, minlength=n_cells
        )

        # np.histogram includes max_val in the final bin.
        histogram_valid = (
            (values >= min_val) &
            (values <= max_val)
        )

        histogram_values = values[histogram_valid]
        histogram_cells = cell_id[histogram_valid]

        bin_id = np.floor(
            (histogram_values - min_val) / histogram_width
        ).astype(np.int64)

        bin_id = np.clip(bin_id, 0, bins - 1)

        combined_id = histogram_cells * bins + bin_id

        histogram_counts += np.bincount(
            combined_id,
            minlength=n_cells * bins,
        ).reshape(n_cells, bins)

    # Preserve original behavior: strictly greater than threshold.
    valid_mask = counts > threshold
    valid_indices = np.flatnonzero(valid_mask)

    histograms = np.full(
        (n_cells, bins),
        np.nan,
        dtype=np.float64,
    )

    histogram_totals = histogram_counts.sum(axis=1)
    histogram_rows_valid = valid_mask & (histogram_totals > 0)

    histograms[histogram_rows_valid] = (
        histogram_counts[histogram_rows_valid]
        / histogram_totals[histogram_rows_valid, None]
        / histogram_width
    )

    stats_array = np.full(
        (n_cells, 4),
        np.nan,
        dtype=np.float64,
    )

    n = counts[valid_mask].astype(np.float64)
    mean = sum1[valid_mask] / n

    raw2 = sum2[valid_mask] / n
    raw3 = sum3[valid_mask] / n
    raw4 = sum4[valid_mask] / n

    variance = np.maximum(raw2 - mean**2, 0.0)
    std = np.sqrt(variance)

    moment3 = raw3 - 3 * mean * raw2 + 2 * mean**3
    moment4 = (
        raw4
        - 4 * mean * raw3
        + 6 * mean**2 * raw2
        - 3 * mean**4
    )

    nonzero_variance = variance > 0

    skewness = np.full_like(mean, np.nan)
    kurtosis = np.full_like(mean, np.nan)

    skewness[nonzero_variance] = (
        moment3[nonzero_variance]
        / variance[nonzero_variance]**1.5
    )
    kurtosis[nonzero_variance] = (
        moment4[nonzero_variance]
        / variance[nonzero_variance]**2
    )

    stats_array[valid_mask, 0] = mean
    stats_array[valid_mask, 1] = std
    stats_array[valid_mask, 2] = skewness
    stats_array[valid_mask, 3] = kurtosis

    return (
        histograms,
        counts,
        valid_indices,
        stats_array,
    )






def create_cdo_grid_description(grid, output_file='grid_description.txt'):
    """
    Create a CDO-compatible grid description file based on input grid cells for a curvilinear grid.

    Parameters:
        grid (list of tuples): List of grid cells as (lon_start, lon_end, lat_start, lat_end).
        output_file (str): File name for the output grid description text file.

    Returns:
        None
    """

    # Extract unique longitudes and latitudes from the grid
    lon_coords = sorted(set(cell[0] for cell in grid))
    lat_coords = sorted(set(cell[2] for cell in grid))

    # Calculate grid dimensions
    xsize = len(lon_coords)
    ysize = len(lat_coords)
    gridsize = xsize * ysize

    # Prepare grid description text for a curvilinear grid
    description = [f"gridtype = curvilinear", f"xsize = {xsize}", f"ysize = {ysize}", f"gridsize = {gridsize}"]

    # Convert coordinates into 2D arrays (longitude and latitude grid points)
    lon_array, lat_array = np.meshgrid(lon_coords, lat_coords)

    # Write to file
    with open(output_file, 'w') as f:
        f.write("\n".join(description))
        f.write("\n")

        # Write longitude (xvals) and latitude (yvals) arrays
        f.write("xvals = ")
        np.savetxt(f, lon_array.flatten(), fmt="%.6f", newline=" ")
        f.write("\n")

        f.write("yvals = ")
        np.savetxt(f, lat_array.flatten(), fmt="%.6f", newline=" ")
        f.write("\n")

    print(f"Curvilinear grid description file written to {output_file}")


# combine_variables_by_grid is not used for multi-region clustering
# combine_variables_by_grid is used for aligning deformation data with other factors data for correlation analysis
def combine_variables_by_grid(*variables):
    """
    Combine multiple variables into a single 3D matrix based on grid indexing.

    Parameters:
        *variables (ndarrays): A list of 2D arrays (e.g., 27x20) representing different variables
                               aligned to the same grid size.

    Returns:
        ndarray: A 3D array with shape (grid_rows, grid_cols, num_variables),
                 preserving spatial arrangement.
    """
    # Ensure all input variables have the same shape
    shapes = [var.shape for var in variables]
    if len(set(shapes)) != 1:
        raise ValueError("All input variables must have the same shape.")

    # Stack the variables along a new axis (depth dimension)
    combined_matrix = np.stack(variables, axis=-1)  # Shape: (rows, cols, num_variables)

    return combined_matrix  # Preserves 2D grid structure


# summarize_cluster_moments is used for analyzing cluster characteristics
def summarize_cluster_moments(H, labels, bin_edges, from_density=True, title_prefix=""):
    """
    Print min/max ranges of mean, std, skewness, kurtosis for each cluster.

    Parameters
    ----------
    H : (n_cells, n_bins)   histogram rows (densities if from_density=True; else probabilities)
    labels : (n_cells,)     cluster labels for those rows
    bin_edges : (n_bins+1,) global histogram edges
    from_density : bool     True if H are densities (np.histogram(..., density=True))
    """
    H = np.asarray(H, float)
    labels = np.asarray(labels)
    widths  = np.diff(bin_edges)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Densities -> per-bin probabilities, then row-normalize
    if from_density:
        P = H * widths
    else:
        P = H.copy()
    P = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-12)

    # Raw moments E[x^k]
    Ex  = P @ centers
    Ex2 = P @ (centers**2)
    Ex3 = P @ (centers**3)
    Ex4 = P @ (centers**4)

    # Central/standardized moments
    var = np.clip(Ex2 - Ex**2, 0.0, np.inf)
    std = np.sqrt(var)
    mu3 = Ex3 - 3*Ex*Ex2 + 2*(Ex**3)
    mu4 = Ex4 - 4*Ex*Ex3 + 6*(Ex**2)*Ex2 - 3*(Ex**4)

    with np.errstate(invalid="ignore", divide="ignore"):
        skew = mu3 / np.maximum(std**3, 1e-12)
        kurt = mu4 / np.maximum(std**4, 1e-12)  # Pearson kurtosis

    for c in sorted(np.unique(labels)):
        idx = (labels == c)
        n = int(idx.sum())
        if n == 0: 
            continue

        def rng(arr):
            return (np.nanmin(arr[idx]), np.nanmax(arr[idx]))

        m_min, m_max = rng(Ex)
        s_min, s_max = rng(std)
        sk_min, sk_max = rng(skew)
        ku_min, ku_max = rng(kurt)

        print(f"{title_prefix}Cluster {c} (n={n})")
        print(f"  mean (m):     [{m_min:.4f}, {m_max:.4f}]")
        print(f"  std  (m):     [{s_min:.4f}, {s_max:.4f}]")
        print(f"  skew (unit):  [{sk_min:.3f}, {sk_max:.3f}]")
        print(f"  kurt (unit):  [{ku_min:.3f}, {ku_max:.3f}]  (Pearson)")
        print()



# ==============================
#   3️⃣  CLUSTERING FUNCTION
# ==============================
def enforce_size_constraints(H, centroids, min_size=50, max_size=1000):
    """
    Given histogram features H and current centroids, assign each row
    to a cluster with the constraint:
        min_size <= cluster_size <= max_size.

    Uses a greedy repair heuristic:
      1) Start with nearest-centroid assignment.
      2) Fix clusters that are too big by moving borderline points out.
      3) Fix clusters that are too small by pulling in suitable points.
    This function makes practical step-by-step fixes to satisfy cluster-size limits, 
    but it is not guaranteed to find the best possible constrained clustering.

    
    Parameters
    ----------
    H : (n_samples, n_bins)
        Histogram densities for all valid grids.
    centroids : (k, n_bins)
        Current cluster centroids (same space as H).
    min_size, max_size : int
        Desired size bounds.

    Returns
    -------
    labels : (n_samples,)
        Cluster labels satisfying the size constraints (if feasible).
    """
    H = np.asarray(H)
    centroids = np.asarray(centroids)
    n_samples, n_bins = H.shape
    k = centroids.shape[0]

    # Feasibility check
    if n_samples < k * min_size or n_samples > k * max_size:
        raise ValueError(
            f"Infeasible constraints: N={n_samples}, k={k}, "
            f"min_size={min_size}, max_size={max_size}"
        )

    # 1) Distance matrix (Wasserstein between each sample and each centroid)
    dist = np.zeros((n_samples, k))
    for i in range(n_samples):
        for c in range(k):
            dist[i, c] = wasserstein_distance(H[i], centroids[c])

    # 2) Initial unconstrained assignment: nearest centroid
    labels = np.argmin(dist, axis=1)
    sizes = np.bincount(labels, minlength=k)

    # -------------------------
    # Phase A: shrink oversized
    # -------------------------
    changed = True
    while changed:
        changed = False
        for c in range(k):
            while sizes[c] > max_size:
                # All points currently in cluster c
                idx_c = np.where(labels == c)[0]
                if len(idx_c) == 0:
                    break

                best_idx = None
                best_new_cluster = None
                best_delta = np.inf

                # Look for a point to move out of c to some other cluster
                for i in idx_c:
                    # Sort clusters by distance for this point
                    order = np.argsort(dist[i])
                    for c2 in order:
                        if c2 == c:
                            continue
                        if sizes[c2] >= max_size:
                            continue  # can't move there, it would overflow
                        delta = dist[i, c2] - dist[i, c]
                        if delta < best_delta:
                            best_delta = delta
                            best_idx = i
                            best_new_cluster = c2
                        break  # next nearest cluster for this point is worse

                if best_idx is None:
                    # No feasible move found without breaking max_size elsewhere
                    break

                old = labels[best_idx]
                labels[best_idx] = best_new_cluster
                sizes[old] -= 1
                sizes[best_new_cluster] += 1
                changed = True

    # -------------------------
    # Phase B: grow undersized
    # -------------------------
    changed = True
    while changed:
        changed = False
        for c in range(k):
            while sizes[c] < min_size:
                best_idx = None
                best_from = None
                best_delta = np.inf

                # Candidates: points not currently in c
                candidates = np.where(labels != c)[0]
                if len(candidates) == 0:
                    break

                for i in candidates:
                    from_cluster = labels[i]
                    # Don't starve the source cluster below min_size
                    if sizes[from_cluster] - 1 < min_size:
                        continue

                    delta = dist[i, c] - dist[i, from_cluster]
                    if delta < best_delta:
                        best_delta = delta
                        best_idx = i
                        best_from = from_cluster

                if best_idx is None:
                    # No legal move found; can't fill this cluster to min_size
                    break

                labels[best_idx] = c
                sizes[best_from] -= 1
                sizes[c] += 1
                changed = True

    # Final sanity check
    sizes = np.bincount(labels, minlength=k)
    if np.any(sizes < min_size) or np.any(sizes > max_size):
        print("⚠️ Could not fully satisfy size constraints; final sizes:", sizes)
    else:
        print("✅ Cluster sizes:", sizes)

    return labels


class ShapeClustering:
    '''# Shape Clustering'''
    def __init__(self, n_clusters, max_iter=100, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.random_state = random_state
        self.centroids = None
        self.assignments = None
        self.n_iter_ = None
        self.centroid_shift_history_ = []


    def fit(self, histograms):
        # Initialize centroids using KMeans++
        kmeans_init = KMeans(
            n_clusters=self.n_clusters, 
            init="k-means++", 
            random_state=self.random_state
        ).fit(histograms)

        self.centroids = kmeans_init.cluster_centers_
        self.centroid_shift_history_ = []

        for iteration in range(self.max_iter):
            # Assign each histogram to the nearest centroid based on EMD
            assignments = np.array([
                np.argmin([wasserstein_distance(hist, centroid) for centroid in self.centroids])
                for hist in histograms
            ])

            # Update centroids
            new_centroids = []
            for i in range(self.n_clusters):
                cluster_data = histograms[assignments == i]
                if len(cluster_data) > 0:
                    new_centroids.append(np.mean(cluster_data, axis=0))
                else:
                    new_centroids.append(self.centroids[i])  # Keep old centroid if cluster is empty
            
            new_centroids = np.array(new_centroids)

            # Check for convergence
            centroid_shift = np.linalg.norm(self.centroids - new_centroids, axis=1).max()
            self.centroid_shift_history_.append(centroid_shift)
            self.centroids = new_centroids
            if centroid_shift < self.tol:
                break

        self.assignments = assignments
        self.n_iter_ = iteration + 1

        return self

    def predict(self, histograms):
        return np.array([
            np.argmin([wasserstein_distance(hist, centroid) for centroid in self.centroids])
            for hist in histograms
        ])
    




# sensitivity test and convergence check
def run_random_state_sensitivity(histograms, n_clusters=2, random_states=range(50)):
    """
    Run ShapeClustering over multiple random initializations and compare label stability.
    """

    labels_list = []
    centroid_list = []
    n_iter_list = []
    final_shift_list = []

    for seed in random_states:
        model = ShapeClustering(
            n_clusters=n_clusters,
            max_iter=100,
            tol=1e-4,
            random_state=seed
        )

        model.fit(histograms)

        labels_list.append(model.assignments)
        centroid_list.append(model.centroids)
        n_iter_list.append(model.n_iter_)
        final_shift_list.append(model.centroid_shift_history_[-1])

    # Pairwise similarity between clustering results
    rows = []

    for i in range(len(random_states)):
        for j in range(i + 1, len(random_states)):
            ari = adjusted_rand_score(labels_list[i], labels_list[j])
            nmi = normalized_mutual_info_score(labels_list[i], labels_list[j])

            rows.append({
                "seed_i": random_states[i],
                "seed_j": random_states[j],
                "ARI": ari,
                "NMI": nmi,
            })

    pairwise_df = pd.DataFrame(rows)

    summary = {
        "n_runs": len(random_states),
        "mean_ARI": pairwise_df["ARI"].mean(),
        "min_ARI": pairwise_df["ARI"].min(),
        "std_ARI": pairwise_df["ARI"].std(),
        "mean_NMI": pairwise_df["NMI"].mean(),
        "min_NMI": pairwise_df["NMI"].min(),
        "std_NMI": pairwise_df["NMI"].std(),
        "mean_n_iter": np.mean(n_iter_list),
        "max_n_iter": np.max(n_iter_list),
        "mean_final_shift": np.mean(final_shift_list),
        "max_final_shift": np.max(final_shift_list),
    }

    summary_df = pd.DataFrame([summary])

    return labels_list, centroid_list, pairwise_df, summary_df



# ==============================
#   4️⃣  VISUALIZATION FUNCTIONS
# ==============================



def plot_with_colorbar(data, lon, lat, title="Subsidence Data", cmap='curl', levels='sub', label='Subsidence (m)', vmin=None, vmax=None):
    '''Plot data with a color bar, correctly labeled longitude/latitude axes, and adjusted aspect ratio'''

    # Ensure lon and lat are 2D arrays
    if len(lon.shape) == 1 and len(lat.shape) == 1:
        lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    else:
        lon_mesh, lat_mesh = lon, lat

    # Use Geod to calculate the distance in kilometers
    geod = Geod(ellps="WGS84")
    lat_km_span = geod.line_length([lon_mesh.min(), lon_mesh.min()],
                                   [lat_mesh.min(), lat_mesh.max()]) / 1000
    lon_km_span = geod.line_length([lon_mesh.min(), lon_mesh.max()],
                                   [lat_mesh.mean(), lat_mesh.mean()]) / 1000

    # Calculate grid size in km
    grid_lat_km = geod.line_length([lon_mesh[0, 0], lon_mesh[0, 0]], [lat_mesh[0, 0], lat_mesh[1, 0]]) / 1000
    grid_lon_km = geod.line_length([lon_mesh[0, 0], lon_mesh[0, 1]], [lat_mesh[0, 0], lat_mesh[0, 0]]) / 1000

    # Print the grid size and the longitude/latitude spans
    print(f"Latitude span: {lat_km_span:.2f} km")
    print(f"Longitude span: {lon_km_span:.2f} km")
    print(f"Grid size: {grid_lat_km:.2f} km x {grid_lon_km:.2f} km")

    # Calculate the aspect ratio based on the span in kilometers
    aspect_ratio = lat_km_span / lon_km_span

    # Set the color map
    if isinstance(cmap, str):  
        if cmap in cmocean.cm.__dict__:
            cmap = cmocean.cm.__dict__[cmap]
        elif cmap in plt.colormaps():
            cmap = plt.get_cmap(cmap)
        else:
            raise ValueError(f"Invalid colormap: '{cmap}' is not in cmocean or matplotlib.")
  

    # Define discrete levels
    if levels == 'sub':
        if vmin is None and vmax is None:
            levels = np.linspace(-0.1, 0.1, 11)
        else:
            levels = np.linspace(vmin, vmax, 11)
        label = 'Subsidence (m)'
    elif levels == 'None':
        if vmin is None and vmax is None:
            vmin = data.min()
            vmax = data.max()
        levels = np.linspace(vmin, vmax, 11)

    # Create a normalization that splits the color map into discrete intervals
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    # Create the plot with proportional aspect ratio
    fig, ax = plt.subplots(figsize=(10, 10 * aspect_ratio))

    # Use pcolormesh for plotting
    mesh = ax.pcolormesh(lon_mesh, lat_mesh, data, shading='auto', cmap=cmap, norm=norm)

    # Add a color bar
    cbar = fig.colorbar(mesh, ax=ax, label=label, ticks=levels)

   

    # Shift the grid lines by half a grid size
    lon_interval = (lon_mesh[0, 1] - lon_mesh[0, 0])
    lat_interval = (lat_mesh[1, 0] - lat_mesh[0, 0])

    # Set axis limits and labels
    ylim_min = lat_mesh.min() - lat_interval / 2
    ylim_max = lat_mesh.max() + lat_interval / 2
    xlim_min = lon_mesh.min() - lon_interval / 2
    xlim_max = lon_mesh.max() + lon_interval / 2
    ax.set_xlim([xlim_min, xlim_max])
    ax.set_ylim([ylim_min, ylim_max])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)
    print(f"Longitude range: {xlim_min:.2f} to {xlim_max:.2f}")
    print(f"Latitude range: {ylim_min:.2f} to {ylim_max:.2f}")
    print(f"grid number: {len(lon_mesh[0])} x {len(lat_mesh)}")
    

    # Add shifted grid lines
    for i in range(lat_mesh.shape[0] + 1):
        shifted_lat = lat_mesh[0, 0] + i * lat_interval - lat_interval / 2
        ax.axhline(shifted_lat, color='black', linestyle='--', linewidth=0.5, alpha=0.7)
    for j in range(lon_mesh.shape[1] + 1):
        shifted_lon = lon_mesh[0, 0] + j * lon_interval - lon_interval / 2
        ax.axvline(shifted_lon, color='black', linestyle='--', linewidth=0.5, alpha=0.7)

    # Set the aspect ratio of the actual data plot
    ax.set_box_aspect(aspect_ratio)  # aspect_ratio is width/height

    plt.show()




def visualize_clustering(
    grid,
    assignments,
    valid_histograms,
    valid_grids_mask,
    data_counts,
    n_clusters,
    title="Clustering Result Visualization",
    legend_loc="upper right",
    legend_bbox=(1.3, 1),
    threshold=3000
):

    '''
    Visualize the clustering result with grids that have less than 3000 data points marked as grey,
    and include histogram plots for each cluster.

    Parameters:
        grid (list of tuples): List of grid cells as (lon_start, lon_end, lat_start, lat_end).
        assignments (ndarray): Cluster assignments for valid grids.
        valid_histograms (ndarray): Normalized histograms for valid grids.
        valid_grids_mask (ndarray): Boolean mask for grids with valid histograms.
        data_counts (list): Number of valid data points in each grid.
        n_clusters (int): Number of clusters.
        title (str): Title of the plot.
    '''
    # Define custom colors for clusters and excluded grids
    custom_colors = [
        "#416E6F",  # Green
        "#EC9F72",  # Orange
        "#8D91C0",  # Purple
        "#DED88B",  # Yellow
        "#A26C66",  # Red-Brown
    ]
    custom_colors = custom_colors[:n_clusters]  # Adjust to the number of clusters
    cmap = {i: custom_colors[i] for i in range(n_clusters)}

    # White for grids with NaN histograms and grey for excluded grids
    cmap[-1] = "white"  # No Data (NaN histograms)
    cmap[-2] = "grey"   # Excluded grids (<threshold data points)

    geod = Geod(ellps="WGS84")

    # Calculate the geographic extent in kilometers
    lat_min = min([cell[2] for cell in grid])
    lat_max = max([cell[3] for cell in grid])
    lon_min = min([cell[0] for cell in grid])
    lon_max = max([cell[1] for cell in grid])

    width_km = geod.line_length([lon_min, lon_max], [lat_min, lat_min]) / 1000
    height_km = geod.line_length([lon_min, lon_min], [lat_min, lat_max]) / 1000

    # Create a plot with proportional aspect ratio
    fig, ax = plt.subplots(figsize=(10, 10 * height_km / width_km))

    for i, cell in enumerate(grid):
        lon_start, lon_end, lat_start, lat_end = cell
        if not valid_grids_mask[i]:  # Grids with NaN histograms or excluded grids
            cluster = -1 if data_counts[i] == 0 else -2
        else:
            valid_index = np.sum(valid_grids_mask[:i])  # Correctly map to valid assignments
            cluster = assignments[valid_index]

        # Draw the grid cell with the corresponding color
        rect = plt.Rectangle((lon_start, lat_start), lon_end - lon_start, lat_end - lat_start,
                             facecolor=cmap[cluster], edgecolor='black', lw=0.5)
        ax.add_patch(rect)

    # Set axis limits and labels
    ax.set_xlim([lon_min, lon_max])
    ax.set_ylim([lat_min, lat_max])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)

    # Create a legend for the clusters
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, facecolor=cmap[i], edgecolor='black') for i in range(n_clusters)
    ]
    legend_labels = [f"Cluster {i}" for i in range(n_clusters)]
    legend_patches.append(plt.Rectangle((0, 0), 1, 1, facecolor="grey", edgecolor='black'))
    legend_labels.append(f"Excluded Grids (<{threshold} points)")
    legend_patches.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor='black'))
    legend_labels.append("No Data")

    ax.legend(legend_patches, legend_labels, loc='upper right', bbox_to_anchor=(1.3, 1))

    plt.show()




def visualize_data_density(grid, data_counts, title="Valid Data Points per Grid", cmap=cmocean.cm.dense,vmax=140000,lvl=11):
    """
    Visualizes the number of valid data points per 10×10 km grid cell,
    ensuring correct geographic distance representation.
    Uses the same grid-based plotting scheme as clustering and subsidence plots.
    """

    geod = Geod(ellps="WGS84")

    # Calculate the geographic extent in kilometers
    lat_min = min([cell[2] for cell in grid])
    lat_max = max([cell[3] for cell in grid])
    lon_min = min([cell[0] for cell in grid])
    lon_max = max([cell[1] for cell in grid])

    width_km = geod.line_length([lon_min, lon_max], [lat_min, lat_min]) / 1000
    height_km = geod.line_length([lon_min, lon_min], [lat_min, lat_max]) / 1000
    print(f"Width: {width_km:.2f} km, Height: {height_km:.2f} km")
    aspect_ratio = 1.0

    # Normalize color scale based on max value
    levels = np.linspace(0, vmax, lvl)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    # Create the plot with proportional aspect ratio
    fig, ax = plt.subplots(figsize=(10, 10 * height_km / width_km), constrained_layout=True)

    for i, cell in enumerate(grid):
        lon_start, lon_end, lat_start, lat_end = cell
        count = data_counts[i]

        # Define grid cell color based on data count
        if np.isnan(count) or count == 0:
            facecolor = "white"  # Empty grids
            edgecolor = "black"
        else:
            facecolor = cmap(norm(count))#cmocean.cm.__dict__.get(cmap, cmocean.cm.matter)(norm(count))
            edgecolor = "black"

        # Draw the grid cell
        ax.add_patch(plt.Rectangle((lon_start, lat_start), lon_end - lon_start, lat_end - lat_start,
                                   facecolor=facecolor, edgecolor=edgecolor, lw=0.5))

    # Set axis limits and labels
    ax.set_xlim([lon_min, lon_max])
    ax.set_ylim([lat_min, lat_max])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)


    # Create a divider for the existing axes instance
    divider = make_axes_locatable(ax)
    # Append axes to the right of ax, with 5% width of ax
    cax = divider.append_axes("right", size="5%", pad=0.2)


    # Add colorbar based on data values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax, label="Number of Data Points")

    

    plt.show()




def visualise_grid_results(grid, data_counts,title="Grid Visualization",limit=6000):
    '''
    Visualise grid results based on the number of data points in each grid.

    Parameters:
        grid (list of tuples): List of grid cells as (lon_start, lon_end, lat_start, lat_end).
        data_counts (list): Number of data points in each grid cell.
    '''
    geod = Geod(ellps="WGS84")

    # Calculate the geographic extent in kilometers
    lat_min = min([cell[2] for cell in grid])
    lat_max = max([cell[3] for cell in grid])
    lon_min = min([cell[0] for cell in grid])
    lon_max = max([cell[1] for cell in grid])

    width_km = geod.line_length([lon_min, lon_max], [lat_min, lat_min]) / 1000
    height_km = geod.line_length([lon_min, lon_min], [lat_min, lat_max]) / 1000

    # Counters for grid statistics
    total_grids = len(grid)
    green_grids = sum(1 for count in data_counts if count >= limit)

    print(f"Total grids: {total_grids}")
    print(f"Green grids (>= {limit} points): {green_grids}")

    # Create the plot with proportional aspect ratio
    fig, ax = plt.subplots(figsize=(10, 10 * height_km / width_km))

    for i, cell in enumerate(grid):
        lon_start, lon_end, lat_start, lat_end = cell
        count = data_counts[i]

        # Define grid cell color based on data count
        if count == 0:
            color = 'white'
        elif count < limit:
            color = 'grey'
        else:
            color = 'green'

        # Draw the grid cell without overlapping edges
        rect = plt.Rectangle((lon_start, lat_start), lon_end - lon_start, lat_end - lat_start,
                             facecolor=color, edgecolor='black', lw=0.5)
        ax.add_patch(rect)

    # Set axis limits and labels
    ax.set_xlim([lon_min, lon_max])
    ax.set_ylim([lat_min, lat_max])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)
    print('longitude:', lon_min, lon_max)
    print('latitude:', lat_min, lat_max)

    plt.show()




def visualise_updated_grid(grid_shape, grid_info, data_counts, title="Updated Grid Visualization with Clustering Results"):
    '''
    Visualise the updated grid with clustering results integrated.

    Parameters:
        grid (list of tuples): List of grid cells as (lon_start, lon_end, lat_start, lat_end).
        data_counts (ndarray): Updated values for each grid cell (should be 2D).
        title (str): Title of the plot.
    '''
    geod = Geod(ellps="WGS84")

    # Calculate the geographic extent in kilometers
    lat_min = min([cell[2] for cell in grid_info])
    lat_max = max([cell[3] for cell in grid_info])
    lon_min = min([cell[0] for cell in grid_info])
    lon_max = max([cell[1] for cell in grid_info])

    width_km = geod.line_length([lon_min, lon_max], [lat_min, lat_min]) / 1000
    height_km = geod.line_length([lon_min, lon_min], [lat_min, lat_max]) / 1000

    # Create the plot with proportional aspect ratio
    fig, ax = plt.subplots(figsize=(10, 10 * height_km / width_km))

    rows, cols = grid_shape.shape[0], grid_shape.shape[1]
    data_counts = np.array((data_counts))
    if data_counts.ndim == 1:
        try:
            data_counts = data_counts.reshape((rows,cols))
        except ValueError:
            print(f"❌ Error: Cannot reshape `data_counts` {data_counts.shape} to `grid_shape` {grid_shape}")
            return

    #assign rows and columns with the same shape as grid 
    


    for i, cell in enumerate(grid_info):
        lon_start, lon_end, lat_start, lat_end = cell

        # **Fix: Properly map 2D data_counts to 1D grid**
        row, col = divmod(i, grid_shape.shape[1])  # Convert linear index to 2D
        value = data_counts[row, col]  # Extract the scalar value

        # **Correct assignment priorities**
        if np.isnan(value) or value == -1:  # No Data
            color = 'white'
        elif value == 0:  # Not Enough Data (<3000 points)
            color = 'white'
        elif value == 111:  # Cluster 1
            color = '#416E6F'
        elif value == 222:  # Cluster 2
            color = '#EC9F72'
        elif value == 333:  # Cluster 3
            color = '#8D91C0'
        elif value == 444:  # Cluster 4
            color = '#DED88B'
        else:  
            color = 'white'  # Default for unexpected values

        # Draw the grid cell
        rect = plt.Rectangle((lon_start, lat_start), lon_end - lon_start, lat_end - lat_start,
                             facecolor=color, edgecolor='black', lw=0.5)
        ax.add_patch(rect)

    # Set axis limits and labels
    ax.set_xlim([lon_min, lon_max])
    ax.set_ylim([lat_min, lat_max])
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylabel('Latitude')
    ax.set_title(title)

    # Add legend
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', label='Invalid Grid'),
        #plt.Rectangle((0, 0), 1, 1, facecolor='grey', edgecolor='black', label='Not Enough Data (<3000 points)'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#416E6F', edgecolor='black', label='Cluster 1'), #greenblue
        plt.Rectangle((0, 0), 1, 1, facecolor='#EC9F72', edgecolor='black', label='Cluster 2'), #orange
        # plt.Rectangle((0, 0), 1, 1, facecolor='#8D91C0', edgecolor='black', label='Cluster 3'),
        # plt.Rectangle((0, 0), 1, 1, facecolor='#DED88B', edgecolor='black', label='Cluster 4')  
    ]
    ax.legend(
        handles=legend_patches,
        loc='center left',  
        bbox_to_anchor=(1.05, 0.5),  
        frameon=True  
    )

    plt.show()


def modify_cmo_topo():
    """Modify the cmocean 'topo' colormap: replace 0-100m (originally blue) with olive green."""
    topo_cmap = cmocean.cm.topo  # Get the cmocean topo colormap
    new_colors = topo_cmap(np.linspace(0.5, 1, 256))  # Remove lowest blue shades (0.25 removes blues)

    # Manually replace first ~50 colors (low elevation) with olive green
    #new_colors[:50] = mcolors.to_rgba("olive")

    # Create a new colormap
    return mcolors.LinearSegmentedColormap.from_list("custom_cmo_topo", new_colors)

# Create the modified colormap
custom_cmo_topo = modify_cmo_topo()






def plot_cluster_histograms(
    H, labels, bin_edges, centroids=None,
    n_examples=12, example_alpha=0.15, title_prefix="",
    legend_loc="upper right", legend_bbox=None,
    random_state=42, custom_colors=None
):
    """
    Plot all clusters in the same figure for direct comparison.
    Band, mean, centroid, and example histograms from the same cluster
    are plotted with the same color family.
    """

    x = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    clusters = np.sort(np.unique(labels))

    fig, ax = plt.subplots(figsize=(9, 6))

    if custom_colors is None:
        custom_colors = [
            "#416E6F",  # Green
            "#EC9F72",  # Orange
            "#8D91C0",  # Purple
            "#DED88B",  # Yellow
            "#A26C66",  # Red-Brown
        ]

    cluster_colors = {
        c: custom_colors[i % len(custom_colors)]
        for i, c in enumerate(clusters)
    }

    rng = np.random.default_rng(random_state)

    for c in clusters:
        Hc = H[labels == c]

        if Hc.size == 0:
            continue

        color = cluster_colors[c]

        mean_h = np.nanmean(Hc, axis=0)
        p10_h  = np.nanpercentile(Hc, 10, axis=0)
        p90_h  = np.nanpercentile(Hc, 90, axis=0)

        # 10–90% band: same cluster color, transparent
        ax.fill_between(
            x, p10_h, p90_h,
            step="mid",
            color=color,
            alpha=0.18,
            label=f"Cluster {c} 10–90% band"
        )

        # Example member histograms: same color, very transparent
        n_show = min(n_examples, Hc.shape[0])
        if n_show > 0:
            idx = rng.choice(Hc.shape[0], size=n_show, replace=False)
            ax.plot(
                x,
                Hc[idx].T,
                color=color,
                alpha=example_alpha,
                lw=0.8
            )

        # Mean curve: same color, solid and thicker
        ax.plot(
            x,
            mean_h,
            color=color,
            lw=2.5,
            label=f"Cluster {c} mean (n={Hc.shape[0]})"
        )



    ax.set_title(f"{title_prefix} Cluster comparison")
    ax.set_xlabel("Subsidence (m)")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend(loc=legend_loc, bbox_to_anchor=legend_bbox)
    fig.tight_layout()
    plt.show()


    
def plot_cluster_cdfs(H, labels, bin_edges, centroids=None, title_prefix=""):
    """
    Same idea, but for CDFs (often easier to see location/shift differences).

    Assumes H are *densities* (from np.histogram(..., density=True)).
    """
    x = bin_edges[1:]  # CDF step endpoints
    widths = np.diff(bin_edges)

    clusters = np.unique(labels)
    fig, axes = plt.subplots(1, len(clusters), figsize=(7 * len(clusters), 4), sharey=True)

    if len(clusters) == 1:
        axes = [axes]

    for ax, c in zip(axes, sorted(clusters)):
        Hc = H[labels == c]
        if Hc.size == 0:
            ax.set_title(f"{title_prefix}Cluster {c} (n=0)")
            ax.axis("off")
            continue

        # Convert densities -> per-bin probabilities, then CDFs
        Pc = Hc * widths  # per-bin probability mass
        Pc = Pc / np.maximum(Pc.sum(axis=1, keepdims=True), 1e-12)
        Cc = np.cumsum(Pc, axis=1)

        mean_cdf = np.nanmean(Cc, axis=0)
        p10_cdf  = np.nanpercentile(Cc, 10, axis=0)
        p90_cdf  = np.nanpercentile(Cc, 90, axis=0)

        ax.fill_between(x, p10_cdf, p90_cdf, step="post", alpha=0.2, label="10–90% band (CDF)")
        ax.plot(x, mean_cdf, lw=2, label="Mean CDF")

        if centroids is not None:
            Pc0 = centroids[c] * widths
            Pc0 = Pc0 / max(Pc0.sum(), 1e-12)
            C0  = np.cumsum(Pc0)
            ax.plot(x, C0, lw=2, ls="--", label="Centroid CDF")

        ax.set_title(f"{title_prefix}Cluster {c} (n={Hc.shape[0]})")
        ax.set_xlabel("Subsidence (m)")
        ax.set_ylabel("CDF")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    handles, labels_ = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_, loc="lower right")
    fig.tight_layout()
    plt.show()

    import numpy as np
    

def remap_cluster_labels(assignments, mapping):
    """
    Remap cluster labels manually.

    Parameters
    ----------
    assignments : array-like
        Original cluster labels, e.g. [0, 1, 1, 0, ...]
    mapping : dict
        Dictionary defining the new labels.
        Example: {0: 1, 1: 0} swaps cluster 0 and 1.

    Returns
    -------
    new_assignments : ndarray
        Relabelled assignments.
    """
    assignments = np.asarray(assignments)

    new_assignments = np.empty_like(assignments)

    for old_label, new_label in mapping.items():
        new_assignments[assignments == old_label] = new_label

    return new_assignments





# ==============================
#   5️⃣  COMPARISON FUNCTIONS
# ==============================





class EuclideanKMeansClustering:
    """
    Standard k-means clustering for histogram vectors.

    Assignment:
        each histogram is assigned to the nearest centroid using Euclidean distance.

    Update:
        each centroid is updated as the arithmetic mean of all histograms
        assigned to that cluster.

    This treats each histogram as a vector in n_bins-dimensional Euclidean space.
    """

    def __init__(
        self,
        n_clusters,
        max_iter=300,
        tol=1e-4,
        random_state=42,
        n_init=10,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init

        self.model = None
        self.assignments = None
        self.centroids = None
        self.n_iter_ = None
        self.inertia_ = None

    def fit(self, histograms):
        histograms = np.asarray(histograms, dtype=float)

        if histograms.ndim != 2:
            raise ValueError("histograms must have shape (n_samples, n_bins).")

        self.model = KMeans(
            n_clusters=self.n_clusters,
            init="k-means++",
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )

        self.model.fit(histograms)

        self.assignments = self.model.labels_
        self.centroids = self.model.cluster_centers_
        self.n_iter_ = self.model.n_iter_
        self.inertia_ = self.model.inertia_

        return self
    







class DeformationWassersteinClustering:
    """
    Clustering of deformation histograms using Wasserstein geometry.

    Both assignment and centroid update are based on the physical deformation
    histogram, not on the frequency-profile shape.

    Assignment:
        Each grid-cell histogram is assigned to the nearest centroid using
        Wasserstein distance over deformation-bin locations.

    Update:
        Each centroid is updated as a 1D Wasserstein barycenter of all member
        histograms in that cluster.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.

    bin_edges : array-like, shape (n_bins + 1,)
        Physical deformation bin edges, e.g. np.linspace(-0.08, 0.08, 51).

    max_iter : int
        Maximum number of clustering iterations.

    tol : float
        Convergence tolerance in physical deformation units.
        If bin_edges are in metres, tol=1e-4 means 0.1 mm.

    random_state : int
        Random seed for initialization.

    n_quantiles : int
        Number of quantile points used to approximate Wasserstein distance
        and Wasserstein barycenters.

    input_is_density : bool
        If True, input histograms are density histograms and will be converted
        to probability masses using bin widths.
        If False, input histograms are already probability masses.

    p : int
        Wasserstein order. Use p=2 for W2 distance and W2 barycenter.
        This is recommended if you want assignment and barycenter update to be
        mathematically consistent.
    """

    def __init__(
        self,
        n_clusters,
        bin_edges,
        max_iter=100,
        tol=1e-4,
        random_state=42,
        n_quantiles=1000,
        input_is_density=True,
        p=2,
    ):
        self.n_clusters = n_clusters
        self.bin_edges = np.asarray(bin_edges, dtype=float)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        self.bin_widths = np.diff(self.bin_edges)

        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_quantiles = n_quantiles
        self.input_is_density = input_is_density
        self.p = p

        self.centroids = None                 # density histograms
        self.centroid_masses_ = None          # probability-mass histograms
        self.centroid_quantiles_ = None       # quantile representation
        self.assignments = None
        self.n_iter_ = None
        self.centroid_shift_history_ = []

    def _normalize_masses(self, masses):
        masses = np.asarray(masses, dtype=float)
        masses = np.clip(masses, 0, None)

        if masses.ndim == 1:
            masses = masses[None, :]

        row_sums = masses.sum(axis=1, keepdims=True)

        if np.any(row_sums == 0):
            raise ValueError("At least one histogram has zero total mass.")

        return masses / row_sums

    def _histograms_to_masses(self, histograms):
        histograms = np.asarray(histograms, dtype=float)

        if histograms.ndim != 2:
            raise ValueError("histograms must have shape (n_samples, n_bins).")

        if histograms.shape[1] != len(self.bin_centers):
            raise ValueError(
                f"Expected {len(self.bin_centers)} bins, "
                f"got {histograms.shape[1]}."
            )

        if self.input_is_density:
            masses = histograms * self.bin_widths
        else:
            masses = histograms.copy()

        return self._normalize_masses(masses)

    def _masses_to_densities(self, masses):
        masses = np.asarray(masses, dtype=float)
        return masses / self.bin_widths

    def _mass_to_quantile(self, mass, q):
        """
        Convert one deformation histogram into a quantile function.

        This is the crucial step: the returned values are deformation values,
        not frequency values.
        """
        mass = np.asarray(mass, dtype=float)
        mass = np.clip(mass, 0, None)
        mass = mass / mass.sum()

        cdf = np.cumsum(mass)

        cdf_ext = np.concatenate([[0.0], cdf])
        x_ext = np.concatenate([[self.bin_centers[0]], self.bin_centers])

        return np.interp(q, cdf_ext, x_ext)

    def _masses_to_quantiles(self, masses):
        q = (np.arange(self.n_quantiles) + 0.5) / self.n_quantiles

        return np.array([
            self._mass_to_quantile(mass, q)
            for mass in masses
        ])

    def _wasserstein_distance_from_quantiles(self, q1, q2):
        """
        Wasserstein distance between two 1D distributions represented by
        quantile functions.
        """
        if self.p == 1:
            return np.mean(np.abs(q1 - q2))
        elif self.p == 2:
            return np.sqrt(np.mean((q1 - q2) ** 2))
        else:
            return np.mean(np.abs(q1 - q2) ** self.p) ** (1.0 / self.p)

    def _barycenter_quantile(self, cluster_quantiles, weights=None):
        """
        Wasserstein barycenter in 1D.

        For W2, the barycenter quantile function is the weighted mean of
        member quantile functions.
        """
        cluster_quantiles = np.asarray(cluster_quantiles, dtype=float)
        n_members = cluster_quantiles.shape[0]

        if weights is None:
            weights = np.ones(n_members) / n_members
        else:
            weights = np.asarray(weights, dtype=float)
            weights = weights / weights.sum()

        if self.p == 2:
            return np.average(cluster_quantiles, axis=0, weights=weights)

        elif self.p == 1:
            # For W1, a median quantile function is more appropriate.
            return np.median(cluster_quantiles, axis=0)

        else:
            # Practical fallback: weighted mean quantile.
            return np.average(cluster_quantiles, axis=0, weights=weights)

    def _quantile_to_mass(self, quantile_values):
        """
        Project a barycenter quantile function back onto the original deformation bins.
        """
        bary_mass, _ = np.histogram(
            quantile_values,
            bins=self.bin_edges,
            weights=np.ones_like(quantile_values) / len(quantile_values),
        )

        bary_mass = np.clip(bary_mass, 0, None)

        if bary_mass.sum() == 0:
            raise RuntimeError("Barycenter projection produced zero mass.")

        return bary_mass / bary_mass.sum()

    def fit(self, histograms):
        histograms = np.asarray(histograms, dtype=float)

        # Convert density histograms into probability masses over deformation bins
        masses = self._histograms_to_masses(histograms)

        # Convert each deformation histogram into quantile representation
        quantiles = self._masses_to_quantiles(masses)

        # Initialization using k-means++ on mass histograms.
        # This is only for initial centroids.
        kmeans_init = KMeans(
            n_clusters=self.n_clusters,
            init="k-means++",
            random_state=self.random_state,
            n_init=10,
        ).fit(masses)

        centroid_masses = self._normalize_masses(kmeans_init.cluster_centers_)
        centroid_quantiles = self._masses_to_quantiles(centroid_masses)

        self.centroid_shift_history_ = []

        for iteration in range(self.max_iter):

            # ============================================================
            # Assignment step:
            # Assign every deformation histogram to the nearest centroid
            # using Wasserstein distance in deformation-value space.
            # ============================================================
            assignments = np.array([
                np.argmin([
                    self._wasserstein_distance_from_quantiles(q_hist, q_centroid)
                    for q_centroid in centroid_quantiles
                ])
                for q_hist in quantiles
            ])

            # ============================================================
            # Update step:
            # Compute new centroid of each cluster as a Wasserstein barycenter
            # of the assigned deformation histograms.
            # ============================================================
            new_centroid_quantiles = []
            new_centroid_masses = []

            for i in range(self.n_clusters):
                cluster_quantiles = quantiles[assignments == i]

                if len(cluster_quantiles) > 0:
                    bary_q = self._barycenter_quantile(cluster_quantiles)
                    bary_mass = self._quantile_to_mass(bary_q)

                    new_centroid_quantiles.append(bary_q)
                    new_centroid_masses.append(bary_mass)

                else:
                    # If a cluster becomes empty, keep the old centroid.
                    new_centroid_quantiles.append(centroid_quantiles[i])
                    new_centroid_masses.append(centroid_masses[i])

            new_centroid_quantiles = np.array(new_centroid_quantiles)
            new_centroid_masses = np.array(new_centroid_masses)

            # ============================================================
            # Convergence check:
            # Measure the maximum Wasserstein movement of centroids.
            # ============================================================
            centroid_shift = max(
                self._wasserstein_distance_from_quantiles(old_q, new_q)
                for old_q, new_q in zip(centroid_quantiles, new_centroid_quantiles)
            )

            self.centroid_shift_history_.append(centroid_shift)

            centroid_quantiles = new_centroid_quantiles
            centroid_masses = new_centroid_masses

            if centroid_shift < self.tol:
                break

        # Recompute final assignments using final centroids
        self.assignments = np.array([
            np.argmin([
                self._wasserstein_distance_from_quantiles(q_hist, q_centroid)
                for q_centroid in centroid_quantiles
            ])
            for q_hist in quantiles
        ])

        self.n_iter_ = iteration + 1

        self.centroid_quantiles_ = centroid_quantiles
        self.centroid_masses_ = centroid_masses

        # Store as densities, so plotting matches your original histograms
        self.centroids = self._masses_to_densities(centroid_masses)

        return self
    



min_value = -0.05
max_value = 0.05
bin_edges = np.linspace(min_value, max_value, 51)
