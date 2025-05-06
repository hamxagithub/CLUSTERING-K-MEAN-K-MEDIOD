import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


# Load the dataset similar to Q3
def load_data(filename):
    try:
        # First try to read the Information sheet to understand variables
        print("Reading Information sheet to understand variable meanings...")
        try:
            info_df = pd.read_excel(filename, sheet_name="Information")
            print("\nVariable Information from Excel:")
            print(info_df)

            # Extract variable descriptions if available
            variable_info = {}
            for _, row in info_df.iterrows():
                if 'Variable' in info_df.columns and 'Description' in info_df.columns:
                    var_name = str(row['Variable']) if not pd.isna(row['Variable']) else ""
                    var_desc = str(row['Description']) if not pd.isna(row['Description']) else ""
                    if var_name:
                        variable_info[var_name] = var_desc

            if variable_info:
                print("\nVariable Descriptions:")
                for var, desc in variable_info.items():
                    print(f"- {var}: {desc}")
        except Exception as e:
            print(f"Could not read Information sheet: {e}")
            variable_info = {}

        # Read from the Normalized_Data sheet
        print("\nReading data from Normalized_Data sheet...")
        data = pd.read_excel(filename, sheet_name="Normalized_Data")

        # Get column names
        column_names = list(data.columns)
        print(f"All columns in dataset: {column_names}")

        # Use only the first three columns (ignore the last column labeled 'M')
        data = data.iloc[:, 0:3]
        used_columns = list(data.columns)

        # Shuffle the records
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"Using columns for clustering: {used_columns}")

        # Return as numpy array for clustering, plus column info
        return data.values, used_columns, variable_info
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None


# Calculate SSE/WCSS for a clustering result
def calculate_sse(X, labels, centers):
    n_clusters = len(np.unique(labels))
    sse_per_cluster = np.zeros(n_clusters)

    for i in range(n_clusters):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            center = centers[i]
            squared_distances = np.sum((cluster_points - center) ** 2, axis=1)
            sse_per_cluster[i] = np.sum(squared_distances)

    total_sse = np.sum(sse_per_cluster)
    return total_sse, sse_per_cluster


# Run K-means clustering using scikit-learn
def run_kmeans_sklearn(data, k, random_state=42):
    print(f"Running scikit-learn KMeans with k={k}...")
    start_time = time.time()

    # Initialize and fit KMeans - use same random state as in Q3
    kmeans = KMeans(
        n_clusters=k,
        init='random',  # Use random initialization like in Q3
        n_init=1,  # Single initialization to match Q3
        max_iter=100,  # Match max iterations in Q3
        tol=0.0000001,  # Set very small tolerance to match convergence in Q3
        random_state=random_state
    )

    labels = kmeans.fit_predict(data)
    iterations = kmeans.n_iter_
    centers = kmeans.cluster_centers_

    execution_time = time.time() - start_time

    # Calculate SSE/WCSS
    total_sse, cluster_sse = calculate_sse(data, labels, centers)

    # Count cluster sizes
    cluster_sizes = [np.sum(labels == i) for i in range(k)]

    return {
        'labels': labels,
        'centers': centers,
        'iterations': iterations,
        'time': execution_time,
        'total_sse': total_sse,
        'cluster_sse': cluster_sse,
        'cluster_sizes': cluster_sizes
    }


# Run K-medoids clustering using a custom implementation with scikit-learn components
def run_kmedoids_sklearn(data, k, random_state=42):
    print(f"Running scikit-learn implementation of K-medoids with k={k}...")
    start_time = time.time()

    # Initialize medoids randomly
    np.random.seed(random_state)
    medoid_indices = np.random.choice(len(data), k, replace=False)
    medoids = data[medoid_indices]

    iterations = 0
    old_medoid_indices = np.zeros(k)

    # Calculate pairwise distances once
    distances = pairwise_distances(data, metric='euclidean')

    for iteration in range(100):  # Max 100 iterations
        iterations += 1

        # Assign points to closest medoids
        labels = np.argmin([distances[:, idx] for idx in medoid_indices], axis=0)

        # Update medoids
        old_medoid_indices = np.copy(medoid_indices)

        for i in range(k):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) > 0:
                # Find the point in cluster that minimizes sum of distances to other points in cluster
                cluster_distances = distances[np.ix_(cluster_indices, cluster_indices)]
                within_cluster_sums = np.sum(cluster_distances, axis=1)
                min_idx = np.argmin(within_cluster_sums)
                medoid_indices[i] = cluster_indices[min_idx]

        # Check for convergence
        if np.all(old_medoid_indices == medoid_indices):
            break

    # Get the actual medoids based on the indices
    medoids = data[medoid_indices]

    # Calculate final SSE/WCSS
    total_sse, cluster_sse = calculate_sse(data, labels, medoids)

    # Count cluster sizes
    cluster_sizes = [np.sum(labels == i) for i in range(k)]

    execution_time = time.time() - start_time

    return {
        'labels': labels,
        'centers': medoids,
        'medoid_indices': medoid_indices,
        'iterations': iterations,
        'time': execution_time,
        'total_sse': total_sse,
        'cluster_sse': cluster_sse,
        'cluster_sizes': cluster_sizes
    }


# Create visualization for sklearn clustering results
def visualize_sklearn_clusters(data, kmeans_results, kmedoids_results, used_columns, variable_info):
    # Create a dark style plot with modern aesthetics
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 10))

    # Custom color palette - neon colors for dark background
    neon_colors = ['#00FFFF', '#FF00FF', '#00FF00', '#FFFF00', '#FF9933', '#33CCFF', '#FF3399', '#99FF33']

    # Create a 2x2 grid for different views of the data
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Set up names for axis labels based on Information sheet
    axis_labels = []
    for col in used_columns:
        if variable_info and col in variable_info:
            axis_labels.append(f"{col}: {variable_info[col]}")
        else:
            axis_labels.append(col)

    # K-means visualization - standard view
    ax1 = fig.add_subplot(gs[0, 0])

    # Add a subtle gradient background
    from matplotlib.colors import LinearSegmentedColormap
    background_gradient = LinearSegmentedColormap.from_list('bg_gradient', ['#0D1117', '#1A2233'])
    background = np.outer(np.ones(100), np.linspace(0, 1, 100))
    ax1.imshow(background, cmap=background_gradient, aspect='auto',
               extent=[min(data[:, 0]) - 0.5, max(data[:, 0]) + 0.5,
                       min(data[:, 1]) - 0.5, max(data[:, 1]) + 0.5],
               alpha=0.5, interpolation='bicubic')

    # Add glowing effect for points
    for i in range(len(np.unique(kmeans_results['labels']))):
        cluster_points = data[kmeans_results['labels'] == i]
        if len(cluster_points) > 0:
            # Create glow effect
            for size, alpha in zip([100, 80, 60], [0.1, 0.2, 0.3]):
                ax1.scatter(cluster_points[:, 0], cluster_points[:, 1],
                            s=size, color=neon_colors[i % len(neon_colors)], alpha=alpha, edgecolors='none')

            # Main point
            ax1.scatter(cluster_points[:, 0], cluster_points[:, 1],
                        s=40, color=neon_colors[i % len(neon_colors)], alpha=0.9,
                        edgecolors='white', linewidths=0.5,
                        label=f'Cluster {i + 1} (n={np.sum(kmeans_results["labels"] == i)})')

    # Add starburst effect for centroids
    for centroid in kmeans_results['centers']:
        # Starburst rays
        for angle in range(0, 360, 45):
            angle_rad = np.deg2rad(angle)
            dx = 0.07 * np.cos(angle_rad)
            dy = 0.07 * np.sin(angle_rad)
            ax1.plot([centroid[0], centroid[0] + dx], [centroid[1], centroid[1] + dy],
                     color='white', alpha=0.7, linewidth=1.5)

        ax1.scatter(centroid[0], centroid[1], s=200, color='white', alpha=0.9,
                    marker='*', edgecolors='#00FFFF', linewidths=2)

    ax1.set_title('K-means Clustering', fontsize=16, fontweight='bold', color='white')
    ax1.set_xlabel(axis_labels[0] if len(axis_labels) > 0 else 'Feature 1', fontsize=12, color='white')
    ax1.set_ylabel(axis_labels[1] if len(axis_labels) > 1 else 'Feature 2', fontsize=12, color='white')
    ax1.grid(True, alpha=0.15, linestyle='--')
    ax1.legend(fontsize=10, framealpha=0.3)

    # K-medoids visualization - standard view
    ax2 = fig.add_subplot(gs[0, 1])

    # Add the same gradient background
    ax2.imshow(background, cmap=background_gradient, aspect='auto',
               extent=[min(data[:, 0]) - 0.5, max(data[:, 0]) + 0.5,
                       min(data[:, 1]) - 0.5, max(data[:, 1]) + 0.5],
               alpha=0.5, interpolation='bicubic')

    # Add glowing effect for points
    for i in range(len(np.unique(kmedoids_results['labels']))):
        cluster_points = data[kmedoids_results['labels'] == i]
        if len(cluster_points) > 0:
            # Create glow effect
            for size, alpha in zip([100, 80, 60], [0.1, 0.2, 0.3]):
                ax2.scatter(cluster_points[:, 0], cluster_points[:, 1],
                            s=size, color=neon_colors[i % len(neon_colors)], alpha=alpha, edgecolors='none')

            # Main point
            ax2.scatter(cluster_points[:, 0], cluster_points[:, 1],
                        s=40, color=neon_colors[i % len(neon_colors)], alpha=0.9,
                        edgecolors='white', linewidths=0.5,
                        label=f'Cluster {i + 1} (n={np.sum(kmedoids_results["labels"] == i)})')

    # Add diamond effect for medoids
    for medoid in kmedoids_results['centers']:
        # Diamond outline
        diamond_x = [medoid[0], medoid[0] + 0.05, medoid[0], medoid[0] - 0.05, medoid[0]]
        diamond_y = [medoid[1] + 0.05, medoid[1], medoid[1] - 0.05, medoid[1], medoid[1] + 0.05]
        ax2.plot(diamond_x, diamond_y, color='white', linewidth=2, alpha=0.9)

        ax2.scatter(medoid[0], medoid[1], s=180, color='white', alpha=0.9,
                    marker='D', edgecolors='#FF00FF', linewidths=2)

    ax2.set_title('K-medoids Clustering', fontsize=16, fontweight='bold', color='white')
    ax2.set_xlabel(axis_labels[0] if len(axis_labels) > 0 else 'Feature 1', fontsize=12, color='white')
    ax2.set_ylabel(axis_labels[1] if len(axis_labels) > 1 else 'Feature 2', fontsize=12, color='white')
    ax2.grid(True, alpha=0.15, linestyle='--')
    ax2.legend(fontsize=10, framealpha=0.3)

    # 3D Visualization for K-means
    ax3 = fig.add_subplot(gs[1, 0], projection='3d')
    ax3.set_facecolor('#0D1117')

    # Use first two features and add distance to centroid as 3rd dimension
    for i in range(len(np.unique(kmeans_results['labels']))):
        cluster_points = data[kmeans_results['labels'] == i]
        if len(cluster_points) > 0:
            # Calculate distances to respective centroid for z-axis
            distances = np.sqrt(np.sum((cluster_points[:, :2] - kmeans_results['centers'][i, :2]) ** 2, axis=1))

            # Plot 3D scatter
            ax3.scatter(cluster_points[:, 0], cluster_points[:, 1], distances,
                        s=40, color=neon_colors[i % len(neon_colors)], alpha=0.7,
                        edgecolors='white', linewidths=0.5)

            # Plot vertical lines connecting points to their projection on xy-plane
            for j in range(len(distances)):
                ax3.plot([cluster_points[j, 0], cluster_points[j, 0]],
                         [cluster_points[j, 1], cluster_points[j, 1]],
                         [0, distances[j]], color=neon_colors[i % len(neon_colors)], alpha=0.1)

    # Plot centroids
    ax3.scatter(kmeans_results['centers'][:, 0], kmeans_results['centers'][:, 1],
                [0] * len(kmeans_results['centers']), s=200, color='white', alpha=0.9,
                marker='*', edgecolors='#00FFFF', linewidths=2)

    ax3.set_title('K-means 3D View (z=distance to centroid)', fontsize=14, fontweight='bold', color='white')
    ax3.set_xlabel(axis_labels[0] if len(axis_labels) > 0 else 'Feature 1', fontsize=10, color='white')
    ax3.set_ylabel(axis_labels[1] if len(axis_labels) > 1 else 'Feature 2', fontsize=10, color='white')
    ax3.set_zlabel('Distance to Centroid', fontsize=10, color='white')
    ax3.grid(True, alpha=0.15, linestyle='--')

    # 3D Visualization for K-medoids
    ax4 = fig.add_subplot(gs[1, 1], projection='3d')
    ax4.set_facecolor('#0D1117')

    # Use first two features and add distance to medoid as 3rd dimension
    for i in range(len(np.unique(kmedoids_results['labels']))):
        cluster_points = data[kmedoids_results['labels'] == i]
        if len(cluster_points) > 0:
            # Calculate distances to respective medoid for z-axis
            distances = np.sqrt(np.sum((cluster_points[:, :2] - kmedoids_results['centers'][i, :2]) ** 2, axis=1))

            # Plot 3D scatter
            ax4.scatter(cluster_points[:, 0], cluster_points[:, 1], distances,
                        s=40, color=neon_colors[i % len(neon_colors)], alpha=0.7,
                        edgecolors='white', linewidths=0.5)

            # Plot vertical lines connecting points to their projection on xy-plane
            for j in range(len(distances)):
                ax4.plot([cluster_points[j, 0], cluster_points[j, 0]],
                         [cluster_points[j, 1], cluster_points[j, 1]],
                         [0, distances[j]], color=neon_colors[i % len(neon_colors)], alpha=0.1)

    # Plot medoids
    ax4.scatter(kmedoids_results['centers'][:, 0], kmedoids_results['centers'][:, 1],
                [0] * len(kmedoids_results['centers']), s=180, color='white', alpha=0.9,
                marker='D', edgecolors='#FF00FF', linewidths=2)

    ax4.set_title('K-medoids 3D View (z=distance to medoid)', fontsize=14, fontweight='bold', color='white')
    ax4.set_xlabel(axis_labels[0] if len(axis_labels) > 0 else 'Feature 1', fontsize=10, color='white')
    ax4.set_ylabel(axis_labels[1] if len(axis_labels) > 1 else 'Feature 2', fontsize=10, color='white')
    ax4.set_zlabel('Distance to Medoid', fontsize=10, color='white')
    ax4.grid(True, alpha=0.15, linestyle='--')

    # Add a title for the overall figure
    plt.suptitle('Advanced Visualization of Clustering Results', fontsize=22, fontweight='bold', color='white', y=0.98)

    # Add a subtle watermark
    fig.text(0.98, 0.02, 'Created with scikit-learn', fontsize=8, color='gray',
             ha='right', va='bottom', alpha=0.5, style='italic')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('advanced_clustering_visualization.png', dpi=300, bbox_inches='tight', facecolor='#0D1117')
    plt.show()


# Create visualization for time complexity comparison
def visualize_time_complexity(q3_results, q4_results):
    plt.figure(figsize=(14, 8))

    # Data preparation with readable names
    methods = ['Custom\nK-means', 'scikit-learn\nK-means', 'Custom\nK-medoids', 'scikit-learn\nK-medoids']
    times = [
        q3_results['kmeans_time'],
        q4_results['kmeans']['time'],
        q3_results['kmedoids_time'],
        q4_results['kmedoids']['time']
    ]

    # Using a vibrant color palette
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

    # Create bar chart with enhanced styling
    bars = plt.bar(methods, times, color=colors, width=0.6,
                   edgecolor='black', linewidth=1.5, alpha=0.85)

    # Add time values on top of bars
    for bar, time_value in zip(bars, times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.002,
                 f'{time_value:.4f}s',
                 ha='center', va='bottom', fontsize=12, fontweight='bold', color='#333333')

    # Add percentage labels inside bars for faster readability
    for i, v in enumerate(times):
        if v > max(times) * 0.05:  # Only show label if bar is tall enough
            plt.text(i, v / 2, f'{v:.4f}s',
                     ha='center', va='center', color='white',
                     fontweight='bold', fontsize=11)

    # Calculate and show speed differences with annotations
    kmeans_speedup = q3_results['kmeans_time'] / q4_results['kmeans']['time'] if q4_results['kmeans']['time'] > 0 else 0
    kmedoids_speedup = q3_results['kmedoids_time'] / q4_results['kmedoids']['time'] if q4_results['kmedoids'][
                                                                                           'time'] > 0 else 0

    # Show speed comparison annotations
    if kmeans_speedup != 1:
        faster = "scikit-learn" if kmeans_speedup > 1 else "Custom"
        factor = kmeans_speedup if kmeans_speedup > 1 else 1 / kmeans_speedup
        plt.annotate(f"{faster} is {factor:.2f}x faster",
                     xy=(0.5, max(times[0], times[1]) * 1.1),
                     xytext=(0.5, max(times) * 0.7),
                     arrowprops=dict(facecolor='#333333', shrink=0.05, width=2),
                     ha='center', va='center', fontsize=12, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor='#f8f9fa', alpha=0.8))

    if kmedoids_speedup != 1:
        faster = "scikit-learn" if kmedoids_speedup > 1 else "Custom"
        factor = kmedoids_speedup if kmedoids_speedup > 1 else 1 / kmedoids_speedup
        plt.annotate(f"{faster} is {factor:.2f}x faster",
                     xy=(2.5, max(times[2], times[3]) * 1.1),
                     xytext=(2.5, max(times) * 0.6),
                     arrowprops=dict(facecolor='#333333', shrink=0.05, width=2),
                     ha='center', va='center', fontsize=12, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor='#f8f9fa', alpha=0.8))

    # Title and labels with enhanced styling
    plt.title('Execution Time Comparison: Custom vs scikit-learn Implementations',
              fontsize=18, fontweight='bold', pad=20)
    plt.ylabel('Execution Time (seconds)', fontsize=14, fontweight='bold')
    plt.ylim(0, max(times) * 1.3)  # Add space for annotations

    # Add grid and styling
    plt.grid(True, axis='y', alpha=0.3, linestyle='--')
    plt.gca().set_facecolor('#f5f5f5')

    # Add border to the plot
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#333333')

    # Add a footer note
    plt.figtext(0.5, 0.01,
                "Note: Lower execution time indicates better performance",
                ha="center", fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig('sklearn_vs_custom_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


# Create SSE comparison visualization
def visualize_sse_comparison(q3_results, q4_results):
    # Use dark theme for modern look
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111)
    ax.set_facecolor('#0D1117')
    fig.patch.set_facecolor('#0D1117')

    # Custom color scheme for the hexagonal markers
    neon_colors = ['#00FFFF', '#33FF33', '#FF00FF', '#FFFF00']

    # Data preparation with descriptive labels
    methods = ['K-means\nCustom', 'K-means\nscikit-learn', 'K-medoids\nCustom', 'K-medoids\nscikit-learn']
    sses = [
        q3_results['kmeans_sse'],
        q4_results['kmeans']['total_sse'],
        q3_results['kmedoids_sse'],
        q4_results['kmedoids']['total_sse']
    ]

    # Create hexagonal markers instead of bars for futuristic look
    for i, (sse_val, color) in enumerate(zip(sses, neon_colors)):
        # Create hexagon shape
        from matplotlib.path import Path
        import matplotlib.patches as patches

        # Size proportional to SSE value
        size_factor = 0.3 * sse_val / max(sses)

        # Create hexagon vertices
        theta = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 points for hexagon
        x = i + size_factor * np.sin(theta)
        y = size_factor * np.cos(theta)

        # Create hexagon path
        verts = list(zip(x, y))
        codes = [Path.MOVETO] + [Path.LINETO] * 4 + [Path.CLOSEPOLY]
        path = Path(verts, codes)

        # Create patch
        patch = patches.PathPatch(path, facecolor=color, alpha=0.7,
                                  edgecolor='white', linewidth=2)
        ax.add_patch(patch)

        # Add glow effects
        for scale, alpha in zip([1.2, 1.1], [0.1, 0.2]):
            size_factor_glow = size_factor * scale
            x_glow = i + size_factor_glow * np.sin(theta)
            y_glow = size_factor_glow * np.cos(theta)
            verts_glow = list(zip(x_glow, y_glow))
            path_glow = Path(verts_glow, codes)
            glow = patches.PathPatch(path_glow, facecolor=color, alpha=alpha,
                                     edgecolor='none')
            ax.add_patch(glow)

        # Add central point
        ax.scatter(i, 0, color='white', s=30, zorder=10, edgecolor=color, linewidth=1.5)

        # Add SSE value inside hexagon
        ax.text(i, 0, f'{sse_val:.2f}', color='white', ha='center', va='center',
                fontsize=12, fontweight='bold', zorder=15)

        # Add implementation label
        ax.text(i, -size_factor - 0.1, methods[i], color='white', ha='center', va='top',
                fontsize=14, fontweight='bold')

    # Connect related implementations with energy beams
    beam_y_kmeans = max([q3_results['kmeans_sse'], q4_results['kmeans']['total_sse']]) * 0.4
    # K-means connection
    ax.plot([0, 1], [beam_y_kmeans, beam_y_kmeans], '--', color='#00FFFF', alpha=0.7, linewidth=2)

    beam_y_kmedoids = max([q3_results['kmedoids_sse'], q4_results['kmedoids']['total_sse']]) * 0.4
    # K-medoids connection
    ax.plot([2, 3], [beam_y_kmedoids, beam_y_kmedoids], '--', color='#FF00FF', alpha=0.7, linewidth=2)

    # Add percentage differences for pairs with futuristic callouts
    if q3_results['kmeans_sse'] > 0:
        kmeans_diff_pct = abs(q3_results['kmeans_sse'] - q4_results['kmeans']['total_sse']) / q3_results[
            'kmeans_sse'] * 100

        # Determine which is better
        better_impl = "scikit-learn" if q4_results['kmeans']['total_sse'] < q3_results['kmeans_sse'] else "Custom"

        # Add percentage difference with futuristic callout
        diff_text = f"{kmeans_diff_pct:.1f}% difference\n{better_impl} is better"
        ax.text(0.5, beam_y_kmeans * 1.3, diff_text, ha='center', va='center', color='#00FFFF',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.7,
                          edgecolor='#00FFFF', linewidth=1.5))

    if q3_results['kmedoids_sse'] > 0:
        kmedoids_diff_pct = abs(q3_results['kmedoids_sse'] - q4_results['kmedoids']['total_sse']) / q3_results[
            'kmedoids_sse'] * 100

        # Determine which is better
        better_impl = "scikit-learn" if q4_results['kmedoids']['total_sse'] < q3_results['kmedoids_sse'] else "Custom"

        # Add percentage difference with futuristic callout
        diff_text = f"{kmedoids_diff_pct:.1f}% difference\n{better_impl} is better"
        ax.text(2.5, beam_y_kmedoids * 1.3, diff_text, ha='center', va='center', color='#FF00FF',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.7,
                          edgecolor='#FF00FF', linewidth=1.5))

    # Add grid with futuristic styling
    for y in np.linspace(0, max(sses) * 1.2, 6):
        ax.axhline(y, color='white', alpha=0.1, linestyle='--', linewidth=0.5)

    # Add radial background
    theta = np.linspace(0, 2 * np.pi, 100)
    for radius in [0.5, 1, 1.5, 2]:
        circle_x = 1.5 + radius * np.cos(theta)
        circle_y = radius * np.sin(theta)
        ax.plot(circle_x, circle_y, 'white', alpha=0.05, linewidth=0.5)

    # Create glowing title effect
    title_text = 'Cluster Cohesion Analysis'
    subtitle_text = 'SSE/WCSS Comparison: Custom vs scikit-learn'

    # Main title with shadow effect
    for offset in [0.5, 0.3, 0.1]:
        ax.text(1.5, max(sses) * 1.0 + offset / 10, title_text,
                color='white', alpha=offset,
                fontsize=22, fontweight='bold', ha='center')
    ax.text(1.5, max(sses) * 1.0, title_text,
            color='white', fontsize=22, fontweight='bold', ha='center')

    # Subtitle
    ax.text(1.5, max(sses) * 0.9, subtitle_text,
            color='gray', fontsize=14, fontstyle='italic', ha='center')

    # Add explanatory note
    ax.text(1.5, -max(sses) * 0.3,
            "Note: Lower SSE/WCSS values indicate tighter, more cohesive clusters",
            color='gray', fontstyle='italic', fontsize=10, ha='center')

    # Add visual indicators of which implementation is better in each category
    # K-means comparison
    if q3_results['kmeans_sse'] < q4_results['kmeans']['total_sse']:
        ax.text(0, -max(sses) * 0.2, "✓", color='#33FF33', fontsize=24,
                ha='center', va='top', fontweight='bold')
    else:
        ax.text(1, -max(sses) * 0.2, "✓", color='#33FF33', fontsize=24,
                ha='center', va='top', fontweight='bold')

    # K-medoids comparison
    if q3_results['kmedoids_sse'] < q4_results['kmedoids']['total_sse']:
        ax.text(2, -max(sses) * 0.2, "✓", color='#33FF33', fontsize=24,
                ha='center', va='top', fontweight='bold')
    else:
        ax.text(3, -max(sses) * 0.2, "✓", color='#33FF33', fontsize=24,
                ha='center', va='top', fontweight='bold')

    # Clean up axes
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-max(sses) * 0.35, max(sses) * 1.2)
    ax.set_xticks([])  # Remove x-ticks since we have custom labels
    ax.set_yticks([])  # Remove y-ticks for cleaner look

    # Remove spines for cleaner look
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.text(0.98, 0.02, f"Generated: {timestamp}",
             fontsize=8, color='gray', ha='right', alpha=0.5)

    plt.tight_layout()
    plt.savefig('advanced_sse_comparison.png', dpi=300, bbox_inches='tight', facecolor='#0D1117')
    plt.show()


# Get results from Q3 (by loading the model or rerunning it)
def get_q3_results():
    import Q3

    # Load data
    data, used_columns, variable_info = Q3.load_data('Mine_Dataset.xlsx')

    if data is None:
        print("Failed to load data. Please check the file path.")
        return None, None, None, None

    # Find the optimal K as in Q3
    k_range = range(1, 11)
    sse_values = Q3.create_elbow_chart(data, k_range)

    sse_decrease = [sse_values[i - 1] - sse_values[i] for i in range(1, len(sse_values))]
    decreases = np.array(sse_decrease)
    normalized_decreases = decreases / decreases[0]

    optimal_k_idx = np.where(normalized_decreases < 0.15)[0]
    optimal_k = k_range[optimal_k_idx[0]] if len(optimal_k_idx) > 0 else 3

    # Run K-means with optimal K
    print(f"\nRunning custom K-means with K={optimal_k}...")
    start_time = time.time()
    kmeans_clusters, kmeans_centroids, kmeans_history = Q3.kmeans(data, optimal_k)
    kmeans_time = time.time() - start_time

    # Run K-medoids with optimal K
    print(f"\nRunning custom K-medoids with K={optimal_k}...")
    start_time = time.time()
    kmedoids_clusters, kmedoids_medoids, kmedoids_history = Q3.kmedoids(data, optimal_k)
    kmedoids_time = time.time() - start_time

    # Create labels for data points based on clusters
    kmeans_labels = np.zeros(len(data), dtype=int)
    for i, cluster in enumerate(kmeans_clusters):
        for point in cluster:
            # Find the index of the point in the data
            idx = np.where((data == point).all(axis=1))[0]
            if len(idx) > 0:
                kmeans_labels[idx[0]] = i

    kmedoids_labels = np.zeros(len(data), dtype=int)
    for i, cluster in enumerate(kmedoids_clusters):
        for point in cluster:
            # Find the index of the point in the data
            idx = np.where((data == point).all(axis=1))[0]
            if len(idx) > 0:
                kmedoids_labels[idx[0]] = i

    # Return results from Q3
    return {
        'optimal_k': optimal_k,
        'kmeans_time': kmeans_time,
        'kmeans_iterations': kmeans_history[-1]['iteration'],
        'kmeans_cluster_sizes': kmeans_history[-1]['cluster_sizes'],
        'kmeans_sse': kmeans_history[-1]['total_sse'],
        'kmeans_cluster_sse': kmeans_history[-1]['cluster_sse'],
        'kmeans_centers': kmeans_centroids,
        'kmeans_labels': kmeans_labels,
        'kmedoids_time': kmedoids_time,
        'kmedoids_iterations': kmedoids_history[-1]['iteration'],
        'kmedoids_cluster_sizes': kmedoids_history[-1]['cluster_sizes'],
        'kmedoids_sse': kmedoids_history[-1]['total_sse'],
        'kmedoids_cluster_sse': kmedoids_history[-1]['cluster_sse'],
        'kmedoids_centers': kmedoids_medoids,
        'kmedoids_labels': kmedoids_labels,
    }, data, used_columns, variable_info


def main():
    # Get results from Q3
    print("Retrieving results from Q3 (custom implementation)...")
    q3_results, data, used_columns, variable_info = get_q3_results()

    if q3_results is None:
        print("Failed to get Q3 results. Exiting.")
        return

    optimal_k = q3_results['optimal_k']
    print(f"\nUsing optimal K from Q3: K={optimal_k}")

    # Run scikit-learn implementations with same K
    kmeans_results = run_kmeans_sklearn(data, optimal_k)
    kmedoids_results = run_kmedoids_sklearn(data, optimal_k)

    # Store sklearn results
    q4_results = {
        'kmeans': kmeans_results,
        'kmedoids': kmedoids_results
    }

    # Display scikit-learn results with enhanced terminal output
    print("\n" + "╔" + "═" * 60 + "╗")
    print("║" + "SCIKIT-LEARN K-MEANS CLUSTERING RESULTS".center(60) + "║")
    print("╠" + "═" * 60 + "╣")
    print(f"║ Total iterations: {kmeans_results['iterations']:>42} ║")
    print(f"║ Cluster sizes: {str(kmeans_results['cluster_sizes']):>44} ║")

    # Format cluster SSE values
    sse_values = [f"{sse:.2f}" for sse in kmeans_results['cluster_sse']]
    print(f"║ Cluster SSE: {str(sse_values):>44} ║")
    print(f"║ Total SSE/WCSS: {kmeans_results['total_sse']:.2f}{' ' * 41}║")
    print(f"║ Time complexity: {kmeans_results['time']:.4f} seconds{' ' * 32}║")
    print("╚" + "═" * 60 + "╝")

    print("\n" + "╔" + "═" * 60 + "╗")
    print("║" + "SCIKIT-LEARN K-MEDOIDS CLUSTERING RESULTS".center(60) + "║")
    print("╠" + "═" * 60 + "╣")
    print(f"║ Total iterations: {kmedoids_results['iterations']:>42} ║")
    print(f"║ Cluster sizes: {str(kmedoids_results['cluster_sizes']):>44} ║")

    # Format cluster SSE values
    sse_values = [f"{sse:.2f}" for sse in kmedoids_results['cluster_sse']]
    print(f"║ Cluster SSE: {str(sse_values):>44} ║")
    print(f"║ Total SSE/WCSS: {kmedoids_results['total_sse']:.2f}{' ' * 41}║")
    print(f"║ Time complexity: {kmedoids_results['time']:.4f} seconds{' ' * 32}║")
    print("╚" + "═" * 60 + "╝")

    # Visualize only sklearn clustering results
    visualize_sklearn_clusters(data, kmeans_results, kmedoids_results, used_columns, variable_info)

    # Create comparisons for time complexity and SSE
    visualize_time_complexity(q3_results, q4_results)
    visualize_sse_comparison(q3_results, q4_results)

    # Comparative analysis with enhanced terminal styling
    print("\n" + "┏" + "━" * 70 + "┓")
    print("┃" + " PERFORMANCE COMPARISON: CUSTOM vs SCIKIT-LEARN ".center(70, "━") + "┃")
    print("┗" + "━" * 70 + "┛")

    # Time complexity comparison with colorized output
    print("\n┌─" + "─" * 68 + "─┐")
    print("│ " + "TIME COMPLEXITY COMPARISON".center(68) + " │")
    print("├─" + "─" * 68 + "─┤")

    kmeans_custom = f"{q3_results['kmeans_time']:.4f}s"
    kmeans_sklearn = f"{kmeans_results['time']:.4f}s"
    kmedoids_custom = f"{q3_results['kmedoids_time']:.4f}s"
    kmedoids_sklearn = f"{kmedoids_results['time']:.4f}s"

    print(
        f"│ K-means:   Custom={kmeans_custom}, scikit-learn={kmeans_sklearn}{' ' * (35 - len(kmeans_custom) - len(kmeans_sklearn))} │")
    print(
        f"│ K-medoids: Custom={kmedoids_custom}, scikit-learn={kmedoids_sklearn}{' ' * (35 - len(kmedoids_custom) - len(kmedoids_sklearn))} │")

    kmeans_speed_ratio = q3_results['kmeans_time'] / kmeans_results['time'] if kmeans_results['time'] > 0 else 0
    kmedoids_speed_ratio = q3_results['kmedoids_time'] / kmedoids_results['time'] if kmedoids_results['time'] > 0 else 0

    print("├─" + "─" * 68 + "─┤")
    if kmeans_speed_ratio > 1:
        print(
            f"│ ⚡ scikit-learn's K-means is {kmeans_speed_ratio:.2f}x faster than custom implementation{' ' * (16 - len(f'{kmeans_speed_ratio:.2f}'))} │")
    else:
        print(
            f"│ ⚡ Custom K-means is {1 / kmeans_speed_ratio:.2f}x faster than scikit-learn implementation{' ' * (18 - len(f'{1 / kmeans_speed_ratio:.2f}'))} │")

    if kmedoids_speed_ratio > 1:
        print(
            f"│ ⚡ scikit-learn's K-medoids is {kmedoids_speed_ratio:.2f}x faster than custom implementation{' ' * (14 - len(f'{kmedoids_speed_ratio:.2f}'))} │")
    else:
        print(
            f"│ ⚡ Custom K-medoids is {1 / kmedoids_speed_ratio:.2f}x faster than scikit-learn implementation{' ' * (16 - len(f'{1 / kmedoids_speed_ratio:.2f}'))} │")
    print("└─" + "─" * 68 + "─┘")

    # SSE comparison with stylized output
    print("\n┌─" + "─" * 68 + "─┐")
    print("│ " + "SSE/WCSS COMPARISON".center(68) + " │")
    print("├─" + "─" * 68 + "─┤")

    kmeans_custom_sse = f"{q3_results['kmeans_sse']:.2f}"
    kmeans_sklearn_sse = f"{kmeans_results['total_sse']:.2f}"
    kmedoids_custom_sse = f"{q3_results['kmedoids_sse']:.2f}"
    kmedoids_sklearn_sse = f"{kmedoids_results['total_sse']:.2f}"

    print(
        f"│ K-means:   Custom={kmeans_custom_sse}, scikit-learn={kmeans_sklearn_sse}{' ' * (35 - len(kmeans_custom_sse) - len(kmeans_sklearn_sse))} │")
    print(
        f"│ K-medoids: Custom={kmedoids_custom_sse}, scikit-learn={kmedoids_sklearn_sse}{' ' * (35 - len(kmedoids_custom_sse) - len(kmedoids_sklearn_sse))} │")

    kmeans_sse_diff_pct = abs(q3_results['kmeans_sse'] - kmeans_results['total_sse']) / q3_results[
        'kmeans_sse'] * 100 if q3_results['kmeans_sse'] > 0 else 0
    kmedoids_sse_diff_pct = abs(q3_results['kmedoids_sse'] - kmedoids_results['total_sse']) / q3_results[
        'kmedoids_sse'] * 100 if q3_results['kmedoids_sse'] > 0 else 0

    better_kmeans = "scikit-learn" if kmeans_results['total_sse'] < q3_results['kmeans_sse'] else "Custom"
    better_kmedoids = "scikit-learn" if kmedoids_results['total_sse'] < q3_results['kmedoids_sse'] else "Custom"

    print("├─" + "─" * 68 + "─┤")
    print(
        f"│ 📊 K-means difference: {kmeans_sse_diff_pct:.2f}% ({better_kmeans} implementation is better){' ' * (22 - len(f'{kmeans_sse_diff_pct:.2f}') - len(better_kmeans))} │")
    print(
        f"│ 📊 K-medoids difference: {kmedoids_sse_diff_pct:.2f}% ({better_kmedoids} implementation is better){' ' * (20 - len(f'{kmedoids_sse_diff_pct:.2f}') - len(better_kmedoids))} │")
    print("└─" + "─" * 68 + "─┘")

    # Reasons for differences with fancy formatting
    print("\n┏━" + "━" * 68 + "━┓")
    print("┃ " + "REASONS FOR DIFFERENCES IN PERFORMANCE".center(68) + " ┃")
    print("┣━" + "━" * 68 + "━┫")
    print("┃ 1. 🚀 Optimized code: scikit-learn uses highly optimized C/C++ implementations   ┃")
    print("┃ 2. 🔄 Vectorization: scikit-learn leverages NumPy's vectorized operations         ┃")
    print("┃ 3. 🔧 Implementation details: Different algorithm implementations affect results  ┃")
    print("┃ 4. 📊 Data structures: Optimized internal structures improve execution speed      ┃")
    print("┃ 5. 🧮 Numerical precision: Different floating-point handling affects convergence  ┃")
    print("┗━" + "━" * 68 + "━┛")


if __name__ == "__main__":
    main()