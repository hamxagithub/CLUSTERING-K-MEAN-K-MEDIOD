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
    plt.figure(figsize=(10, 6))
    methods = ['K-means\nCustom', 'K-means\nsklearn', 'K-medoids\nCustom', 'K-medoids\nsklearn']
    sses = [
        q3_results['kmeans_sse'],
        q4_results['kmeans']['total_sse'],
        q3_results['kmedoids_sse'],
        q4_results['kmedoids']['total_sse']
    ]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    bars = plt.bar(methods, sses, color=colors)

    # Add SSE values on bars
    for bar, sse_value in zip(bars, sses):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                 f'{sse_value:.2f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.title('SSE/WCSS Comparison: Custom vs sklearn', fontsize=14)
    plt.ylabel('SSE/WCSS Value')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('sklearn_vs_custom_sse.png', dpi=300)
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

    # Display only sklearn results
    print("\n" + "=" * 50)
    print("SKLEARN K-MEANS RESULTS")
    print("=" * 50)
    print(f"Total iterations: {kmeans_results['iterations']}")
    print(f"Cluster sizes: {kmeans_results['cluster_sizes']}")
    print(f"Cluster SSE: {[round(sse, 2) for sse in kmeans_results['cluster_sse']]}")
    print(f"Total SSE/WCSS: {round(kmeans_results['total_sse'], 2)}")
    print(f"Time complexity: {kmeans_results['time']:.4f} seconds")

    print("\n" + "=" * 50)
    print("SKLEARN K-MEDOIDS RESULTS")
    print("=" * 50)
    print(f"Total iterations: {kmedoids_results['iterations']}")
    print(f"Cluster sizes: {kmedoids_results['cluster_sizes']}")
    print(f"Cluster SSE: {[round(sse, 2) for sse in kmedoids_results['cluster_sse']]}")
    print(f"Total SSE/WCSS: {round(kmedoids_results['total_sse'], 2)}")
    print(f"Time complexity: {kmedoids_results['time']:.4f} seconds")

    # Visualize only sklearn clustering results
    visualize_sklearn_clusters(data, kmeans_results, kmedoids_results, used_columns, variable_info)

    # Create comparisons for time complexity and SSE
    visualize_time_complexity(q3_results, q4_results)
    visualize_sse_comparison(q3_results, q4_results)

    # Comparative analysis focusing only on performance metrics
    print("\n" + "=" * 50)
    print("PERFORMANCE COMPARISON: CUSTOM vs SKLEARN IMPLEMENTATIONS")
    print("=" * 50)

    # Time complexity comparison
    print("\nTime Complexity Comparison:")
    print(f"K-means: Custom={q3_results['kmeans_time']:.4f}s, sklearn={kmeans_results['time']:.4f}s")
    print(f"K-medoids: Custom={q3_results['kmedoids_time']:.4f}s, sklearn={kmedoids_results['time']:.4f}s")

    kmeans_speed_ratio = q3_results['kmeans_time'] / kmeans_results['time'] if kmeans_results['time'] > 0 else 0
    kmedoids_speed_ratio = q3_results['kmedoids_time'] / kmedoids_results['time'] if kmedoids_results['time'] > 0 else 0

    if kmeans_speed_ratio > 1:
        print(f"sklearn's K-means is {kmeans_speed_ratio:.2f}x faster than custom implementation")
    else:
        print(f"Custom K-means is {1 / kmeans_speed_ratio:.2f}x faster than sklearn implementation")

    if kmedoids_speed_ratio > 1:
        print(f"sklearn's K-medoids is {kmedoids_speed_ratio:.2f}x faster than custom implementation")
    else:
        print(f"Custom K-medoids is {1 / kmedoids_speed_ratio:.2f}x faster than sklearn implementation")

    # SSE comparison
    print("\nSSE/WCSS Comparison:")
    print(f"K-means: Custom={round(q3_results['kmeans_sse'], 2)}, sklearn={round(kmeans_results['total_sse'], 2)}")
    print(
        f"K-medoids: Custom={round(q3_results['kmedoids_sse'], 2)}, sklearn={round(kmedoids_results['total_sse'], 2)}")

    kmeans_sse_diff_pct = abs(q3_results['kmeans_sse'] - kmeans_results['total_sse']) / q3_results[
        'kmeans_sse'] * 100 if q3_results['kmeans_sse'] > 0 else 0
    kmedoids_sse_diff_pct = abs(q3_results['kmedoids_sse'] - kmedoids_results['total_sse']) / q3_results[
        'kmedoids_sse'] * 100 if q3_results['kmedoids_sse'] > 0 else 0

    print(f"K-means SSE difference: {kmeans_sse_diff_pct:.2f}%")
    print(f"K-medoids SSE difference: {kmedoids_sse_diff_pct:.2f}%")

    # Reasons for any differences in performance
    print("\nReasons for differences in performance:")
    print("1. Optimized code: scikit-learn uses highly optimized C/C++ implementations")
    print("2. Vectorization: scikit-learn leverages NumPy's vectorized operations")
    print("3. Implementation details: Different algorithm implementations can affect performance")
    print("4. Data structures: Optimized internal data structures can improve execution speed")


if __name__ == "__main__":
    main()