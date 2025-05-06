import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from copy import deepcopy


# Load the dataset
def load_data(filename):
    try:
        # First, try to import openpyxl directly to check if it's installed
        try:
            import openpyxl
            print("openpyxl is installed.")
        except ImportError:
            print("\nERROR: Missing required dependency 'openpyxl' for reading Excel files.")
            print("Please install it using: pip install openpyxl")
            print("Attempting to continue by trying to auto-install openpyxl...\n")

            # Try to automatically install openpyxl
            try:
                import subprocess
                print("Installing openpyxl...")
                subprocess.check_call(["pip", "install", "openpyxl"])
                print("openpyxl installed successfully!")
                # Import it now that it's installed
                import openpyxl
            except Exception as e:
                print(f"Auto-installation failed: {e}")
                print("\nAlternative: If you have CSV version of the dataset, modify the code to use:")
                print("data = pd.read_csv('your_file.csv')")
                return None, None, None

        # First, read the Information sheet to get details about the variables
        print(f"Reading Information sheet from {filename}...")
        try:
            info_df = pd.read_excel(filename, sheet_name="Information")
            print("Information sheet found. Analyzing variable information...")
            # Print the information to understand the data structure
            print("\nVariable Information:")
            print(info_df)

            # Store variable descriptions and types for better visualization labels
            variable_info = {}
            for _, row in info_df.iterrows():
                if 'Variable' in row and 'Description' in row:
                    var_name = str(row['Variable'])
                    var_desc = str(row['Description'])
                    var_type = 'Dependent' if 'dependent' in var_desc.lower() else 'Independent'
                    variable_info[var_name] = {
                        'description': var_desc,
                        'type': var_type
                    }

            print("\nIdentified variables:")
            for var, info in variable_info.items():
                print(f"- {var}: {info['type']} - {info['description']}")

        except Exception as e:
            print(f"Could not process Information sheet: {e}")
            print("Proceeding without variable information.")
            variable_info = {}

        # Now read the Normalized_Data sheet
        print(f"\nReading data from Normalized_Data sheet...")
        data = pd.read_excel(filename, sheet_name="Normalized_Data")

        # Get column names before filtering
        all_columns = list(data.columns)
        print(f"All columns in data: {all_columns}")

        # Use only the first three columns (ignore the last column labeled 'M')
        data = data.iloc[:, 0:3]

        # Shuffle the records
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)

        # Get the column names we're using
        used_columns = list(data.columns)
        print(f"Using columns for clustering: {used_columns}")

        # Return data and variable information
        return data.values, used_columns, variable_info

    except Exception as e:
        print(f"Error loading data: {e}")
        print("\nTroubleshooting steps:")
        print("1. Make sure the file exists at the path:", filename)
        print("2. Make sure the file has 'Normalized_Data' and 'Information' sheets")
        print("3. Check if the first three columns contain numeric data")

        # Check if file exists
        import os
        if not os.path.exists(filename):
            print(f"\nFile '{filename}' not found. Available files in the current directory:")
            files = os.listdir('.')
            for f in files:
                if f.endswith(('.xlsx', '.xls', '.csv')):
                    print(f"- {f}")

        return None, None, None


# Calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


# Calculate Manhattan distance between two points
def manhattan_distance(point1, point2):
    return np.sum(np.abs(point1 - point2))


# Calculate SSE/WCSS (Within-Cluster Sum of Squares)
def calculate_sse(clusters, centroids, distance_func=euclidean_distance):
    sse = 0
    cluster_sse = []

    for i, cluster in enumerate(clusters):
        cluster_sum = 0
        if len(cluster) > 0:
            for point in cluster:
                cluster_sum += distance_func(point, centroids[i]) ** 2
        sse += cluster_sum
        cluster_sse.append(cluster_sum)

    return sse, cluster_sse


# K-means clustering implementation
def kmeans(data, k, max_iterations=100):
    # Initialize centroids by randomly selecting k data points
    np.random.seed(42)  # For reproducibility
    indices = np.random.choice(len(data), k, replace=False)
    centroids = data[indices]

    iteration_history = []

    for iteration in range(max_iterations):
        # Initialize clusters
        clusters = [[] for _ in range(k)]

        # Assign points to nearest centroid
        for point in data:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(point)

        # Store old centroids for convergence check
        old_centroids = deepcopy(centroids)

        # Update centroids
        for i in range(k):
            if len(clusters[i]) > 0:
                centroids[i] = np.mean(clusters[i], axis=0)

        # Calculate SSE for this iteration
        total_sse, cluster_sse = calculate_sse(clusters, centroids)

        # Store iteration results
        iteration_info = {
            'iteration': iteration + 1,
            'cluster_sizes': [len(cluster) for cluster in clusters],
            'cluster_sse': cluster_sse,
            'total_sse': total_sse
        }
        iteration_history.append(iteration_info)

        # Check for convergence
        if all(np.array_equal(old_centroids[i], centroids[i]) for i in range(k)):
            break

    return clusters, centroids, iteration_history


# K-medoid clustering implementation
def kmedoids(data, k, max_iterations=100):
    # Initialize medoids by randomly selecting k data points
    np.random.seed(42)  # For reproducibility
    indices = np.random.choice(len(data), k, replace=False)
    medoid_indices = indices.copy()
    medoids = data[medoid_indices]

    iteration_history = []

    for iteration in range(max_iterations):
        # Initialize clusters
        clusters = [[] for _ in range(k)]
        cluster_indices = [[] for _ in range(k)]

        # Assign points to nearest medoid
        for idx, point in enumerate(data):
            distances = [euclidean_distance(point, medoid) for medoid in medoids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(point)
            cluster_indices[cluster_idx].append(idx)

        # Store old medoids for convergence check
        old_medoids = deepcopy(medoids)

        # Update medoids - find the point in each cluster that minimizes total distance
        for i in range(k):
            if len(clusters[i]) > 0:
                min_distance = float('inf')
                new_medoid_idx = -1

                for idx in cluster_indices[i]:
                    total_distance = sum(euclidean_distance(data[idx], data[j]) for j in cluster_indices[i])
                    if total_distance < min_distance:
                        min_distance = total_distance
                        new_medoid_idx = idx

                if new_medoid_idx != -1:
                    medoid_indices[i] = new_medoid_idx
                    medoids[i] = data[new_medoid_idx]

        # Calculate SSE for this iteration
        total_sse, cluster_sse = calculate_sse(clusters, medoids)

        # Store iteration results
        iteration_info = {
            'iteration': iteration + 1,
            'cluster_sizes': [len(cluster) for cluster in clusters],
            'cluster_sse': cluster_sse,
            'total_sse': total_sse
        }
        iteration_history.append(iteration_info)

        # Check for convergence
        if all(np.array_equal(old_medoids[i], medoids[i]) for i in range(k)):
            break

    return clusters, medoids, iteration_history


# Function to create elbow chart
def create_elbow_chart(data, k_range):
    sse_values = []

    for k in k_range:
        print(f"Calculating SSE for k={k}...")
        _, _, iteration_history = kmeans(data, k, max_iterations=100)
        sse_values.append(iteration_history[-1]['total_sse'])

    # Plot the elbow chart
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, sse_values, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('SSE/WCSS')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.savefig('elbow_chart.png')
    plt.show()

    return sse_values


# Function to display clustering results
def display_results(algorithm_name, iteration_history):
    print(f"\n{algorithm_name} Clustering Results:")

    final_iteration = iteration_history[-1]
    print(f"Total Iterations: {final_iteration['iteration']}")

    print("\nIteration History:")
    for info in iteration_history:
        print(f"\nIteration {info['iteration']}:")
        print(f"  Cluster sizes: {info['cluster_sizes']}")
        print(f"  Cluster SSE: {[round(sse, 2) for sse in info['cluster_sse']]}")
        print(f"  Total SSE: {round(info['total_sse'], 2)}")


# Save the final cluster visualization
def create_visualizations(data, kmeans_clusters, kmeans_centroids, kmedoids_clusters, kmedoids_medoids, column_names,
                          variable_info):
    # Color palette for consistency
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Get better axis labels if we have variable information
    x_label = column_names[0] if len(column_names) > 0 else 'Feature 1'
    y_label = column_names[1] if len(column_names) > 1 else 'Feature 2'
    z_label = column_names[2] if len(column_names) > 2 else 'Feature 3'

    # Get variable descriptions if available
    if variable_info and x_label in variable_info:
        x_desc = f"{x_label} - {variable_info[x_label]['description']}"
        x_type = variable_info[x_label]['type']
        x_label = f"{x_label} ({x_type})"
    else:
        x_desc = x_label

    if variable_info and y_label in variable_info:
        y_desc = f"{y_label} - {variable_info[y_label]['description']}"
        y_type = variable_info[y_label]['type']
        y_label = f"{y_label} ({y_type})"
    else:
        y_desc = y_label

    if variable_info and z_label in variable_info and len(column_names) > 2:
        z_desc = f"{z_label} - {variable_info[z_label]['description']}"
        z_type = variable_info[z_label]['type']
        z_label = f"{z_label} ({z_type})"
    else:
        z_desc = z_label

    # Print the descriptions to help understand the variables
    print("\nVariable Descriptions used in visualization:")
    print(f"X-axis: {x_desc}")
    print(f"Y-axis: {y_desc}")
    if len(column_names) > 2:
        print(f"Z-axis: {z_desc}")

    # 1. 2D Visualization with improved styling
    plt.figure(figsize=(16, 7))

    # K-means visualization
    plt.subplot(1, 2, 1)
    for i, cluster in enumerate(kmeans_clusters):
        if len(cluster) > 0:
            cluster = np.array(cluster)
            # Add slight jitter to help with overlapping points
            jitter = np.random.normal(0, 0.01, cluster.shape)
            plt.scatter(cluster[:, 0] + jitter[:, 0],
                        cluster[:, 1] + jitter[:, 1],
                        alpha=0.6,  # Add transparency
                        s=50,  # Increase point size
                        color=colors[i % len(colors)],
                        label=f'Cluster {i + 1} (n={len(cluster)})')

    plt.scatter([c[0] for c in kmeans_centroids], [c[1] for c in kmeans_centroids],
                marker='*', s=300, c='black', label='Centroids', edgecolor='white')
    plt.title('K-means Clustering', fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # K-medoids visualization
    plt.subplot(1, 2, 2)
    for i, cluster in enumerate(kmedoids_clusters):
        if len(cluster) > 0:
            cluster = np.array(cluster)
            # Add slight jitter to help with overlapping points
            jitter = np.random.normal(0, 0.01, cluster.shape)
            plt.scatter(cluster[:, 0] + jitter[:, 0],
                        cluster[:, 1] + jitter[:, 1],
                        alpha=0.6,  # Add transparency
                        s=50,  # Increase point size
                        color=colors[i % len(colors)],
                        label=f'Cluster {i + 1} (n={len(cluster)})')

    plt.scatter([m[0] for m in kmedoids_medoids], [m[1] for m in kmedoids_medoids],
                marker='*', s=300, c='black', label='Medoids', edgecolor='white')
    plt.title('K-medoids Clustering', fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig('clustering_comparison_2d.png', dpi=300)

    # 2. 3D Visualization
    if data.shape[1] >= 3:  # We need at least 3 dimensions for 3D plots
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(16, 7))

        # K-means 3D visualization
        ax1 = fig.add_subplot(121, projection='3d')
        for i, cluster in enumerate(kmeans_clusters):
            if len(cluster) > 0:
                cluster = np.array(cluster)
                ax1.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2],
                            alpha=0.6,
                            s=50,
                            color=colors[i % len(colors)],
                            label=f'Cluster {i + 1}')

        ax1.scatter([c[0] for c in kmeans_centroids],
                    [c[1] for c in kmeans_centroids],
                    [c[2] for c in kmeans_centroids],
                    marker='*', s=300, c='black', label='Centroids')

        ax1.set_title('K-means 3D Clustering', fontsize=14)
        ax1.set_xlabel(x_label, fontsize=12)
        ax1.set_ylabel(y_label, fontsize=12)
        ax1.set_zlabel(z_label, fontsize=12)
        ax1.legend()

        # K-medoids 3D visualization
        ax2 = fig.add_subplot(122, projection='3d')
        for i, cluster in enumerate(kmedoids_clusters):
            if len(cluster) > 0:
                cluster = np.array(cluster)
                ax2.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2],
                            alpha=0.6,
                            s=50,
                            color=colors[i % len(colors)],
                            label=f'Cluster {i + 1}')

        ax2.scatter([m[0] for m in kmedoids_medoids],
                    [m[1] for m in kmedoids_medoids],
                    [m[2] for m in kmedoids_medoids],
                    marker='*', s=300, c='black', label='Medoids')

        ax2.set_title('K-medoids 3D Clustering', fontsize=14)
        ax2.set_xlabel(x_label, fontsize=12)
        ax2.set_ylabel(y_label, fontsize=12)
        ax2.set_zlabel(z_label, fontsize=12)
        ax2.legend()

        plt.tight_layout()
        plt.savefig('clustering_comparison_3d.png', dpi=300)

    # 3. Feature pair plots for more clarity
    if data.shape[1] >= 3:
        plt.figure(figsize=(15, 5))

        # Assign cluster labels for coloring
        kmeans_labels = np.zeros(len(data))
        for i, cluster in enumerate(kmeans_clusters):
            for point in cluster:
                idx = np.where((data == point).all(axis=1))[0]
                if len(idx) > 0:
                    kmeans_labels[idx[0]] = i

        # F1 vs F2
        plt.subplot(1, 3, 1)
        for i in range(len(kmeans_clusters)):
            mask = kmeans_labels == i
            plt.scatter(data[mask, 0], data[mask, 1],
                        alpha=0.7, color=colors[i % len(colors)],
                        label=f'Cluster {i + 1}')

        plt.scatter([c[0] for c in kmeans_centroids],
                    [c[1] for c in kmeans_centroids],
                    marker='*', s=200, c='black', label='Centroids')
        plt.title(f'{x_label} vs {y_label}')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True, alpha=0.3)

        # F1 vs F3
        plt.subplot(1, 3, 2)
        for i in range(len(kmeans_clusters)):
            mask = kmeans_labels == i
            plt.scatter(data[mask, 0], data[mask, 2],
                        alpha=0.7, color=colors[i % len(colors)])

        plt.scatter([c[0] for c in kmeans_centroids],
                    [c[2] for c in kmeans_centroids],
                    marker='*', s=200, c='black')
        plt.title(f'{x_label} vs {z_label}')
        plt.xlabel(x_label)
        plt.ylabel(z_label)
        plt.grid(True, alpha=0.3)

        # F2 vs F3
        plt.subplot(1, 3, 3)
        for i in range(len(kmeans_clusters)):
            mask = kmeans_labels == i
            plt.scatter(data[mask, 1], data[mask, 2],
                        alpha=0.7, color=colors[i % len(colors)])

        plt.scatter([c[1] for c in kmeans_centroids],
                    [c[2] for c in kmeans_centroids],
                    marker='*', s=200, c='black')
        plt.title(f'{y_label} vs {z_label}')
        plt.xlabel(y_label)
        plt.ylabel(z_label)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('feature_pair_plots.png', dpi=300)

    plt.show()


def main():
    # Load data
    data, used_columns, variable_info = load_data('Mine_Dataset.xlsx')

    if data is None:
        print("Failed to load data. Please check the file path.")
        return

    print(f"Dataset loaded successfully with {data.shape[0]} samples and {data.shape[1]} features.")

    # 1. Create elbow chart and find optimal K
    k_range = range(1, 11)
    sse_values = create_elbow_chart(data, k_range)

    # Find the "elbow point" - this is a heuristic approach
    # Calculate the rate of SSE decrease
    sse_decrease = [sse_values[i - 1] - sse_values[i] for i in range(1, len(sse_values))]

    # Find where the rate of decrease slows down (you might want to manually adjust this)
    decreases = np.array(sse_decrease)
    normalized_decreases = decreases / decreases[0]

    # A simple heuristic: find the first point where the normalized decrease is less than 0.15
    optimal_k_idx = np.where(normalized_decreases < 0.15)[0]
    optimal_k = k_range[optimal_k_idx[0]] if len(optimal_k_idx) > 0 else 3

    print(f"\nBased on the elbow chart, the optimal value of K appears to be: {optimal_k}")

    # 2. Run K-means and K-medoid with optimal K
    print(f"\nRunning K-means with K={optimal_k}...")
    start_time = time.time()
    kmeans_clusters, kmeans_centroids, kmeans_history = kmeans(data, optimal_k)
    kmeans_time = time.time() - start_time

    print(f"\nRunning K-medoids with K={optimal_k}...")
    start_time = time.time()
    kmedoids_clusters, kmedoids_medoids, kmedoids_history = kmedoids(data, optimal_k)
    kmedoids_time = time.time() - start_time

    # 3. Display results
    display_results("K-means", kmeans_history)
    print(f"Time complexity: {kmeans_time:.4f} seconds")

    display_results("K-medoids", kmedoids_history)
    print(f"Time complexity: {kmedoids_time:.4f} seconds")

    # Create time complexity visualization
    plt.figure(figsize=(10, 6))
    methods = ['K-means', 'K-medoids']
    times = [kmeans_time, kmedoids_time]
    colors = ['#1f77b4', '#ff7f0e']

    plt.bar(methods, times, color=colors)
    plt.ylabel('Execution Time (seconds)')
    plt.title('Time Complexity Comparison', fontsize=14)

    # Add time values as text on the bars
    for i, v in enumerate(times):
        plt.text(i, v + 0.1, f'{v:.4f} seconds',
                 ha='center', va='bottom', fontweight='bold')

    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('time_complexity_comparison.png', dpi=300)
    plt.show()

    # Print time complexity comparison in a prominent way
    print("\n" + "=" * 50)
    print("TIME COMPLEXITY COMPARISON")
    print("=" * 50)
    print(f"K-means:   {kmeans_time:.4f} seconds")
    print(f"K-medoids: {kmedoids_time:.4f} seconds")
    print(f"Ratio:     K-medoids took {kmedoids_time / kmeans_time:.2f}x longer than K-means")
    print("=" * 50)

    # 4. Comparative analysis
    print("\n\nComparative Analysis:")
    print("-" * 50)
    print(f"K-means final SSE: {kmeans_history[-1]['total_sse']:.2f}")
    print(f"K-medoids final SSE: {kmedoids_history[-1]['total_sse']:.2f}")
    print(f"K-means runtime: {kmeans_time:.4f} seconds")
    print(f"K-medoids runtime: {kmedoids_time:.4f} seconds")

    print("\nAnalysis:")
    if kmeans_history[-1]['total_sse'] < kmedoids_history[-1]['total_sse']:
        sse_winner = "K-means"
    else:
        sse_winner = "K-medoids"

    print(f"- {sse_winner} achieved a lower SSE/WCSS, indicating tighter clusters.")

    if kmeans_time < kmedoids_time:
        time_winner = "K-means"
    else:
        time_winner = "K-medoids"

    print(f"- {time_winner} was faster to execute.")

    print("\nContextual Performance:")
    print("- K-means: Better for spherical clusters and larger datasets due to lower time complexity.")
    print("- K-medoids: More robust to outliers and performs better with non-spherical clusters.")

    # Overall winner determination based on both metrics
    if sse_winner == time_winner:
        print(f"\nOverall Winner for this dataset: {sse_winner}")
    else:
        # If there's a trade-off, consider the magnitude of differences
        sse_diff = abs(kmeans_history[-1]['total_sse'] - kmedoids_history[-1]['total_sse'])
        time_diff = abs(kmeans_time - kmedoids_time)

        if (sse_diff / kmeans_history[-1]['total_sse']) > (time_diff / max(kmeans_time, kmedoids_time)):
            print(f"\nOverall Winner for this dataset: {sse_winner} (SSE advantage outweighs time difference)")
        else:
            print(f"\nOverall Winner for this dataset: {time_winner} (Time advantage outweighs SSE difference)")

    # Create enhanced visualizations
    create_visualizations(data, kmeans_clusters, kmeans_centroids, kmedoids_clusters, kmedoids_medoids, used_columns,
                          variable_info)

    # 5. Create detailed cluster analysis visualization
    plt.figure(figsize=(14, 8))

    # Plot SSE per cluster for K-means
    ax1 = plt.subplot(2, 2, 1)
    cluster_sse_kmeans = []
    for i, cluster in enumerate(kmeans_clusters):
        if len(cluster) > 0:
            # Calculate SSE for this cluster
            sse = sum(euclidean_distance(point, kmeans_centroids[i]) ** 2 for point in cluster)
            cluster_sse_kmeans.append(sse)
        else:
            cluster_sse_kmeans.append(0)

    ax1.bar(range(1, len(cluster_sse_kmeans) + 1), cluster_sse_kmeans, color='#1f77b4')
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('SSE/WCSS')
    ax1.set_title('K-means: SSE per Cluster')
    ax1.grid(True, alpha=0.3)

    # Plot SSE per cluster for K-medoids
    ax2 = plt.subplot(2, 2, 2)
    cluster_sse_kmedoids = []
    for i, cluster in enumerate(kmedoids_clusters):
        if len(cluster) > 0:
            # Calculate SSE for this cluster
            sse = sum(euclidean_distance(point, kmedoids_medoids[i]) ** 2 for point in cluster)
            cluster_sse_kmedoids.append(sse)
        else:
            cluster_sse_kmedoids.append(0)

    ax2.bar(range(1, len(cluster_sse_kmedoids) + 1), cluster_sse_kmedoids, color='#ff7f0e')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('SSE/WCSS')
    ax2.set_title('K-medoids: SSE per Cluster')
    ax2.grid(True, alpha=0.3)

    # Plot cluster sizes for K-means
    ax3 = plt.subplot(2, 2, 3)
    cluster_sizes_kmeans = [len(cluster) for cluster in kmeans_clusters]
    ax3.bar(range(1, len(cluster_sizes_kmeans) + 1), cluster_sizes_kmeans, color='#1f77b4')
    ax3.set_xlabel('Cluster')
    ax3.set_ylabel('Number of Points')
    ax3.set_title('K-means: Cluster Sizes')
    ax3.grid(True, alpha=0.3)

    # Plot cluster sizes for K-medoids
    ax4 = plt.subplot(2, 2, 4)
    cluster_sizes_kmedoids = [len(cluster) for cluster in kmedoids_clusters]
    ax4.bar(range(1, len(cluster_sizes_kmedoids) + 1), cluster_sizes_kmedoids, color='#ff7f0e')
    ax4.set_xlabel('Cluster')
    ax4.set_ylabel('Number of Points')
    ax4.set_title('K-medoids: Cluster Sizes')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cluster_analysis.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()