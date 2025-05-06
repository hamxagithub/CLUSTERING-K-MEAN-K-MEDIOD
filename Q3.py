import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def load_data(filename):
    try:
        print("Reading Information sheet to understand variable meanings...")
        try:
            info_df = pd.read_excel(filename, sheet_name="Information")
            print("\nVariable Information from Excel:")
            print(info_df)


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


        print("\nReading data from Normalized_Data sheet...")
        data = pd.read_excel(filename, sheet_name="Normalized_Data")

        column_names = list(data.columns)
        print(f"All columns in dataset: {column_names}")


        data = data.iloc[:, 0:3]
        used_columns = list(data.columns)


        data = data.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"Using columns for clustering: {used_columns}")


        return data.values, used_columns, variable_info
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None



def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))



def calculate_sse(clusters, centroids):
    sse = 0
    cluster_sse = []

    for i, cluster in enumerate(clusters):
        cluster_sum = 0
        if len(cluster) > 0:
            for point in cluster:
                cluster_sum += euclidean_distance(point, centroids[i]) ** 2
        sse += cluster_sum
        cluster_sse.append(cluster_sum)

    return sse, cluster_sse



def kmeans(data, k, max_iterations=100):

    np.random.seed(42)
    indices = np.random.choice(len(data), k, replace=False)
    centroids = data[indices]

    iteration_history = []

    for iteration in range(max_iterations):

        clusters = [[] for _ in range(k)]


        for point in data:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(point)

        # Store old centroids for convergence check
        old_centroids = centroids.copy()

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
        if np.all(old_centroids == centroids):
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


        old_medoids = medoids.copy()

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
        if np.all(old_medoids == medoids):
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


# Create visualization for clusters using information from Information sheet
def create_cluster_visualization(data, kmeans_clusters, kmeans_centroids,
                                 kmedoids_clusters, kmedoids_medoids,
                                 used_columns, variable_info):
    # Set up names for axis labels based on Information sheet
    axis_labels = []
    for col in used_columns:
        if variable_info and col in variable_info:
            axis_labels.append(f"{col}: {variable_info[col]}")
        else:
            axis_labels.append(col)

    # Create 2D visualization
    plt.figure(figsize=(15, 6))

    # K-means visualization
    plt.subplot(1, 2, 1)
    for i, cluster in enumerate(kmeans_clusters):
        if len(cluster) > 0:
            cluster = np.array(cluster)
            # Add slight jitter to reduce overlap
            jitter = np.random.normal(0, 0.01, cluster.shape)
            plt.scatter(cluster[:, 0] + jitter[:, 0],
                        cluster[:, 1] + jitter[:, 1],
                        alpha=0.6, s=50, label=f'Cluster {i + 1} (n={len(cluster)})')

    plt.scatter([c[0] for c in kmeans_centroids],
                [c[1] for c in kmeans_centroids],
                marker='*', s=300, c='black', label='Centroids')

    plt.title('K-means Clustering', fontsize=14)
    plt.xlabel(axis_labels[0] if len(axis_labels) > 0 else 'Feature 1', fontsize=12)
    plt.ylabel(axis_labels[1] if len(axis_labels) > 1 else 'Feature 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # K-medoid visualization
    plt.subplot(1, 2, 2)
    for i, cluster in enumerate(kmedoids_clusters):
        if len(cluster) > 0:
            cluster = np.array(cluster)
            # Add slight jitter to reduce overlap
            jitter = np.random.normal(0, 0.01, cluster.shape)
            plt.scatter(cluster[:, 0] + jitter[:, 0],
                        cluster[:, 1] + jitter[:, 1],
                        alpha=0.6, s=50, label=f'Cluster {i + 1} (n={len(cluster)})')

    plt.scatter([m[0] for m in kmedoids_medoids],
                [m[1] for m in kmedoids_medoids],
                marker='*', s=300, c='black', label='Medoids')

    plt.title('K-medoids Clustering', fontsize=14)
    plt.xlabel(axis_labels[0] if len(axis_labels) > 0 else 'Feature 1', fontsize=12)
    plt.ylabel(axis_labels[1] if len(axis_labels) > 1 else 'Feature 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig('clustering_comparison_with_info.png', dpi=300)
    plt.show()


# Create time complexity visualization
def visualize_time_complexity(kmeans_time, kmedoids_time):
    plt.figure(figsize=(10, 6))

    # Create bar chart for time comparison
    algorithms = ['K-means', 'K-medoids']
    times = [kmeans_time, kmedoids_time]
    colors = ['#1f77b4', '#ff7f0e']

    bars = plt.bar(algorithms, times, color=colors, width=0.5)

    # Add exact time values on top of bars
    for bar, time_value in zip(bars, times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{time_value:.4f}s',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Calculate speedup ratio
    speedup = kmedoids_time / kmeans_time if kmeans_time > 0 else 0
    plt.title(f'Time Complexity Comparison\nK-medoids took {speedup:.2f}x longer than K-means', fontsize=14)

    plt.ylabel('Execution Time (seconds)', fontsize=12)
    plt.ylim(0, max(times) * 1.2)  # Add some space at top for text

    # Add grid for better readability
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    plt.savefig('time_complexity_comparison.png', dpi=300)
    plt.show()


def main():
    # Load data
    data, used_columns, variable_info = load_data('Mine_Dataset.xlsx')

    if data is None:
        print("Failed to load data. Please check the file path.")
        return

    print(f"Dataset loaded successfully with {data.shape[0]} samples and {data.shape[1]} features.")

    # 1. Create elbow chart and find optimal K
    k_range = range(1, 11)  # At least 10 SSE/WCSS values of K
    sse_values = create_elbow_chart(data, k_range)

    # Find the "elbow point" using simple heuristic
    sse_decrease = [sse_values[i - 1] - sse_values[i] for i in range(1, len(sse_values))]
    decreases = np.array(sse_decrease)
    normalized_decreases = decreases / decreases[0]

    # Find where the rate of decrease slows down
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

    # Display results
    display_results("K-means", kmeans_history)
    print(f"Time complexity: {kmeans_time:.4f} seconds")

    display_results("K-medoids", kmedoids_history)
    print(f"Time complexity: {kmedoids_time:.4f} seconds")

    # Print time comparison
    print("\n" + "=" * 50)
    print("TIME COMPLEXITY COMPARISON")
    print("=" * 50)
    print(f"K-means:   {kmeans_time:.4f} seconds")
    print(f"K-medoids: {kmedoids_time:.4f} seconds")
    print(f"Ratio:     K-medoids took {kmedoids_time / kmeans_time:.2f}x longer than K-means")
    print("=" * 50)

    # Comparative analysis
    print("\n\nComparative Analysis:")
    print("-" * 50)
    print(f"K-means final SSE: {kmeans_history[-1]['total_sse']:.2f}")
    print(f"K-medoids final SSE: {kmedoids_history[-1]['total_sse']:.2f}")

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

    # Overall winner determination
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

    # Create visualizations
    create_cluster_visualization(data, kmeans_clusters, kmeans_centroids,
                                 kmedoids_clusters, kmedoids_medoids,
                                 used_columns, variable_info)
    visualize_time_complexity(kmeans_time, kmedoids_time)


if __name__ == "__main__":
    main()