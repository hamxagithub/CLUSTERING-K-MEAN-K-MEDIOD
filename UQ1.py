import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


# Load the dataset
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


# Calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


# Calculate SSE/WCSS (Within-Cluster Sum of Squares)
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

        # Store old medoids for convergence check
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

    # Plot the elbow chart with modern dark theme
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.set_facecolor('#0D1117')
    fig.patch.set_facecolor('#0D1117')

    # Enhanced visualization with gradient line
    gradient_points = 100
    for i in range(len(k_range) - 1):
        # Create gradient segments
        for j in range(gradient_points):
            segment_start = k_range[i] + j / gradient_points
            segment_end = k_range[i] + (j + 1) / gradient_points

            segment_start_sse = sse_values[i] + (sse_values[i + 1] - sse_values[i]) * (j / gradient_points)
            segment_end_sse = sse_values[i] + (sse_values[i + 1] - sse_values[i]) * ((j + 1) / gradient_points)

            # Color goes from cyan to magenta based on elbow curve position
            pos = i / (len(k_range) - 2) if len(k_range) > 2 else 0.5
            color = (pos, 1 - pos, 1)

            ax.plot([segment_start, segment_end],
                    [segment_start_sse, segment_end_sse],
                    color=color, linewidth=3, alpha=0.8)

    # Add data points with glowing effect
    for k_idx, (k, sse) in enumerate(zip(k_range, sse_values)):
        # Add glow effect
        for size, alpha in zip([150, 100, 60], [0.1, 0.2, 0.3]):
            ax.scatter(k, sse, s=size, color='cyan' if k_idx < len(k_range) / 2 else 'magenta',
                       alpha=alpha, edgecolor='none')

        # Main point
        ax.scatter(k, sse, s=120, color='white', edgecolor='cyan' if k_idx < len(k_range) / 2 else 'magenta',
                   linewidth=2, alpha=0.9)

        # Add value label
        ax.text(k, sse * 1.03, f"{sse:.2f}", ha='center', va='bottom',
                color='white', fontweight='bold', fontsize=9)

    # Find and highlight the elbow point
    sse_decrease = [sse_values[i - 1] - sse_values[i] for i in range(1, len(sse_values))]
    decreases = np.array(sse_decrease)
    normalized_decreases = decreases / decreases[0]
    elbow_idx = np.where(normalized_decreases < 0.15)[0]
    elbow_k = k_range[elbow_idx[0]] if len(elbow_idx) > 0 else k_range[1]
    elbow_sse = sse_values[k_range.index(elbow_k)]

    # Highlight elbow point with annotation
    ax.scatter(elbow_k, elbow_sse, s=180, color='yellow', alpha=0.8,
               marker='*', edgecolors='white', linewidth=2, zorder=10)
    ax.annotate(f'Optimal k={elbow_k}', xy=(elbow_k, elbow_sse),
                xytext=(elbow_k + 1, elbow_sse * 1.2),
                arrowprops=dict(arrowstyle="->", color='yellow', linewidth=2),
                fontsize=12, fontweight='bold', color='yellow')

    # Set labels and title with glowing effect
    ax.set_xlabel('Number of clusters (k)', fontsize=14, color='white', fontweight='bold')
    ax.set_ylabel('SSE/WCSS', fontsize=14, color='white', fontweight='bold')

    # Add glowing title
    title_text = 'Elbow Method for Optimal k'
    for offset in [0.3, 0.2, 0.1]:
        ax.text(np.mean(k_range), max(sse_values) * 1.2 + offset, title_text,
                color='white', alpha=offset,
                fontsize=18, fontweight='bold', ha='center')
    ax.text(np.mean(k_range), max(sse_values) * 1.2, title_text,
            color='white', fontsize=18, fontweight='bold', ha='center')

    # Add descriptive text
    ax.text(np.mean(k_range), max(sse_values) * 1.1,
            "Finding the point where adding another cluster doesn't significantly reduce error",
            color='gray', fontsize=10, fontstyle='italic', ha='center')

    # Add grid with futuristic styling
    ax.grid(True, linestyle='--', alpha=0.2)

    # Add radial background
    theta = np.linspace(0, 2 * np.pi, 100)
    for radius in [max(sse_values) * 0.25, max(sse_values) * 0.5, max(sse_values) * 0.75]:
        circle_x = np.mean(k_range) + (max(k_range) - min(k_range)) / 2 * np.cos(theta)
        circle_y = max(sse_values) / 2 + radius * np.sin(theta)
        ax.plot(circle_x, circle_y, 'white', alpha=0.05, linewidth=0.5)

    # Clean up axes
    ax.set_xticks(k_range)
    ax.set_xlim(min(k_range) - 0.5, max(k_range) + 0.5)
    ax.set_ylim(0, max(sse_values) * 1.3)

    plt.tight_layout()
    plt.savefig('elbow_chart.png', dpi=300, bbox_inches='tight', facecolor='#0D1117')
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

    # Create 2D visualization with improved styling
    plt.figure(figsize=(16, 8))

    # Define distinct color palette for better differentiation
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6', '#c4c1e0', '#f0b3ff']

    # K-means visualization
    plt.subplot(1, 2, 1)
    for i, cluster in enumerate(kmeans_clusters):
        if len(cluster) > 0:
            cluster = np.array(cluster)
            # Add slight jitter to reduce overlap
            jitter = np.random.normal(0, 0.01, cluster.shape)
            plt.scatter(cluster[:, 0] + jitter[:, 0],
                        cluster[:, 1] + jitter[:, 1],
                        alpha=0.7, s=60, label=f'Cluster {i + 1} (n={len(cluster)})',
                        color=colors[i % len(colors)], edgecolors='w', linewidths=0.5)

    plt.scatter([c[0] for c in kmeans_centroids],
                [c[1] for c in kmeans_centroids],
                marker='*', s=350, c='black', label='Centroids', edgecolors='white', linewidths=1.5)

    plt.title('K-means Clustering', fontsize=16, fontweight='bold')
    plt.xlabel(axis_labels[0] if len(axis_labels) > 0 else 'Feature 1', fontsize=13)
    plt.ylabel(axis_labels[1] if len(axis_labels) > 1 else 'Feature 2', fontsize=13)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=10, framealpha=0.7)

    # Add border to the subplot
    plt.gca().set_facecolor('#f8f9fa')
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#333333')

    # K-medoid visualization
    plt.subplot(1, 2, 2)
    for i, cluster in enumerate(kmedoids_clusters):
        if len(cluster) > 0:
            cluster = np.array(cluster)
            # Add slight jitter to reduce overlap
            jitter = np.random.normal(0, 0.01, cluster.shape)
            plt.scatter(cluster[:, 0] + jitter[:, 0],
                        cluster[:, 1] + jitter[:, 1],
                        alpha=0.7, s=60, label=f'Cluster {i + 1} (n={len(cluster)})',
                        color=colors[i % len(colors)], edgecolors='w', linewidths=0.5)

    plt.scatter([m[0] for m in kmedoids_medoids],
                [m[1] for m in kmedoids_medoids],
                marker='X', s=350, c='black', label='Medoids', edgecolors='white', linewidths=1.5)

    plt.title('K-medoids Clustering', fontsize=16, fontweight='bold')
    plt.xlabel(axis_labels[0] if len(axis_labels) > 0 else 'Feature 1', fontsize=13)
    plt.ylabel(axis_labels[1] if len(axis_labels) > 1 else 'Feature 2', fontsize=13)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=10, framealpha=0.7)

    # Add border to the subplot
    plt.gca().set_facecolor('#f8f9fa')
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#333333')

    # Add a title for the overall figure
    plt.suptitle('Comparison of Clustering Algorithms', fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the suptitle
    plt.savefig('clustering_comparison_with_info.png', dpi=300, bbox_inches='tight')
    plt.show()


# Create time complexity visualization
def visualize_time_complexity(kmeans_time, kmedoids_time):
    plt.figure(figsize=(12, 7))

    # Create bar chart for time comparison with improved styling
    algorithms = ['K-means', 'K-medoids']
    times = [kmeans_time, kmedoids_time]
    colors = ['#2c7fb8', '#d95f02']

    bars = plt.bar(algorithms, times, color=colors, width=0.6, edgecolor='black', linewidth=1.5, alpha=0.85)

    # Add exact time values on top of bars
    for bar, time_value in zip(bars, times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{time_value:.4f}s',
                 ha='center', va='bottom', fontsize=13, fontweight='bold', color='#333333')

    # Calculate speedup ratio
    speedup = kmedoids_time / kmeans_time if kmeans_time > 0 else 0
    plt.title(f'Time Complexity Comparison\nK-medoids took {speedup:.2f}x longer than K-means',
              fontsize=16, fontweight='bold', pad=20)

    plt.ylabel('Execution Time (seconds)', fontsize=14, fontweight='bold')
    plt.ylim(0, max(times) * 1.25)  # Add some space at top for text

    # Add data labels inside bars
    for i, v in enumerate(times):
        if v > max(times) * 0.05:  # Only show inside label if bar is tall enough
            plt.text(i, v / 2, f'{v:.4f}s',
                     ha='center', va='center', color='white', fontweight='bold', fontsize=12)

    # Add grid for better readability
    plt.grid(True, axis='y', alpha=0.3, linestyle='--')

    # Add a light background color
    plt.gca().set_facecolor('#f5f5f5')

    # Add border to the plot
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#333333')

    # Add annotations
    if speedup > 1:
        plt.annotate(f'K-means is {speedup:.2f}x faster!', xy=(0, kmeans_time * 1.1),
                     xytext=(0.5, max(times) * 0.8),
                     arrowprops=dict(facecolor='#333333', shrink=0.05, width=2, headwidth=10),
                     ha='center', va='center', fontsize=13, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor='#f0f0f0', alpha=0.8))

    plt.tight_layout()
    plt.savefig('time_complexity_comparison.png', dpi=300, bbox_inches='tight')
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