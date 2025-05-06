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

    # Create a dark style plot with modern aesthetics
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 10))

    # Custom color palette - neon colors for dark background
    neon_colors = ['#00FFFF', '#FF00FF', '#00FF00', '#FFFF00', '#FF9933', '#33CCFF', '#FF3399', '#99FF33']

    # Create a 2x2 grid for different views of the data
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

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

    # Add data points with glowing effect for K-means
    for i, cluster in enumerate(kmeans_clusters):
        if len(cluster) > 0:
            cluster = np.array(cluster)

            # Create glow effect
            for size, alpha in zip([100, 80, 60], [0.1, 0.2, 0.3]):
                ax1.scatter(cluster[:, 0], cluster[:, 1],
                            s=size, color=neon_colors[i % len(neon_colors)], alpha=alpha, edgecolors='none')

            # Main points
            ax1.scatter(cluster[:, 0], cluster[:, 1],
                        s=40, color=neon_colors[i % len(neon_colors)], alpha=0.9,
                        edgecolors='white', linewidths=0.5,
                        label=f'Cluster {i + 1} (n={len(cluster)})')

    # Add starburst effect for centroids
    for centroid in kmeans_centroids:
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

    # Add data points with glowing effect for K-medoids
    for i, cluster in enumerate(kmedoids_clusters):
        if len(cluster) > 0:
            cluster = np.array(cluster)

            # Create glow effect
            for size, alpha in zip([100, 80, 60], [0.1, 0.2, 0.3]):
                ax2.scatter(cluster[:, 0], cluster[:, 1],
                            s=size, color=neon_colors[i % len(neon_colors)], alpha=alpha, edgecolors='none')

            # Main points
            ax2.scatter(cluster[:, 0], cluster[:, 1],
                        s=40, color=neon_colors[i % len(neon_colors)], alpha=0.9,
                        edgecolors='white', linewidths=0.5,
                        label=f'Cluster {i + 1} (n={len(cluster)})')

    # Add diamond effect for medoids
    for medoid in kmedoids_medoids:
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
    for i, cluster in enumerate(kmeans_clusters):
        if len(cluster) > 0:
            cluster = np.array(cluster)

            # Calculate distances to respective centroid for z-axis
            distances = np.sqrt(np.sum((cluster[:, :2] - kmeans_centroids[i][:2]) ** 2, axis=1))

            # Plot 3D scatter
            ax3.scatter(cluster[:, 0], cluster[:, 1], distances,
                        s=40, color=neon_colors[i % len(neon_colors)], alpha=0.7,
                        edgecolors='white', linewidths=0.5)

            # Plot vertical lines connecting points to their projection on xy-plane
            for j in range(min(len(distances), 20)):  # Limit to 20 lines for clarity
                ax3.plot([cluster[j, 0], cluster[j, 0]],
                         [cluster[j, 1], cluster[j, 1]],
                         [0, distances[j]], color=neon_colors[i % len(neon_colors)], alpha=0.1)

    # Plot centroids
    ax3.scatter([c[0] for c in kmeans_centroids],
                [c[1] for c in kmeans_centroids],
                [0] * len(kmeans_centroids), s=200, color='white', alpha=0.9,
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
    for i, cluster in enumerate(kmedoids_clusters):
        if len(cluster) > 0:
            cluster = np.array(cluster)

            # Calculate distances to respective medoid for z-axis
            distances = np.sqrt(np.sum((cluster[:, :2] - kmedoids_medoids[i][:2]) ** 2, axis=1))

            # Plot 3D scatter
            ax4.scatter(cluster[:, 0], cluster[:, 1], distances,
                        s=40, color=neon_colors[i % len(neon_colors)], alpha=0.7,
                        edgecolors='white', linewidths=0.5)

            # Plot vertical lines connecting points to their projection on xy-plane
            for j in range(min(len(distances), 20)):  # Limit to 20 lines for clarity
                ax4.plot([cluster[j, 0], cluster[j, 0]],
                         [cluster[j, 1], cluster[j, 1]],
                         [0, distances[j]], color=neon_colors[i % len(neon_colors)], alpha=0.1)

    # Plot medoids
    ax4.scatter([m[0] for m in kmedoids_medoids],
                [m[1] for m in kmedoids_medoids],
                [0] * len(kmedoids_medoids), s=180, color='white', alpha=0.9,
                marker='D', edgecolors='#FF00FF', linewidths=2)

    ax4.set_title('K-medoids 3D View (z=distance to medoid)', fontsize=14, fontweight='bold', color='white')
    ax4.set_xlabel(axis_labels[0] if len(axis_labels) > 0 else 'Feature 1', fontsize=10, color='white')
    ax4.set_ylabel(axis_labels[1] if len(axis_labels) > 1 else 'Feature 2', fontsize=10, color='white')
    ax4.set_zlabel('Distance to Medoid', fontsize=10, color='white')
    ax4.grid(True, alpha=0.15, linestyle='--')

    # Add a title for the overall figure
    plt.suptitle('Advanced Visualization of Clustering Results', fontsize=22, fontweight='bold', color='white', y=0.98)

    # Add a subtle watermark
    fig.text(0.98, 0.02, 'Custom K-means & K-medoids Implementation', fontsize=8, color='gray',
             ha='right', va='bottom', alpha=0.5, style='italic')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('advanced_clustering_visualization.png', dpi=300, bbox_inches='tight', facecolor='#0D1117')
    plt.show()


# Create time complexity visualization
def visualize_time_complexity(kmeans_time, kmedoids_time):
    # Use dark theme for modern look
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    ax.set_facecolor('#0D1117')
    fig.patch.set_facecolor('#0D1117')

    # Create data
    algorithms = ['K-means', 'K-medoids']
    times = [kmeans_time, kmedoids_time]

    # Custom neon color scheme
    neon_colors = ['#00FFFF', '#FF00FF']

    # Create gradient for bars
    from matplotlib.colors import LinearSegmentedColormap
    gradients = []
    for color in neon_colors:
        cmap = LinearSegmentedColormap.from_list('custom', ['#1A2233', color])
        gradient = np.linspace(0, 1, 100).reshape(-1, 1)
        gradients.append(cmap(gradient))

    # Create bars with gradient fill
    bars = []
    bar_width = 0.6
    for i, (time_val, color_gradient) in enumerate(zip(times, gradients)):
        # Create bar with gradient
        for j in range(99):
            height_fraction = time_val * (j + 1) / 100
            bar = ax.bar(i, height_fraction - times[i] * (j / 100),
                         bottom=times[i] * (j / 100), color=color_gradient[j][0],
                         width=bar_width, edgecolor='none', alpha=0.8)
            if j == 0:
                bars.append(bar)

    # Add reflective effect at bottom
    for i, time_val in enumerate(times):
        # Create reflection
        reflection = ax.bar(i, time_val * 0.2, bottom=-time_val * 0.2,
                            alpha=0.2, width=bar_width, color=neon_colors[i], edgecolor='none')
        # Add vertical line for connectivity
        ax.plot([i - bar_width / 2, i - bar_width / 2], [0, time_val], color='white', alpha=0.1, linewidth=1)
        ax.plot([i + bar_width / 2, i + bar_width / 2], [0, time_val], color='white', alpha=0.1, linewidth=1)

    # Add glow effects around bars
    for i, (time_val, neon_color) in enumerate(zip(times, neon_colors)):
        for size, alpha in zip([bar_width * 1.2, bar_width * 1.1], [0.05, 0.1]):
            ax.bar(i, time_val, width=size, color=neon_color, alpha=alpha, edgecolor='none')

    # Add pulsing effect indicator at top of each bar
    for i, (time_val, neon_color) in enumerate(zip(times, neon_colors)):
        circle = plt.Circle((i, time_val * 1.05), 0.05, color=neon_color, alpha=0.8)
        ax.add_patch(circle)

    # Add value labels with modern styling
    for i, time_val in enumerate(times):
        ax.text(i, time_val + max(times) * 0.05, f'{time_val:.4f}s',
                ha='center', va='bottom', fontsize=12,
                color=neon_colors[i], fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8,
                          edgecolor=neon_colors[i], linewidth=2))

    # Calculate speedup ratio
    speedup = kmedoids_time / kmeans_time if kmeans_time > 0 else 0

    # Add comparison connection with animated styling
    if speedup > 1:
        # Create connecting arc between implementations
        arc_x = np.linspace(0, 1, 100)
        arc_y = 0.3 * np.sin(np.pi * arc_x) + max(times) * 1.1
        ax.plot(arc_x, arc_y, '--', color='#FFFFFF', alpha=0.7, linewidth=1.5)

        # Add speedup annotation
        ax.text(0.5, max(times) * 1.1 + 0.3,
                f"K-medoids is\n{speedup:.2f}x slower", color='white',
                ha='center', va='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.8,
                          edgecolor='#FFFFFF', linewidth=2))

    # Set axis labels and title with neon glow effect
    ax.set_ylabel('Execution Time (seconds)', fontsize=14, color='white', fontweight='bold')
    ax.set_xticks(range(len(algorithms)))
    ax.set_xticklabels(algorithms, fontsize=14, color='white', fontweight='bold')

    # Create glowing title effect
    title_text = 'Algorithm Performance Comparison'
    subtitle_text = 'Execution Time Analysis'

    # Main title with shadow effect
    for offset in [0.5, 0.3, 0.1]:
        ax.text(0.5, max(times) * 1.4 + offset / 10, title_text,
                color='white', alpha=offset,
                fontsize=22, fontweight='bold', ha='center')
    ax.text(0.5, max(times) * 1.4, title_text,
            color='white', fontsize=22, fontweight='bold', ha='center')

    # Subtitle
    ax.text(0.5, max(times) * 1.3, subtitle_text,
            color='gray', fontsize=14, fontstyle='italic', ha='center')

    # Add explanatory note
    ax.text(0.5, -max(times) * 0.3,
            "Note: Shorter bars indicate better performance (faster execution)",
            color='gray', fontstyle='italic', fontsize=10, ha='center')

    # Set custom grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.2)

    # Add radial background effect
    circle = plt.Circle((0.5, max(times) / 2), max(times) * 2,
                        color='#1A2233', alpha=0.3,
                        transform=ax.transData)
    ax.add_patch(circle)

    # Set y-axis limits to include annotations and reflections
    ax.set_ylim(-max(times) * 0.35, max(times) * 1.5)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.text(0.98, 0.02, f"Generated: {timestamp}",
             fontsize=8, color='gray', ha='right', alpha=0.5)

    plt.tight_layout()
    plt.savefig('time_complexity_comparison.png', dpi=300, bbox_inches='tight', facecolor='#0D1117')
    plt.show()


# Create SSE comparison visualization
def visualize_sse_comparison(kmeans_history, kmedoids_history):
    # Use dark theme for modern look
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111)
    ax.set_facecolor('#0D1117')
    fig.patch.set_facecolor('#0D1117')

    # Custom color scheme for the hexagonal markers
    neon_colors = ['#00FFFF', '#FF00FF']

    # Extract SSE values for comparison
    kmeans_sse = kmeans_history[-1]['total_sse']
    kmedoids_sse = kmedoids_history[-1]['total_sse']

    # Data preparation
    algorithms = ['K-means', 'K-medoids']
    sse_values = [kmeans_sse, kmedoids_sse]

    # Create hexagonal markers instead of bars for futuristic look
    for i, (sse_val, color) in enumerate(zip(sse_values, neon_colors)):
        # Create hexagon shape
        from matplotlib.path import Path
        import matplotlib.patches as patches

        # Size proportional to SSE value
        size_factor = 0.3 * sse_val / max(sse_values)

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
        ax.text(i, -size_factor - 0.1, algorithms[i], color='white', ha='center', va='top',
                fontsize=14, fontweight='bold')

    # Add comparison annotation
    sse_diff = abs(kmeans_sse - kmedoids_sse)
    sse_percent = (sse_diff / min(kmeans_sse, kmedoids_sse)) * 100

    # Draw connecting line between algorithms
    ax.plot([0, 1], [0, 0], '--', color='#FFFFFF', alpha=0.5, linewidth=1.5)

    # Add percentage difference with futuristic callout
    better_algo = "K-means" if kmeans_sse < kmedoids_sse else "K-medoids"
    diff_text = f"{sse_percent:.1f}% difference\n{better_algo} has better cohesion"

    # Create callout box with glowing effect
    ax.text(0.5, max(sse_values) * 0.4, diff_text,
            ha='center', va='center', color='white',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.7,
                      edgecolor='#FFFF00', linewidth=1.5))

    # Create glowing title effect
    title_text = 'Cluster Cohesion Comparison'
    subtitle_text = 'Sum of Squared Errors (SSE/WCSS)'

    # Main title with shadow effect
    for offset in [0.5, 0.3, 0.1]:
        ax.text(0.5, max(sse_values) * 0.8 + offset / 5, title_text,
                color='white', alpha=offset,
                fontsize=22, fontweight='bold', ha='center')
    ax.text(0.5, max(sse_values) * 0.8, title_text,
            color='white', fontsize=22, fontweight='bold', ha='center')

    # Subtitle
    ax.text(0.5, max(sse_values) * 0.7, subtitle_text,
            color='gray', fontsize=14, fontstyle='italic', ha='center')

    # Add explanatory note
    ax.text(0.5, -max(sse_values) * 0.6,
            "Note: Lower SSE/WCSS values indicate tighter, more cohesive clusters",
            color='gray', fontstyle='italic', fontsize=10, ha='center')

    # Add radial background for futuristic feel
    theta = np.linspace(0, 2 * np.pi, 100)
    for radius in np.linspace(0.1, 0.5, 5):
        circle_x = 0.5 + radius * max(sse_values) * np.cos(theta)
        circle_y = radius * max(sse_values) * np.sin(theta)
        ax.plot(circle_x, circle_y, color='white', alpha=0.05, linewidth=0.5)

    # Add visual indicators of which algorithm is better
    if kmeans_sse < kmedoids_sse:
        ax.text(0, -max(sse_values) * 0.5, "✓", color='#33FF33', fontsize=24,
                ha='center', va='center', fontweight='bold')
    else:
        ax.text(1, -max(sse_values) * 0.5, "✓", color='#33FF33', fontsize=24,
                ha='center', va='center', fontweight='bold')

    # Clean up axes
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-max(sse_values) * 0.7, max(sse_values) * 1.0)
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
    plt.savefig('sse_comparison.png', dpi=300, bbox_inches='tight', facecolor='#0D1117')
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
    visualize_sse_comparison(kmeans_history, kmedoids_history)


if __name__ == "__main__":
    main()