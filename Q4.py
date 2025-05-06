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
    # Add scatter plot with jitter for better visualization
    plt.scatter(data[:, 0], data[:, 1], c=kmeans_results['labels'], cmap='viridis',
                alpha=0.6, s=50, edgecolor='w')

    plt.scatter(kmeans_results['centers'][:, 0], kmeans_results['centers'][:, 1],
                marker='*', s=300, c='red', edgecolor='k', label='Centroids')

    plt.title('scikit-learn K-means Clustering', fontsize=14)
    plt.xlabel(axis_labels[0] if len(axis_labels) > 0 else 'Feature 1', fontsize=12)
    plt.ylabel(axis_labels[1] if len(axis_labels) > 1 else 'Feature 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # K-medoids visualization
    plt.subplot(1, 2, 2)
    plt.scatter(data[:, 0], data[:, 1], c=kmedoids_results['labels'], cmap='viridis',
                alpha=0.6, s=50, edgecolor='w')

    plt.scatter(kmedoids_results['centers'][:, 0], kmedoids_results['centers'][:, 1],
                marker='*', s=300, c='red', edgecolor='k', label='Medoids')

    plt.title('scikit-learn K-medoids Clustering', fontsize=14)
    plt.xlabel(axis_labels[0] if len(axis_labels) > 0 else 'Feature 1', fontsize=12)
    plt.ylabel(axis_labels[1] if len(axis_labels) > 1 else 'Feature 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig('sklearn_clustering_results.png', dpi=300)
    plt.show()


# Create visualization for time complexity comparison
def visualize_time_complexity(q3_results, q4_results):
    plt.figure(figsize=(10, 6))
    methods = ['K-means\nCustom', 'K-means\nsklearn', 'K-medoids\nCustom', 'K-medoids\nsklearn']
    times = [
        q3_results['kmeans_time'],
        q4_results['kmeans']['time'],
        q3_results['kmedoids_time'],
        q4_results['kmedoids']['time']
    ]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    bars = plt.bar(methods, times, color=colors)

    # Add time values on bars
    for bar, time_value in zip(bars, times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                 f'{time_value:.4f}s',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.title('Time Complexity Comparison: Custom vs sklearn', fontsize=14)
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('sklearn_vs_custom_time.png', dpi=300)
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