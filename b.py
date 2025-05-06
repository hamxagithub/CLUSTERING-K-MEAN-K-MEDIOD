import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

# Set a global variable to track KMedoids availability
KMEDOIDS_AVAILABLE = False

# Try to import KMedoids, if not available we'll use our fallback implementation
try:
    from sklearn.cluster import KMedoids

    KMEDOIDS_AVAILABLE = True
    print("Using sklearn's KMedoids implementation")
except ImportError:
    try:
        # Try to import from sklearn_extra which sometimes contains KMedoids
        from sklearn_extra.cluster import KMedoids

        KMEDOIDS_AVAILABLE = True
        print("Using sklearn_extra's KMedoids implementation")
    except ImportError:
        print("KMedoids not available in scikit-learn or sklearn_extra. Will use fallback implementation.")
        KMEDOIDS_AVAILABLE = False


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


# Run K-medoids clustering using scikit-learn
def run_kmedoids_sklearn(data, k, random_state=42):
    global KMEDOIDS_AVAILABLE  # Explicitly use the global variable

    print(f"Running scikit-learn KMedoids with k={k}...")
    start_time = time.time()

    if KMEDOIDS_AVAILABLE:
        try:
            # Initialize and fit KMedoids - use same settings as in Q3 as much as possible
            kmedoids = KMedoids(
                n_clusters=k,
                metric='euclidean',
                method='alternate',  # faster method
                init='random',  # random initialization like in Q3
                max_iter=100,  # Match max iterations in Q3
                random_state=random_state
            )

            labels = kmedoids.fit_predict(data)
            medoid_indices = kmedoids.medoid_indices_
            centers = data[medoid_indices]
            iterations = kmedoids.n_iter_

            execution_time = time.time() - start_time

            # Calculate SSE/WCSS
            total_sse, cluster_sse = calculate_sse(data, labels, centers)

            # Count cluster sizes
            cluster_sizes = [np.sum(labels == i) for i in range(k)]

            return {
                'labels': labels,
                'centers': centers,
                'medoid_indices': medoid_indices,
                'iterations': iterations,
                'time': execution_time,
                'total_sse': total_sse,
                'cluster_sse': cluster_sse,
                'cluster_sizes': cluster_sizes
            }
        except Exception as e:
            print(f"Error running KMedoids: {e}")
            print("Falling back to custom implementation...")
            # Don't modify the global variable here

    # If KMedoids is not available or failed, use custom implementation
    print("Using custom implementation for KMedoids using scikit-learn components...")

    # For the PAM (Partitioning Around Medoids) algorithm
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


# Create visualization for cluster comparison
def create_comparison_visualization(data, q3_results, q4_results, used_columns, variable_info):
    # Set up names for axis labels based on Information sheet
    axis_labels = []
    for col in used_columns:
        if variable_info and col in variable_info:
            axis_labels.append(f"{col}: {variable_info[col]}")
        else:
            axis_labels.append(col)

    # Create figure for comparison
    plt.figure(figsize=(12, 10))

    # K-means comparison: Custom vs sklearn
    plt.subplot(2, 1, 1)
    plt.scatter(data[:, 0], data[:, 1], c=q3_results['kmeans_labels'], cmap='viridis',
                alpha=0.5, s=50, edgecolors='w', label='Custom K-means')
    plt.scatter(q3_results['kmeans_centers'][:, 0], q3_results['kmeans_centers'][:, 1],
                marker='*', s=300, c='red', label='Custom Centroids')

    plt.scatter(data[:, 0] + 0.03, data[:, 1] + 0.03, c=q4_results['kmeans']['labels'], cmap='plasma',
                alpha=0.5, s=30, edgecolors='k', label='sklearn K-means')
    plt.scatter(q4_results['kmeans']['centers'][:, 0], q4_results['kmeans']['centers'][:, 1],
                marker='X', s=200, c='black', label='sklearn Centroids')

    plt.title('K-means Comparison: Custom vs sklearn', fontsize=14)
    plt.xlabel(axis_labels[0] if len(axis_labels) > 0 else 'Feature 1')
    plt.ylabel(axis_labels[1] if len(axis_labels) > 1 else 'Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # K-medoids comparison: Custom vs sklearn
    plt.subplot(2, 1, 2)
    plt.scatter(data[:, 0], data[:, 1], c=q3_results['kmedoids_labels'], cmap='viridis',
                alpha=0.5, s=50, edgecolors='w', label='Custom K-medoids')
    plt.scatter(q3_results['kmedoids_centers'][:, 0], q3_results['kmedoids_centers'][:, 1],
                marker='*', s=300, c='red', label='Custom Medoids')

    plt.scatter(data[:, 0] + 0.03, data[:, 1] + 0.03, c=q4_results['kmedoids']['labels'], cmap='plasma',
                alpha=0.5, s=30, edgecolors='k', label='sklearn K-medoids')
    plt.scatter(q4_results['kmedoids']['centers'][:, 0], q4_results['kmedoids']['centers'][:, 1],
                marker='X', s=200, c='black', label='sklearn Medoids')

    plt.title('K-medoids Comparison: Custom vs sklearn', fontsize=14)
    plt.xlabel(axis_labels[0] if len(axis_labels) > 0 else 'Feature 1')
    plt.ylabel(axis_labels[1] if len(axis_labels) > 1 else 'Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sklearn_vs_custom_comparison.png', dpi=300)
    plt.show()

    # Create time comparison bar chart
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

    # Create SSE comparison bar chart
    plt.figure(figsize=(10, 6))
    methods = ['K-means\nCustom', 'K-means\nsklearn', 'K-medoids\nCustom', 'K-medoids\nsklearn']
    sses = [
        q3_results['kmeans_sse'],
        q4_results['kmeans']['total_sse'],
        q3_results['kmedoids_sse'],
        q4_results['kmedoids']['total_sse']
    ]

    bars = plt.bar(methods, sses, color=colors)

    # Add SSE values on bars
    for bar, sse_value in zip(bars, sses):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
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
        return None, None, None

    # Create elbow chart and find optimal K (re-using from Q3)
    k_range = range(1, 11)
    sse_values = Q3.create_elbow_chart(data, k_range)

    # Find the optimal K as in Q3
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

    # Display results for sklearn KMeans
    print("\n" + "=" * 50)
    print("SKLEARN K-MEANS RESULTS")
    print("=" * 50)
    print(f"Total iterations: {kmeans_results['iterations']}")
    print(f"Cluster sizes: {kmeans_results['cluster_sizes']}")
    print(f"Cluster SSE: {[round(sse, 2) for sse in kmeans_results['cluster_sse']]}")
    print(f"Total SSE/WCSS: {round(kmeans_results['total_sse'], 2)}")
    print(f"Time complexity: {kmeans_results['time']:.4f} seconds")

    # Display results for sklearn KMedoids
    print("\n" + "=" * 50)
    print("SKLEARN K-MEDOIDS RESULTS")
    print("=" * 50)
    print(f"Total iterations: {kmedoids_results['iterations']}")
    print(f"Cluster sizes: {kmedoids_results['cluster_sizes']}")
    print(f"Cluster SSE: {[round(sse, 2) for sse in kmedoids_results['cluster_sse']]}")
    print(f"Total SSE/WCSS: {round(kmedoids_results['total_sse'], 2)}")
    print(f"Time complexity: {kmedoids_results['time']:.4f} seconds")

    # Store sklearn results
    q4_results = {
        'kmeans': kmeans_results,
        'kmedoids': kmedoids_results
    }

    # Create comparison visualizations
    create_comparison_visualization(data, q3_results, q4_results, used_columns, variable_info)

    # Perform comparative analysis
    print("\n" + "=" * 50)
    print("COMPARATIVE ANALYSIS: CUSTOM vs SKLEARN IMPLEMENTATIONS")
    print("=" * 50)

    # K-means comparison
    print("\nK-means Comparison:")
    print(f"Iterations: Custom={q3_results['kmeans_iterations']}, sklearn={kmeans_results['iterations']}")
    print(f"Cluster sizes: Custom={q3_results['kmeans_cluster_sizes']}, sklearn={kmeans_results['cluster_sizes']}")
    print(
        f"Total SSE/WCSS: Custom={round(q3_results['kmeans_sse'], 2)}, sklearn={round(kmeans_results['total_sse'], 2)}")
    print(f"Time complexity: Custom={q3_results['kmeans_time']:.4f}s, sklearn={kmeans_results['time']:.4f}s")

    # Speed comparison
    kmeans_speed_ratio = q3_results['kmeans_time'] / kmeans_results['time'] if kmeans_results['time'] > 0 else 0
    if kmeans_speed_ratio > 1:
        print(f"sklearn's K-means is {kmeans_speed_ratio:.2f}x faster than custom implementation")
    else:
        print(f"Custom K-means is {1 / kmeans_speed_ratio:.2f}x faster than sklearn implementation")

    # SSE difference percentage for K-means
    kmeans_sse_diff_pct = abs(q3_results['kmeans_sse'] - kmeans_results['total_sse']) / q3_results['kmeans_sse'] * 100
    print(f"SSE difference: {kmeans_sse_diff_pct:.2f}%")

    # K-medoids comparison
    print("\nK-medoids Comparison:")
    print(f"Iterations: Custom={q3_results['kmedoids_iterations']}, sklearn={kmedoids_results['iterations']}")
    print(f"Cluster sizes: Custom={q3_results['kmedoids_cluster_sizes']}, sklearn={kmedoids_results['cluster_sizes']}")
    print(
        f"Total SSE/WCSS: Custom={round(q3_results['kmedoids_sse'], 2)}, sklearn={round(kmedoids_results['total_sse'], 2)}")
    print(f"Time complexity: Custom={q3_results['kmedoids_time']:.4f}s, sklearn={kmedoids_results['time']:.4f}s")

    # Speed comparison
    kmedoids_speed_ratio = q3_results['kmedoids_time'] / kmedoids_results['time'] if kmedoids_results['time'] > 0 else 0
    if kmedoids_speed_ratio > 1:
        print(f"sklearn's K-medoids is {kmedoids_speed_ratio:.2f}x faster than custom implementation")
    else:
        print(f"Custom K-medoids is {1 / kmedoids_speed_ratio:.2f}x faster than sklearn implementation")

    # SSE difference percentage for K-medoids
    kmedoids_sse_diff_pct = abs(q3_results['kmedoids_sse'] - kmedoids_results['total_sse']) / q3_results[
        'kmedoids_sse'] * 100
    print(f"SSE difference: {kmedoids_sse_diff_pct:.2f}%")

    # Overall Analysis and Reasons for Differences
    print("\nReasons for differences in results (if any):")
    print("1. Implementation details: scikit-learn may use optimized algorithms and data structures")
    print("2. Initialization strategy: Different methods for selecting initial centroids/medoids")
    print("3. Convergence criteria: Different tolerances or methods to determine convergence")
    print("4. Distance calculations: Potential differences in distance metric implementations")
    print("5. Vectorization: scikit-learn likely uses vectorized operations for better performance")
    print("6. Optimizations: scikit-learn may include additional optimizations for large datasets")

    print("\nOverall conclusions:")
    if kmeans_speed_ratio > 1 and kmedoids_speed_ratio > 1:
        print("scikit-learn implementations are generally faster due to optimized C/C++ code")

    if kmeans_sse_diff_pct < 5 and kmedoids_sse_diff_pct < 5:
        print("Both implementations produce very similar clustering results (SSE difference < 5%)")
    elif kmeans_sse_diff_pct < 10 and kmedoids_sse_diff_pct < 10:
        print("Both implementations produce reasonably similar clustering results (SSE difference < 10%)")
    else:
        print("There are notable differences in clustering results, likely due to implementation details")


if __name__ == "__main__":
    main()