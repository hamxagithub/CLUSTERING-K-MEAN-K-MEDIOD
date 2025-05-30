# K-means vs K-medoid Clustering: A Comparative Analysis

This repository implements K-means and K-medoid clustering algorithms from scratch, performing a comparative analysis on a given dataset. The objective is to examine the performance of these two clustering techniques and identify the optimal number of clusters (K) through various metrics, such as SSE/WCSS values, cluster sizes, and time complexity.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Algorithm Details](#algorithm-details)
  - [K-means Clustering](#k-means-clustering)
  - [K-medoid Clustering](#k-medoid-clustering)
- [Instructions](#instructions)
  - [Step 1: Elbow Chart](#step-1-elbow-chart)
  - [Step 2: K-means and K-medoid Execution](#step-2-k-means-and-k-medoid-execution)
  - [Step 3: Time Complexity Analysis](#step-3-time-complexity-analysis)
- [Results](#results)
  - [Comparative Analysis](#comparative-analysis)
- [Dependencies](#dependencies)
- [License](#license)

## Introduction

The primary goal of this repository is to implement and evaluate K-means and K-medoid clustering algorithms. Both methods aim to group data into clusters, but they differ in how they represent cluster centers. K-means uses the centroid (mean) of the points, while K-medoid uses the most centrally located point as the representative for each cluster.

We will compare both algorithms by evaluating the following metrics:

- **Optimal K value** using the elbow method.
- **Cluster characteristics** such as the size, SSE (sum of squared errors) or WCSS (within-cluster sum of squares), and overall SSE/WCSS.
- **Time complexity** for both methods to assess their efficiency.

## Features

- **K-means Clustering**: An iterative algorithm that assigns each point to the nearest cluster centroid, followed by recomputing the centroids based on the current assignment.
- **K-medoid Clustering**: A variation of K-means where the centroid is replaced by the most centrally located point (medoid) of each cluster.
- **Elbow Method**: A method to determine the optimal number of clusters by plotting the SSE/WCSS values for different values of K.
- **Visualization**: Graphical representation of clustering results, including the elbow chart and final cluster distributions.

## Algorithm Details

### K-means Clustering

K-means is an unsupervised clustering algorithm that aims to partition the dataset into K clusters. It works by:

1. Initializing K centroids randomly.
2. Assigning each data point to the nearest centroid.
3. Recalculating the centroids based on the mean of the points in each cluster.
4. Repeating steps 2 and 3 until convergence (i.e., when centroids no longer change).

### K-medoid Clustering

K-medoid is a variant of K-means that uses actual data points as the center of the clusters. Instead of computing the mean, K-medoid selects the point that minimizes the total dissimilarity (distance) to other points in the cluster.

### Step-by-step Instructions

#### Step 1: Elbow Chart

1. Implement the K-means algorithm to calculate SSE/WCSS for different values of K (from 1 to 10).
2. Plot an elbow chart to visualize how the SSE/WCSS values change with increasing K.
3. Identify the optimal value of K by finding the "elbow" point in the plot (where the reduction in SSE/WCSS slows down).

#### Step 2: K-means and K-medoid Execution

1. Run both K-means and K-medoid algorithms on the dataset using the optimal value of K identified in Step 1.
2. For each algorithm, display the following results:
   - **Iteration number**: Track the number of iterations it took to converge.
   - **Size of each cluster**: Display the number of points assigned to each cluster.
   - **SSE/WCSS for each cluster**: Display the sum of squared errors (SSE) or within-cluster sum of squares (WCSS) for each cluster.
   - **Overall SSE/WCSS**: Show the total SSE/WCSS for all clusters combined.

#### Step 3: Time Complexity Analysis

1. Measure the time taken by both algorithms to complete the clustering task.
2. Display the time complexity for both K-means and K-medoid in seconds.

## Results

### Comparative Analysis

After running both clustering algorithms, the following comparisons will be drawn:

- **Clustering Quality**: Which algorithm produced more cohesive and well-separated clusters?
- **Time Efficiency**: How long did each algorithm take to complete the clustering process? This is important for large datasets.
- **SSE/WCSS Analysis**: Which algorithm resulted in the lowest SSE/WCSS? Did K-means or K-medoid provide a better fit for the dataset?
- **Overall Winner**: Based on the metrics above, we will determine which clustering method performed better for this dataset.

### Conclusion

Through this implementation, you will gain a deep understanding of the K-means and K-medoid algorithms and how they compare in terms of accuracy, efficiency, and scalability. The comparison will allow us to draw conclusions about which method works better under different conditions.

## Dependencies

The following libraries are required to run the code:

- `numpy` for numerical operations.
- `pandas` for data manipulation.
- `matplotlib` for plotting the elbow chart and visualizing the results.

You can install the required dependencies using:

```bash
pip install numpy pandas matplotlib
