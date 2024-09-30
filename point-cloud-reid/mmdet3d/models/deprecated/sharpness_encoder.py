import torch
import numpy as np
from torch_cluster import knn_graph

def calculate_distances(point_cloud):
    if point_cloud.size(0) < 2:
        return None, None  # Not enough points to calculate distances

    # Expand point cloud to calculate pairwise differences
    point_cloud_expanded = point_cloud.unsqueeze(1)  # Shape: [N, 1, 3]
    differences = point_cloud_expanded - point_cloud  # Broadcasting to get pairwise differences
    distances = torch.sqrt((differences ** 2).sum(dim=2))  # Euclidean distances

    # Mask the diagonal of the distance matrix to exclude zero distances (distance to itself)
    mask = torch.eye(distances.size(0), dtype=torch.bool, device=point_cloud.device)
    distances = distances.masked_fill(mask, float('inf'))

    # Flatten the upper triangle of the distance matrix to get all unique pairs
    distances = distances.triu(1)
    distances_list = distances[distances != float('inf')].tolist()  # Convert to list excluding 'inf'
    distances_list = distances[distances > float(0.0)].tolist()  # Convert to list excluding '0.0'

    min_distance = min(distances_list)
    max_distance = max(distances_list)

    return distances_list, min_distance, max_distance

def compute_covariance_matrix(points):
    """ Compute the covariance matrix for a given set of points, with regularization to improve numerical stability. """
    if points.size(0) < 2:
        return torch.zeros((points.size(1), points.size(1)), dtype=points.dtype, device=points.device)
    
    centroid = points.mean(dim=0)
    centered_points = points - centroid
    covariance_matrix = centered_points.T @ centered_points / (centered_points.shape[0] - 1)
    
    # Regularization to avoid singular matrix issues, especially with small neighborhoods
    regularization_term = 1e-5 * torch.eye(centered_points.size(1), device=points.device)
    covariance_matrix += regularization_term
    
    return covariance_matrix

def eigenvalue_analysis(points):
    """ Perform eigenvalue analysis to categorize the point, handling cases with insufficient data. """
    if points.size(0) < 3:
        # Not enough points to form a plane in 3D, return zero eigenvalues
        return torch.zeros(3, device=points.device)
    
    covariance_matrix = compute_covariance_matrix(points)
    # eigenvalues = torch.linalg.eigvals(covariance_matrix)
    eigenvalues, _ = torch.linalg.eigh(covariance_matrix)
    if torch.isnan(eigenvalues).any():
        # Handle potential numerical issues by setting nan eigenvalues to zero
        eigenvalues[torch.isnan(eigenvalues)] = 0
    return eigenvalues

def radius_neighbors(point_cloud, radius):
    distances = torch.cdist(point_cloud, point_cloud)
    distances = distances.fill_diagonal_(float('inf'))
    neighbors_mask = distances < radius
    return neighbors_mask

def encode_sharpness_radius(point_cloud, radius=0.5):
    """ Encode a single point cloud by analyzing local geometric features based on a distance threshold. """
    device = point_cloud.device
    neighbors_mask = radius_neighbors(point_cloud, radius)

    features = []
    for i in range(point_cloud.size(0)):
        # print("i: ", i)
        # print(torch.tensor(range(0,point_cloud.size(0)))[neighbors_mask[i]])
        neighbors = point_cloud[neighbors_mask[i]]

        if neighbors.size(0) < 3:
          feature_tensor = torch.zeros(3, device=device)
        else:
          neighbors = neighbors.cpu().numpy()
          cov = np.cov(neighbors.T)
          feature_tensor, _ = np.linalg.eig(cov)
          feature_tensor = torch.tensor(feature_tensor).to(device)
        
        # print("feature_tensor: ", feature_tensor)
        features.append(feature_tensor)

    feature_tensor = torch.stack(features)
        # print("feature_tensor: ", feature_tensor)

    # Normalize and scale feature tensor
    feature_tensor = torch.log1p(feature_tensor)
    # print("feature_tensor log1p: ", feature_tensor)

    feature_tensor = (feature_tensor - feature_tensor.mean()) / feature_tensor.std()
    # print("feature_tensor final: ", feature_tensor)

    return feature_tensor

def safe_normalize(tensor):
    mean = tensor.mean()
    std = tensor.std()
    epsilon = 1e-8  # A small constant to prevent division by zero
    if std == 0:
        return tensor - mean
    return (tensor - mean) / (std + epsilon)

def encode_sharpness(point_cloud, radius=0.5):
    """ Encode a single point cloud by analyzing local geometric features based on a distance threshold. """
    neighbors_mask = radius_neighbors(point_cloud, radius)

    features = []
    for i in range(point_cloud.size(0)):
        neighbors = point_cloud[neighbors_mask[i]]
        eigenvalues = eigenvalue_analysis(neighbors)
        # eigenvalues = torch.linalg.eigvals(neighbors)
        features.append(eigenvalues)
        # print(f"{i}-th batch: {eigenvalues}")

    feature_tensor = torch.stack(features)
    # print("1", feature_tensor)

    # Normalize and scale feature tensor
    feature_tensor = torch.log1p(feature_tensor)
    # print("2", feature_tensor)
    
    feature_tensor = safe_normalize(feature_tensor)
    # print("3", feature_tensor)

    # feature_tensor = (feature_tensor - feature_tensor.mean()) / feature_tensor.std()

    return feature_tensor


# def encode_sharpness_knn(point_cloud, k=20):
#     group_idx = knn_point(point_cloud, point_cloud)
#     # group_idx (B, N, k(indices))

#     group_points = torch.tensor([point_cloud[i] for i in range(point_cloud.shape[0])])
#     # group_points = (B, k, C)

#     eigenvalues = torch.linalg.eigvalsh(group_points)
#     # eigenvalues = (B, 3, 1)

#     print("eig: ", eigenvalues.shape)
