import numpy as np
from kernels.kernel_points import load_kernels as create_kernel_points
import torch


def unary_convolution(features,
                      K_values):
    """
    Simple unary convolution in tensorflow. Equivalent to matrix multiplication (space projection) for each features
    :param features: float32[n_points, in_fdim] - input features
    :param K_values: float32[in_fdim, out_fdim] - weights of the kernel
    :return: output_features float32[n_points, out_fdim]
    """

    return torch.matmul(features, K_values)


def radius_gaussian(sq_r, sig, eps=1e-9):
    """
    Compute a radius gaussian (gaussian of distance)
    :param sq_r: input radiuses [dn, ..., d1, d0]
    :param sig: extents of gaussians [d1, d0] or [d0] or float
    :return: gaussian of sq_r [dn, ..., d1, d0]
    """
    return torch.exp(-sq_r / (2 * torch.pow(sig, 2) + eps))


def KPConv(query_points,
           support_points,
           neighbors_indices,
           features,
           K_values,
           fixed='center',
           KP_extent=1.0,
           KP_influence='linear',
           aggregation_mode='sum'):
    """
    This function initiates the kernel point disposition before building KPConv graph ops
    :param query_points: float32[n_points, dim] - input query points (center of neighborhoods)
    :param support_points: float32[n0_points, dim] - input support points (from which neighbors are taken)
    :param neighbors_indices: int32[n_points, n_neighbors] - indices of neighbors of each point
    :param features: float32[n_points, in_fdim] - input features
    :param K_values: float32[n_kpoints, in_fdim, out_fdim] - weights of the kernel
    :param fixed: string in ('none', 'center' or 'verticals') - fix position of certain kernel points
    :param KP_extent: float32 - influence radius of each kernel point
    :param KP_influence: string in ('constant', 'linear', 'gaussian') - influence function of the kernel points
    :param aggregation_mode: string in ('closest', 'sum') - whether to sum influences, or only keep the closest
    :return: output_features float32[n_points, out_fdim]
    """

    # Initial kernel extent for this layer
    K_radius = 1.5 * KP_extent

    # Number of kernel points
    num_kpoints = int(K_values.shape[0])

    # Check point dimension (currently only 3D is supported)
    points_dim = int(query_points.shape[1])

    # Create one kernel disposition (as numpy array). Choose the KP distance to center thanks to the KP extent
    K_points_numpy = create_kernel_points(K_radius,
                                          num_kpoints,
                                          num_kernels=1,
                                          dimension=points_dim,
                                          fixed=fixed)
    K_points_numpy = K_points_numpy.reshape((num_kpoints, points_dim))

    # Create the tensorflow variable
    K_points = torch.from_numpy(K_points_numpy.astype(np.float32))
    if K_values.is_cuda:
        K_points = K_points.to(K_values.device)

    return KPConv_ops(query_points,
                      support_points,
                      neighbors_indices,
                      features,
                      K_points,
                      K_values,
                      KP_extent,
                      KP_influence,
                      aggregation_mode)


def KPConv_ops(query_points,
               support_points,
               neighbors_indices,
               features,
               K_points,
               K_values,
               KP_extent,
               KP_influence,
               aggregation_mode):
    """
    This function creates a graph of operations to define Kernel Point Convolution in tensorflow. See KPConv function
    above for a description of each parameter

    :param query_points:        [n_points, dim]
    :param support_points:      [n0_points, dim]
    :param neighbors_indices:   [n_points, n_neighbors]
    :param features:            [n_points, in_fdim]
    :param K_points:            [n_kpoints, dim]
    :param K_values:            [n_kpoints, in_fdim, out_fdim]
    :param KP_extent:           float32
    :param KP_influence:        string
    :param aggregation_mode:    string
    :return:                    [n_points, out_fdim]
    """

    # Get variables
    n_kp = int(K_points.shape[0])

    # Add a fake point in the last row for shadow neighbors
    shadow_point = torch.ones_like(support_points[:1, :]) * 1e6
    support_points = torch.cat([support_points, shadow_point], dim=0)

    # Get neighbor points [n_points, n_neighbors, dim]
    neighbors = support_points[neighbors_indices.long(), :]

    # Center every neighborhood
    neighbors = neighbors - query_points.unsqueeze(1)

    # Get all difference matrices [n_points, n_neighbors, n_kpoints, dim]
    differences = neighbors.unsqueeze(2) - K_points

    # Get the square distances [n_points, n_neighbors, n_kpoints]
    sq_distances = torch.sum(torch.mul(differences, differences), dim=3)

    # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
    if KP_influence == 'constant':
        # Every point get an influence of 1.
        all_weights = torch.ones_like(sq_distances)
        all_weights = all_weights.transpose(1, 2)

    elif KP_influence == 'linear':
        # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
        corr = 1 - torch.sqrt(sq_distances + 1e-10) / KP_extent
        all_weights = torch.max(corr, torch.zeros_like(sq_distances))
        all_weights = all_weights.transpose(1, 2)

    elif KP_influence == 'gaussian':
        # Influence in gaussian of the distance.
        sigma = KP_extent * 0.3
        all_weights = radius_gaussian(sq_distances, sigma)
        all_weights = all_weights.transpose(1, 2)
    else:
        raise ValueError('Unknown influence function type (config.KP_influence)')

    # In case of closest mode, only the closest KP can influence each point
    if aggregation_mode == 'closest':
        pass
    #     neighbors_1nn = tf.argmin(sq_distances, axis=2, output_type=tf.int32)
    #     all_weights *= tf.one_hot(neighbors_1nn, n_kp, axis=1, dtype=tf.float32)

    elif aggregation_mode != 'sum':
        raise ValueError("Unknown convolution mode. Should be 'closest' or 'sum'")

    features = torch.cat([features, torch.zeros_like(features[:1, :])], dim=0)

    # Get the features of each neighborhood [n_points, n_neighbors, in_fdim]
    neighborhood_features = features[neighbors_indices.long(), :]

    # Apply distance weights [n_points, n_kpoints, in_fdim]
    weighted_features = torch.matmul(all_weights, neighborhood_features)

    # Apply network weights [n_kpoints, n_points, out_fdim]
    weighted_features = weighted_features.transpose(0, 1)
    kernel_outputs = torch.matmul(weighted_features, K_values)

    # Convolution sum to get [n_points, out_fdim]
    output_features = torch.sum(kernel_outputs, dim=0, keepdim=False)

    # normalization term.
    # neighbor_features_sum = torch.sum(neighborhood_features, dim=-1)
    # neighbor_num = torch.sum(torch.gt(neighbor_features_sum, 0.0), dim=-1)
    # neighbor_num = torch.max(neighbor_num, torch.ones_like(neighbor_num))
    # output_features = output_features / neighbor_num.unsqueeze(1)

    return output_features
