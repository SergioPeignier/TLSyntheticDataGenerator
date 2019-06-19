"""
Changes
-------
This file contains functions to apply changes in dataset generators, in order to
simulate the transition between a source to a target dataset

Attributes:

"""
import pandas as pd
import numpy as np
from Generator import Cluster

__author__ = "Sergio Peignier, Mounir Atiq"
__copyright__ = ""
__credits__ = ["Ludovic Minvielle", "Mathilde Mougeot", "Nicolas Vayatis"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "Sergio Peignier, Mounir Atiq"
__email__ = "sergio.peignier@insa-lyon.fr, atiq.mounir@gmail.com"
__status__ = "pre-alpha"


def apply_drift(clusters, cluster_feature_speed, min_coordinate, max_coordinate):
	"""
	Apply a drift to the dataset clusters

	Args:
		clusters (list): list of Cluster objects
		cluster_feature_speed (dict): dict keys are cluster index and values
		are numpy.array representing the "speed" of the cluster centroid along
		each dimension.
		min_coordinate (float): minimal coordinate (keep the points in a
		hyper-cube)
		max_coordinate (float): maximal coordinate (keep the points in a
		hyper-cube)
	"""
    def collision(cluster, min_coordinate, max_coordinate):
        cluster.centroid[cluster.centroid >= max_coordinate] = max_coordinate
        cluster.centroid[cluster.centroid <= min_coordinate] = min_coordinate
    for cluster_id, feature_speed in cluster_feature_speed.items():
        for feature, speed in feature_speed.items():
            clusters[cluster_id].centroid[feature] += speed
        collision(clusters[cluster_id], min_coordinate, max_coordinate)

def apply_density_change(clusters, cluster_feature_std):
	"""
	Apply a std change to the gaussian features of the clusters

	Args:
		clusters (list): list of Cluster objects
		cluster_feature_std (dict): dict keys are cluster index and values are
		dicts with feature ids as keys and new stds as values
	"""
    for cluster_id, feature_std in cluster_feature_std.items():
        for feature, std in feature_std.items():
            clusters[cluster_id].radii[feature] = std

def create_new_clusters(clusters, list_parameters_cluster_creation):
	"""
	Create new clusters from a list or argument dicts.

	Args:
		clusters (list): list of Cluster objects (new clusters are appended)
		list_parameters_cluster_creation (list): list of dicts s.t. keys
		represent the argument names for the Cluster object creation and values
		represent the argument values
	"""
    for parameter in list_parameters_cluster_creation:
        new_cluster = Cluster(**parameter)
        clusters.append(new_cluster)

def delete_clusters(clusters, clusters_id):
	"""
	Delete clusters

	Args:
		clusters (list): list of Cluster objects (new clusters are appended)
		clusters_id (list): list of clusters ids to pop
	"""
    for cluster_id in clusters_id:
        clusters.pop(cluster_id)

def change_cluster_weight(clusters, clusters_weights):
	"""
	Change clusters weights

	Args:
		clusters (list): list of Cluster objects
		clusters_weights (dict): dict keys are cluster index and values are
		the new clusters weights
	"""
    for cluster_id, weight in clusters_weights.items():
        clusters[cluster_id].weight = weight

def loose_feature(clusters, clusters_features):
	"""
	Turn coordinate distribution along specific features to uniform distributions

	Args:
		clusters (list): list of Cluster objects
		clusters_features (dict): dict keys are cluster index and values are
		the features that should follow a uniform distribution from now on
	"""
    for cluster_id, features in clusters_features.items():
        for feature in features:
            if clusters[cluster_id].projected_dims[feature]:
                clusters[cluster_id].projected_dims[feature] = 0
                clusters[cluster_id].nb_projected_dims -= 1
                clusters[cluster_id].nb_not_projected_dims += 1

def gain_feature(clusters, clusters_features):
	"""
	Turn coordinate distribution along specific features to gaussian distributions

	Args:
		clusters (list): list of Cluster objects
		clusters_features (dict): dict keys are cluster index and values are
		the features that should follow a gaussian distribution from now on
	"""
    for cluster_id, features in clusters_features.items():
        for feature in features:
            if not clusters[cluster_id].projected_dims[feature]:
                clusters[cluster_id].projected_dims[feature] = 1
                clusters[cluster_id].nb_not_projected_dims -= 1
                clusters[cluster_id].nb_projected_dims += 1
