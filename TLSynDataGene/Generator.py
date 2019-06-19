"""
Generator
---------
This file contains the classes to generate clusters of points as well as full
datasets

Attributes:

"""
import pandas as pd
import numpy as np

__author__ = "Sergio Peignier, Mounir Atiq"
__copyright__ = ""
__credits__ = ["Ludovic Minvielle", "Mathilde Mougeot", "Nicolas Vayatis"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "Sergio Peignier, Mounir Atiq"
__email__ = "sergio.peignier@insa-lyon.fr, atiq.mounir@gmail.com"
__status__ = "pre-alpha"

class Cluster:
    def __init__(self,
                 weight,
                 dimensionality,
                 min_gauss_dim,
                 max_gauss_dim,
                 min_coordinate,
                 max_coordinate,
                 min_gauss_dim_var,
                 max_gauss_dim_var,
                 class_label,
                 ):
        """
        Cluster is a class that allows to generate parametrized clusters of data
        points with coordiantes following gaussian/uniform distributions

        Args:
            weight (float): Weight of this cluster w.r.t. the full dataset.
            The probability that a new data point belongs to cluster C is simply
            its the clusters C's weight devided by the sum of all clusters
            weights.
            dimensionality (int): Number of features
            min_gauss_dim (int): Minimal number of features for wich the
            cluster point coordinates follow a gaussian distribution
            max_gauss_dim (int): Maximal number of features for wich the
            cluster point coordinates follow a gaussian distribution
            min_coordinate (int): Minimal possible location for the cluster
            centroid location
            max_coordinate (int): Maximal possible location for the cluster
            centroid location
            min_gauss_dim_var (int): Minimal variance for the cluster points
            coordinates following a gaussian distribution
            max_gauss_dim_var (int): Maximal variance for the cluster points
            coordinates following a gaussian distribution
            class_label (int): Class membership label for the current cluster
            Different clusters may constitute a class

        Examples:
            >>> import numpy as np
            >>> np.random.seed(0)
            >>> clus = Cluster(weight=1,
                               dimensionality=3,
                               min_gauss_dim=2,
                               max_gauss_dim=3,
                               min_coordinate=0,
                               max_coordinate=2,
                               min_gauss_dim_var=0.1,
                               max_gauss_dim_var=1,
                               class_label=1)
            >>> clus.generate_point()
            [1.09762701 1.43037873 1.20552675]
        """
        # set dimensionality, weight, boundary and class label
        self.dimensionality = dimensionality
        self.weight = weight
        self.min_coordinate = min_coordinate
        self.max_coordinate = max_coordinate
        self.min_gauss_dim_var = min_gauss_dim_var
        self.max_gauss_dim_var = max_gauss_dim_var
        self.min_gauss_dim = min_gauss_dim
        self.max_gauss_dim = max_gauss_dim
        self.class_label = class_label
        # Draw initial centroids locations
        self.centroid = np.random.uniform(self.min_coordinate,
                                          self.max_coordinate,
                                          self.dimensionality)
        # Draw clusters radii
        self.radii = np.random.uniform(self.min_gauss_dim_var,
                                       self.max_gauss_dim_var,
                                       self.dimensionality)
        self.radii = np.sqrt(self.radii)
        # Draw gauss dimensions and not gauss dimensions
        self.nb_gauss_dims = np.random.randint(self.min_gauss_dim,
                                                   self.max_gauss_dim+1)
        self.nb_not_gauss_dims = self.dimensionality-self.nb_gauss_dims
        gauss_dims_0 = np.zeros(self.nb_gauss_dims)
        unif_dims_1 = np.ones(self.nb_not_gauss_dims)
        self.uniform_dims = np.hstack((unif_dims_1,gauss_dims_0))
        self.uniform_dims = self.uniform_dims.astype(bool)
        np.random.shuffle(self.uniform_dims)
        print(self.centroid)

    def generate_point(self):
        """
        Generate a random point within the cluster

        Returns: numpy.array
        point coordinates along each feature
        """
        point_coordinates = np.random.randn(self.dimensionality)
        point_coordinates *= self.radii
        point_coordinates += self.centroid
        uniform_coord = np.random.uniform(self.min_coordinate,
                                          self.max_coordinate,
                                          self.nb_not_gauss_dims)
        point_coordinates[self.uniform_dims] = uniform_coord
        return point_coordinates

class DatasetGenerator:
    def __init__(self,
                 number_points,
                 weights,
                 dimensionality,
                 min_gauss_dim,
                 max_gauss_dim,
                 min_coordinate,
                 max_coordinate,
                 min_gauss_dim_var,
                 max_gauss_dim_var,
                 class_labels = None,
                 ):
        """
        Cluster is a class that allows to generate datasets consituted of points
        belonging to parametrized clusters.

        Args:
            number_points (int): Number of points in the dataset
            weights (list): List of cluster weights. The probability that a new
            data point belongs to cluster C is simply its the clusters C's
            weight devided by the sum of all clusters weights.
            dimensionality (int): Number of features
            min_gauss_dim (int): Minimal number of features for wich the
            cluster point coordinates follow a gaussian distribution
            max_gauss_dim (int): Maximal number of features for wich the
            cluster point coordinates follow a gaussian distribution
            min_coordinate (int): Minimal possible location for the cluster
            centroid location
            max_coordinate (int): Maximal possible location for the cluster
            centroid location
            min_gauss_dim_var (int): Minimal variance for the cluster points
            coordinates following a gaussian distribution
            max_gauss_dim_var (int): Maximal variance for the cluster points
            coordinates following a gaussian distribution
            class_labels (list): Class membership label for eachs cluster
            Different clusters may constitute a class

        Examples:
            >>> import numpy as np
            >>> np.random.seed(0)
            >>> datagen = DatasetGenerator(
            number_points=100,
            weights=[1,2,3,2],
            dimensionality=2,
            min_gauss_dim=0,
            max_gauss_dim=2,
            min_coordinate=0,
            max_coordinate=10,
            min_gauss_dim_var=0.1,
            max_gauss_dim_var=0.5,
            class_labels=[1,1,2,3],
            )
            >>> X,y = datagen.generate_dataset(100)
            >>> X.head()
                      0         1
            0  7.972476  6.788795
            1  8.538181  7.586156
            2  7.369182  7.087554
            3  6.189440  3.799520
            4  6.093388  3.453909
            >>> y.head()
            0    2
            1    2
            2    1
            3    1
            4    1
            >>> import matplotlib.pyplot as plt
            >>> import seaborn as sns
            >>> X["class"] = y
            >>> sns.pairplot(X, hue="class")
        """
        self.number_points = number_points
        self.dimensionality = dimensionality
        self.min_gauss_dim = min_gauss_dim
        self.max_gauss_dim = max_gauss_dim
        self.min_coordinate = min_coordinate
        self.max_coordinate = max_coordinate
        self.min_gauss_dim_var = min_gauss_dim_var
        self.max_gauss_dim_var = max_gauss_dim_var
        self.weights = np.asarray(weights)
        self.clusters = []
        if class_labels is None:
            class_labels = range(len(self.weights))
        self.class_labels = class_labels
        for i,weight in enumerate(self.weights):
            cluster  = Cluster(weight,
                               dimensionality,
                               min_gauss_dim,
                               max_gauss_dim,
                               min_coordinate,
                               max_coordinate,
                               min_gauss_dim_var,
                               max_gauss_dim_var,
                               class_labels[i],
                               )
            self.clusters.append(cluster)
        self._compute_probabilities_draw_cluster()


    def generate_stream(self):
        """
        Generate the dataset by yielding data objects one after the other.
        This function is suitable for very large datasets
        Returns:
            numpy.array: new data point coordinates
            int: new data point class label
        """
        for i in range(self.number_points):
            self._compute_probabilities_draw_cluster()
            cluster_id = np.random.choice(range(len(self.clusters)),
                                          1,
                                          p=self.probability_draw_cluster)[0]
            new_point = self.clusters[cluster_id].generate_point()
            yield new_point,self.clusters[cluster_id].class_label

    def _compute_probabilities_draw_label(self):
        """
        Compute the probability vector to draw a data point belonging to each
        class

        Returns:
            numpy.array: probability vector
        """
        labels = list(set(self.class_labels))
        N = len(labels)
        props = np.zeros(N)
        S = sum(self.weights)
        for i in range(N):
            props[i] = sum(self.weights[np.array(self.class_labels) == labels[i]])/S
        return props


    def _compute_probabilities_draw_cluster(self):
        """
        Compute the probability vector to draw a data point belonging to each
        cluster

        Returns:
            numpy.array : probability vector
        """
        weights = np.asarray([cluster.weight for cluster in self.clusters])
        self.probability_draw_cluster = weights / weights.sum()

    def generate_dataset(self, size):
        """
        Generate an enitre synthetic dataset and returns it

        Returns:
            pandas.DataFrame: data object coordinates X
            pandas.Series: data object class membership Y
        """
        self.number_points = size
        X = []
        Y = []
        for i,point in enumerate(self.generate_stream()):
            x,y = point
            X.append(x)
            Y.append(y)
        X = pd.DataFrame(X,columns= list(range(self.dimensionality)))
        Y = pd.Series(Y)
        return X,Y

    def _get_file_name(self):
        """
        Generate a standard file name describing the dataset to save it

        Returns:
            str: file name
        """
        d = str(self.dimensionality)
        c = str(self.weights.size)
        n = str(self.number_points)
        return "D_"+d+"_C_"+c+"_N_"+n
