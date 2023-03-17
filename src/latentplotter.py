"""
@brief  TODO
@author Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   1 Mar 2023.
"""
import random
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.manifold
import scipy.spatial


class Plotter():
    #methods = {
    #    'pca' : Plotter._pca,
    #    'tsne': Plotter._tsne,
    #}

    def __init__(self, method='pca', width=1920, height=1080, thumb_width=32,
                 thumb_height=32, **kwargs):
        """
        @param[in]  method  Method that you want to use to  
        """
        # Save attributes
        self.method = method
        self.width = width
        self.height = height
        self.thumb_width = thumb_width
        self.thumb_height = thumb_height

        # Save attributes for the dimensionality reduction technique
        self.options = kwargs

    def plot(self, images, labels, fv):
        """
        @param[in]  images  Images to display in latent space.
        @param[in]  labels  Labels corresponding to the images.
        @param[in]  fv      Feature vectors corresponding to the images.
        """
        # Convert latent vectors to 2D
        reduced_fv = self._reduce_fv(fv)
        print(reduced_fv)

        # TODO: Compute the boundaries of the reduced latent space

    def _reduce_fv(self, fv, dim=2):
        methods = {
            'pca': self._pca,
            'tsne': self._tsne,
        }
        if self.method not in methods:
            ValueError("[ERROR] Reduction method {} unknown.".format(self.method))
        else:
            return methods[self.method](self, fv, dim)

    @staticmethod
    def _pca(self, fv, dim):
        # TODO
        raise NotImplemented

    @staticmethod
    def _tsne(self, fv, dim, perplexity=40, n_iter=1000):
        """
        @brief Reduce dimensionality with t-SNE.
        """
        tsne = sklearn.manifold.TSNE(n_components=dim, verbose=0,
                                     perplexity=perplexity, n_iter=n_iter)
        return tsne.fit_transform(fv)
	
    @staticmethod
    def get_tsne_df(train_latent, test_latent, coord_labels=['x', 'y'],
            split_col_name='split', perplexity=40, n_iter=1000):
        """
        @brief Takes a list of vectors for training and testing and produces a 
               Pandas DataFrame with the t-SNE representation of them.
        @param[in]  train_latent    List of arrays. Each array has only one 
                                    dimension and represents an image.
        @param[in]  test_latent     List of arrays. Each array has only one 
                                    dimension and represents an image.
        @param[in]  coord_labels    List of strings with the names of the
                                    axes of the reduced space, for example if
                                    you want a 2D reduced space pass something 
                                    like ['x', 'y'] or ['tsne-1', 'tsne-2'].
                                    For 3D you can pass ['x', 'y', 'z'] or 
                                    ['tsne-1', 'tsne-2', tsne-3'].
        @param[in]  split_col_name  Name of the dataframe column with the 
                                    string 'train' or 'test'.
        @param[in]  perplexity      t-SNE perplexity.
        @param[in]  n_iter          Number of t-SNE iterations.
        @returns a pandas.DataFrame with the columns 'x', 'y', 'z', 'mode' 
                 (without 'z' if n_components=2).
        """
        n_components = len(coord_labels)
        assert(n_components == 2 or n_components == 3)

        # Reduce dimensionality with t-SNE 
        tsne = sklearn.manifold.TSNE(n_components=n_components, verbose=0, 
                                     perplexity=perplexity, n_iter=n_iter)
        latent_reduced = tsne.fit_transform(np.vstack((train_latent, 
                                                       test_latent)))
        train_size = len(train_latent)
        val_size = len(test_latent)
        
        # Add train samples to DataFrame
        data = []
        for i in range(train_size):
            sample = {}
            for c in range(n_components):
                sample[coord_labels[c]] = latent_reduced[i, c]
            sample[split_col_name] = 'train'
            data.append(sample)
        
        # Add validation samples to DataFrame
        for i in range(train_size, train_size + val_size):
            sample = {}
            for c in range(n_components):
                sample[coord_labels[c]] = latent_reduced[i, c]
            sample[split_col_name] = 'test'
            data.append(sample)
        
        # Construct DataFrame
        df = pd.DataFrame(data=data, columns=coord_labels[:n_components] \
            + [split_col_name])
        
        return df
    
    @staticmethod
    def prune_tsne_df(df, coord_labels=['x', 'y'], spacing = 1.0):
        """
        @brief Prune a t-SNE data frame of points so that each point can be 
               visualised as an image without having overlap among them.
        @param[in]  df  Pandas DataFrame with columns: 'x', 'y', 'z', 'split' 
                        (without 'z' if n_components is == 2).
        @returns two lists (centroid_pos, indices):
                 centroid_pos: list of points [x, y] that represent the cell 
                               centroids.
                 indices: list of the indices of the images that were closest 
                          to the centroids.
        """
        
        # Get minimum and maximum for all the axes
        n_components = len(coord_labels)
        mins = {}
        maxs = {}
        for k in coord_labels:
            mins[k] = df.min()[k]
            maxs[k] = df.max()[k]

        # Insert all the points on a kdtree
        kdtree = scipy.spatial.cKDTree(df.to_numpy()[:, :n_components])

        # Loop through the integer version of the grid and capture points closest to centroid
        indices = []
        centroid_pos = []
        cell_half = .5 * spacing
        max_dist = np.sqrt(cell_half ** 2 + cell_half ** 2)
        for x in np.arange(int(np.floor(mins['x'])), int(np.ceil(maxs['x'])), spacing).tolist():
            for y in np.arange(int(np.floor(mins['y'])), int(np.ceil(maxs['y'])), spacing).tolist():
                list_points = kdtree.query_ball_point([x, y], max_dist)
                if list_points:
                    # Find the closest point within the cell
                    p = kdtree.data[list_points[0], ...]

                    # Insert it into the new data frame
                    if p[0] >= x - cell_half and p[0] < x + cell_half and p[1] >= y - cell_half \
                        and p[1] < y + cell_half:
                            centroid_pos.append([x, y])
                            indices.append(list_points[0])
        return centroid_pos, indices 


if __name__ == '__main__':
    raise RuntimeError('[ERROR] This module (latentplot) cannot be run as a script.')
