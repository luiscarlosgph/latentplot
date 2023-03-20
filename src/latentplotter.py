"""
@brief  TODO
@author Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   1 Mar 2023.
"""
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import sklearn.manifold
import scipy.spatial
import os
import tempfile
import cv2


class Plotter():
    #methods = {
    #    'pca' : Plotter._pca,
    #    'tsne': Plotter._tsne,
    #}

    def __init__(self, method='pca', width=7680, height=4320, dpi=300,
                 cell_factor=0.05, **kwargs):
        """
        @param[in]  method       TODO
        @param[in]  width        Desired image width. 
                                 Default is 8K (7680x4320 pixels).
        @param[in]  height       Desired image height.
                                 Default is 8K (7680x4320 pixels).
        @param[in]  dpi          TODO
        @param[in]  cell_factor  TODO
        """
        # Save attributes
        self.method = method
        self.width = width
        self.height = height
        self.dpi = dpi
        self.cell_factor = cell_factor

        # Save attributes for the dimensionality reduction technique
        self.options = kwargs

    def plot(self, images, fv, labels=[]) -> np.ndarray:
        """
        @brief  Method that converts all the feature vectors to a 2D space and
                plots the corresponding images on the reduced space. 

        @param[in]  images  BGR images to display in latent space.
        @param[in]  fv      Feature vectors corresponding to the images. 
        @param[in]  labels  Labels corresponding to the images. Either a list
                            of integers or an empty list is expected.

        @returns a BGR image.
        """
        # Convert lists to np.array to simplify the code
        int_images = images if type(images) == np.ndarray else np.array(images)
        int_fv = fv if type(fv) == np.ndarray else np.array(fv)
        int_labels = labels if type(labels) == np.ndarray else np.array(labels)

        # Convert latent vectors to 2D
        reduced_fv = self._reduce_fv(int_fv)
        if reduced_fv.shape[1] != 2: 
            msg = '[ERROR] The class Plotter does not support other reduced ' \
                + 'dimensions than 2.'
            raise ValueError(msg)

        # Hijack latent vectors so that there is one image per cell in the
        # plot
        pruned_fv = self._prune_fv(reduced_fv)

        # Generate plot
        fig = self._generate_plot(int_images, pruned_fv, int_labels)

        # Convert figure into an image
        temp_path = os.path.join(tempfile.gettempdir(), 'lplot.png')
        fig.savefig(temp_path, dpi=self.dpi, format='png')
        plt.close(fig)
        im = cv2.imread(temp_path)
        os.unlink(temp_path)

        return im

    def _prune_fv(self, reduced_fv):
        # TODO
        return reduced_fv

    def _generate_plot(self, images: np.ndarray, reduced_fv: np.ndarray, 
                       labels: np.ndarray, margin=1.):
        """
        @brief  Method that contains the matplotlib/seaborn code to produce
                the plot of the reduced embedded space.

        @param[in]  images  BGR images to display in latent space.
        @param[in]  fv      Feature vectors corresponding to the images. 
        @param[in]  labels  Labels corresponding to the images.
        @param[in]  margin  Margin to add to the sides of the reduced 
                            embedded space so that we plot the images properly.

        @returns  a matplotlib figure containing the plot.
        """
        # sanity check, we work only with numpy arrays to simplify the code
        assert(type(images) == np.ndarray)
        assert(type(reduced_fv) == np.ndarray)
        assert(type(labels) == np.ndarray)

        # Create figure
        sns.set_style('white')
        fig, ax = plt.subplots(figsize=(float(self.width) / self.dpi, 
            float(self.height) / self.dpi))
        ax.set_axis_bgcolor = 'black'
        min_x = np.min(reduced_fv[:, 0])
        max_x = np.max(reduced_fv[:, 0])
        min_y = np.min(reduced_fv[:, 1])
        max_y = np.max(reduced_fv[:, 1])
        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(min_y - margin, max_y + margin)
        cell_width = float(max_x - min_x) * self.cell_factor
        cell_height = float(max_y - min_y) * self.cell_factor

        # Plot all the images
        for i in range(len(images)):
            # Read image and reduced feature vector
            im = images[i, ...]
            fv = reduced_fv[i, :]
            
            # Display image on plot
            self._imscatter(im, fv[0], fv[1], ax, (cell_width, cell_height))

            # If the type of the label is a class index, we use it
            #if type(label) == int:
            #label = labels[i]
            #    print('label.shape:', label.shape)

        # Tight layout and black background
        ax.set_facecolor('black')
        plt.tight_layout()

        return ax.get_figure()

    def _imscatter(self, im, x, y, ax, size, border=False, linewidth=0, 
            edge_val=None, edge_min=0, edge_max=1, cmap='RdYlGn', fontsize=6):
       	# Convert image to RGB
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        
        # Display image on the plot
        xmin = x - size[0] / 2.
        xmax = xmin + size[0] 
        ymin = y - size[1] / 2.
        ymax = ymin + size[1]
        ax.imshow(im_rgb, extent=(xmin, xmax, ymin, ymax), origin='upper')

        # FIXME: Decide edgecolur according to the value given by the user
        if edge_val is not None:
            norm = matplotlib.colors.Normalize(vmin=edge_min, vmax=edge_max, clip=True)
            mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
            edgecolor = mapper.to_rgba(edge_val) 
            ax.text(x, y, "%.1f" % (edge_val * 100), fontsize=fontsize, color='white')
        else:
            edgecolor = None

        # FIXME: Create a rectangle patch
        rect = matplotlib.patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
            linewidth=linewidth, edgecolor=edgecolor, facecolor='None')
        ax.add_patch(rect)

        return ax  

    def _reduce_fv(self, fv, dim=2):
        # Check that the vectors come in an array
        assert(type(fv) == np.ndarray)

        # These are the dimensionality reduction methods that we support
        methods = {
            'pca': self._pca,
            'tsne': self._tsne,
        }

        # If we are asked for a method we don't know, raise an error
        if self.method not in methods:
            msg = "[ERROR] Reduction method {} unknown."
            ValueError(msg.format(self.method))
        else:
            # If the output of the method is a list, we convert it to an array
            result = methods[self.method](self, fv, dim)
            if type(result) == np.ndarray:
                pass
            elif type(result) == list:
                result = np.array(result)
            else:
                msg = '[ERROR] The reduction technique did not return ' \
                    + 'a list or an array.' 
                raise ValueError(msg)
            return result

    @staticmethod
    def _pca(self, fv, dim):
        """
        @brief Performs dimensionality reduction based on PCA.
        @returns an array of (samples, features) of shape (N, 2).
        """
        pca = sklearn.decomposition.PCA(n_components=dim)
        return pca.fit_transform(fv)

    @staticmethod
    def _tsne(self, fv, dim, perplexity=40, n_iter=1000):
        """
        @brief Reduce dimensionality with t-SNE.
        @returns an array of (samples, features) of shape (N, 2).
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
