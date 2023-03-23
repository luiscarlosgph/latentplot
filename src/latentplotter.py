"""
@brief  Module to plot the latent space of a group of images.
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


class Plotter:
    def __init__(self, method='pca', width=15360, height=8640, dpi=300,
                 cell_factor=0.05, **kwargs):
        """
        @param[in]  method       Method used to reduce the feature vectors to
                                 a 2D space. Available options: pca, tsne.
        @param[in]  width        Desired image width. 
        @param[in]  height       Desired image height.
        @param[in]  dpi          DPI for the output image.
        @param[in]  cell_factor  Proportion of the reduced space that each
                                 cell will occupy.
        """
        # Check that the reduction method is known
        if method not in Plotter.methods:
            msg = "[ERROR] Reduction method {} unknown."
            ValueError(msg.format(self.method))

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

        @param[in]  images         BGR images to display in latent space.
        @param[in]  fv             Feature vectors corresponding to the images. 
        @param[in]  labels         Labels corresponding to the images. 
                                   Either a list of integers or an empty list 
                                   is expected.

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
        hijacked_images, hijacked_fv, hijacked_labels = \
            self._hijack_fv(int_images, reduced_fv, int_labels)

        # Generate plot
        fig = self._generate_plot(hijacked_images, hijacked_fv, 
            hijacked_labels)

        # Convert figure into an image
        temp_path = os.path.join(tempfile.gettempdir(), 'lplot.png')
        fig.savefig(temp_path, dpi=self.dpi, format='png', bbox_inches='tight')
        plt.close(fig)
        im = cv2.imread(temp_path)
        os.unlink(temp_path)

        return im

    def _hijack_fv(self, images: np.ndarray, reduced_fv: np.ndarray, 
                   labels: np.ndarray):
        """
        @brief This method divides the reduced space into a grid. For each cell
               in the grid it chooses one image to be displayed, the closest
               to the centre of the cell.

        @param[in]  images      Numpy array of images, shape (n, H, W, 3).
        @param[in]  reduced_fv  Numpy array of 2D vectors (n, m), with n
                                samples of dimension m = 2.
        @param[in]  labels      Numpy array of labels, can be empty.

        @returns  a tuple of images, feature vectors and labels.
        """
        hijacked_images = []
        hijacked_fv = []
        hijacked_labels = []

        # Check that the number of images, feature vectors and labels is the
        # same
        if images.shape[0] != reduced_fv.shape[0]:
            raise ValueError('[ERROR] We need as many images as ' \
                + 'feature vectors.')
        if labels.shape[0] != 0 and labels.shape[0] != reduced_fv.shape[0]:   
            raise ValueError('[ERROR] We need as many labels as ' \
                'feature vectors.')

        # Create a KDTree of all the points
        kdtree = scipy.spatial.cKDTree(reduced_fv)

        # Generate a grid on the reduced latent space
        min_x = np.min(reduced_fv[:, 0])
        max_x = np.max(reduced_fv[:, 0])
        min_y = np.min(reduced_fv[:, 1])
        max_y = np.max(reduced_fv[:, 1])
        cell_width = (max_x - min_x) * self.cell_factor
        cell_height = (max_y - min_y) * self.cell_factor
        cell_half_width = .5 * cell_width
        cell_half_height = .5 * cell_height
        x_values = np.linspace(min_x - cell_width, max_x + cell_width)
        y_values = np.linspace(min_y - cell_height, max_y + cell_height)

        # Loop over the reduced latent space
        for i in range(y_values.shape[0]):
            for j in range(x_values.shape[0]):
                # Get coordinate of the point
                x = x_values[j]
                y = y_values[i]

                # Find the closest point in the reduced space
                idx = self._get_closest_image(np.array([x, y]), kdtree)
                
                # Check if the closest point is inside the cell 
                closest_x = reduced_fv[idx, 0]
                closest_y = reduced_fv[idx, 1]
                if abs(closest_x - x) < cell_half_width \
                        and abs(closest_y - y) < cell_half_height:
                    hijacked_images.append(images[idx])
                    hijacked_fv.append([x, y])
                    if labels.shape[0] != 0:
                        hijacked_labels.append(labels[idx])

        return np.array(hijacked_images), np.array(hijacked_fv), np.array(hijacked_labels)

    def _get_closest_image(self, fv, kdtree):
        """
        @brief Given a feature vector, find the closest image (and 
               corresponding label) in the reduced latent space.

        @param[in]  fv          Input feature vector.
        @param[in]  kdtree      KDTree of the reduced latent space.

        @returns  the index of the closest element.
        """
        nearest_image = None
        label = None

        # Sanity checks
        if fv.shape[0] != 2:
            raise NotImplemented('[ERROR] Only feature vectors of ' \
                + 'dimension 2 are supported.')

        # Find the closest vector to the one provided
        distances, indices = kdtree.query([fv])

        return indices[0]

    def _generate_plot(self, images: np.ndarray, reduced_fv: np.ndarray, 
                       labels: np.ndarray):
        """
        @brief  Method that contains the matplotlib/seaborn code to produce
                the plot of the reduced embedded space.

        @param[in]  images         BGR images to display in latent space.
        @param[in]  fv             Feature vectors corresponding to the images. 
        @param[in]  labels         Labels corresponding to the images.

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
        #ax.set_axis_bgcolor = 'black'
        min_x = np.min(reduced_fv[:, 0])
        max_x = np.max(reduced_fv[:, 0])
        min_y = np.min(reduced_fv[:, 1])
        max_y = np.max(reduced_fv[:, 1])
        margin_x = (max_x - min_x) * self.cell_factor
        margin_y = (max_y - min_y) * self.cell_factor
        ax.set_xlim(min_x - margin_x, max_x + margin_x)
        ax.set_ylim(min_y - margin_y, max_y + margin_y)
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
        #ax.set_facecolor('black')
        #plt.tight_layout()

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

        # If the output of the method is a list, we convert it to an array
        result = Plotter.methods[self.method](self, fv, dim)
        if type(result) == np.ndarray:
            pass
        elif type(result) == list:
            result = np.array(result)
        else:
            msg = '[ERROR] The reduction technique did not return ' \
                + 'a list or an array.' 
            raise ValueError(msg)
        return result

    def _pca(self, fv, dim):
        """
        @brief Performs dimensionality reduction based on PCA.
        @returns an array of (samples, features) of shape (N, 2).
        """
        pca = sklearn.decomposition.PCA(n_components=dim)
        return pca.fit_transform(fv)

    def _tsne(self, fv, dim, perplexity=40, n_iter=1000):
        """
        @brief Reduce dimensionality with t-SNE.
        @returns an array of (samples, features) of shape (N, 2).
        """
        tsne = sklearn.manifold.TSNE(n_components=dim, verbose=0,
                                     perplexity=perplexity, n_iter=n_iter)
        return tsne.fit_transform(fv)
    
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
    
    # These are the dimensionality reduction methods that we support
    methods = {
        'pca': _pca,
        'tsne': _tsne,
    }


if __name__ == '__main__':
    raise RuntimeError('[ERROR] This module (latentplot) cannot be run as a script.')
