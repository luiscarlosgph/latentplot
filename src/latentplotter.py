"""
@brief  Module to plot the latent space of a group of images.
@author Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   1 Mar 2023.
"""
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn.manifold
import scipy.spatial
import umap
import os
import tempfile
import PIL
import typing


class Plotter:
    def __init__(self, method='pca', width=15360, height=8640, dpi=300,
                 cell_factor=0.01, dark_mode=True, hide_axes=True, **kwargs):
        """
        @param[in]  method       Method used to reduce the feature vectors to
                                 a 2D space. Available options: pca, tsne,
                                 umap.
        @param[in]  width        Desired image width. 
        @param[in]  height       Desired image height.
        @param[in]  dpi          DPI for the output image.
        @param[in]  cell_factor  Proportion of the reduced space that each
                                 cell will occupy.
        @param[in]  dark_mode    Set it to False to have a white background
                                 with black font. Default is True.
        @param[in]  hide_axes    Hide axes, ticks and marks. Default is True.
        @param[in]  kwargs       The rest of the arguments for the 
                                 dimensionality reduction method.
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
        self.dark_mode = dark_mode
        self.hide_axes = hide_axes

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
            self._hijack_reduced_space(int_images, reduced_fv, int_labels)

        # Generate plot
        fig = self._generate_plot(hijacked_images, hijacked_fv, 
            hijacked_labels)

        # Convert figure into an image
        temp_path = os.path.join(tempfile.gettempdir(), 'lplot.png')
        fig.savefig(temp_path, dpi=self.dpi, format='png', 
                    bbox_inches='tight')
        plt.close(fig)
        im = PIL.Image.open(temp_path)
        os.unlink(temp_path)
        
        # Resize image to the resolution expected by the user
        im_resized = im.resize((self.width, self.height), 
                               resample=PIL.Image.Resampling.LANCZOS)

        # Return BGR image
        return np.array(im_resized)[:, :, ::-1].copy()


    def _hijack_reduced_space(self, images: np.ndarray, reduced_fv: np.ndarray, 
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
        min_x = reduced_fv[:, 0].min()
        max_x = reduced_fv[:, 0].max()
        min_y = reduced_fv[:, 1].min()
        max_y = reduced_fv[:, 1].max()
        num_cells = int(round(1. / self.cell_factor))
        x_values = np.linspace(min_x, max_x, num_cells)
        y_values = np.linspace(min_y, max_y, num_cells)
        cell_width = x_values[1] - x_values[0]
        cell_height = y_values[1] - y_values[0]
        cell_half_width = .5 * cell_width
        cell_half_height = .5 * cell_height

        # Loop over the reduced latent space
        for i in range(y_values.shape[0]):
            for j in range(x_values.shape[0]):
                # Get coordinate of the point
                x = x_values[j]
                y = y_values[i]

                # Find the closest point in the reduced space
                idx = self._get_closest_image(np.array([x, y]), kdtree)
                closest_x = reduced_fv[idx, 0]
                closest_y = reduced_fv[idx, 1]

                # Check if the closest point is inside the cell 
                x_start = x - cell_half_width
                x_end = x + cell_half_width
                y_start = y - cell_half_height
                y_end = y + cell_half_height
                if x_start <= closest_x < x_end:
                    if y_start <= closest_y < y_end:
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
        # Sanity check, we work only with numpy arrays to simplify the code
        assert(type(images) == np.ndarray)
        assert(type(reduced_fv) == np.ndarray)
        assert(type(labels) == np.ndarray)

        # Set dark mode if requested
        if self.dark_mode:
            plt.style.use('dark_background')

        # Create figure
        fig, ax = plt.subplots(figsize=(float(self.width) / self.dpi, 
                                        float(self.height) / self.dpi))
        
        # Configure figure limits depending on the data
        min_x = reduced_fv[:, 0].min()
        max_x = reduced_fv[:, 0].max()
        min_y = reduced_fv[:, 1].min()
        max_y = reduced_fv[:, 1].max()
        num_cells = int(round(1. / self.cell_factor))
        x_values = np.linspace(min_x, max_x, num_cells)
        y_values = np.linspace(min_y, max_y, num_cells)
        cell_width = x_values[1] - x_values[0]
        cell_height = y_values[1] - y_values[0]
        cell_half_width = .5 * cell_width
        cell_half_height = .5 * cell_height
        ax.set_xlim(min_x - cell_width - cell_half_width, max_x + cell_width + cell_half_width)
        ax.set_ylim(min_y - cell_height - cell_half_height, max_y + cell_height + cell_half_height)

        # Hide axes if requested by the user
        if self.hide_axes:
            ax.axis('off')
        
        # If labels are provided, find the smallest and largest class indices
        edge_min = None
        edge_max = None
        if labels.shape[0] != 0:
            edge_min = np.min(labels)
            edge_max = np.max(labels)
        
        # Plot all the images
        for i in range(len(images)):
            # Read image and reduced feature vector
            im = images[i, ...]
            fv = reduced_fv[i, :]
            
            # If we have labels for the images
            edge_val = None
            if labels.shape[0] != 0:
                # If the type of the label is a class index, we use it
                if np.issubdtype(labels[i], np.integer):
                    edge_val = labels[i] 
            
            # Display image on plot
            self._imscatter(im, fv[0], fv[1], ax, (cell_width, cell_height),
                linewidth=2, edge_val=edge_val, edge_min=edge_min, 
                edge_max=edge_max)

        return ax.get_figure()

    def _imscatter(self, im: np.ndarray, x: float, y: float, ax, 
            size: typing.Tuple[int, int], interpolation: str = 'bilinear', 
            linewidth: int = 0, edge_val: int = None, edge_min: int = None, 
            edge_max: int = None, cmap: str = 'RdYlGn'):
        """
        @brief Displays an image on the reduced latent space plot.
        @details The x and y coordinates represent the coordinate where the
                 centre of the image will be located.

        @param[in]  im             BGR image to be displayed.
        @param[in]  x              Horizontal coordinate in the plot.
        @param[in]  y              Vertical coorinfate in the plot.
        @param[in]  ax             Axes of the plt.subplots of the figure.
        @param[in]  size           Size of the cell in reduced latent space 
                                   coordinates.
        @param[in]  interpolation  Interpolation method used to resample the
                                   image onto the plot.
        @param[in]  linewidth      Width of the rectangle line. A rectangle
                                   that is not filled is displayed around each
                                   image to indicate the class label of the 
                                   image.
        @param[in]  edge_val       Class index or value that will define the
                                   colour of the rectangle line around the
                                   image.
        @param[in]  edge_min       Minimum value of the class indices.
        @param[in]  edge_max       Maximum class index value.
        @param[in]  cmap           Colourmap used to map the class labels into
                                   colours.
        """
        # Sanity check
        assert(edge_val is None or np.issubdtype(edge_val, np.integer))
        if edge_val is not None:
            assert(np.issubdtype(edge_min, np.integer))
            assert(np.issubdtype(edge_max, np.integer))

       	# Convert image to RGB for matplotlib
        im_rgb = im[...,::-1].copy()

        # Display image on the plot
        xmin = x - .5 * size[0]
        xmax = x + .5 * size[0] 
        ymin = y - .5 * size[1]
        ymax = y + .5 * size[1]
        ax.imshow(im_rgb, origin='upper', extent=(xmin, xmax, ymin, ymax),
                  interpolation=interpolation)

        # Decide edge colour according to the value (typically class index)
        # provided by the user
        edgecolor = None
        if edge_val is not None:
            norm = matplotlib.colors.Normalize(vmin=edge_min, vmax=edge_max, clip=True)
            mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
            edgecolor = mapper.to_rgba(edge_val)

        # Create a rectangle patch
        rect = matplotlib.patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
            linewidth=linewidth, edgecolor=edgecolor, facecolor='None')
        ax.add_patch(rect)

        return ax  

    def _reduce_fv(self, fv: np.ndarray, dim: int = 2):
        """
        @brief Method used to reduce the feature vectors to a 2D latent
               space that we can visualise.
        @param[in]  fv   Feature vectors corresponding to the images. 
        @param[in]  dim  Number of dimensions to reduce the vectors to.
        """
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

    def _pca(self, fv: np.ndarray, dim: int):
        """
        @brief Performs dimensionality reduction based on PCA.
        @param[in]  fv   Feature vectors corresponding to the images. 
        @param[in]  dim  Number of dimensions to reduce the vectors to.
        @returns an array of (samples, features) of shape (N, 2).
        """
        pca = sklearn.decomposition.PCA(n_components=dim)
        return pca.fit_transform(fv)

    def _tsne(self, fv: np.ndarray, dim: int, perplexity=40, n_iter=1000,
            learning_rate='auto', init='pca'):
        """
        @brief Reduce dimensionality with t-SNE.
        @param[in]  fv   Feature vectors corresponding to the images. 
        @param[in]  dim  Number of dimensions to reduce the vectors to.
        @returns an array of (samples, features) of shape (N, 2).
        """
        tsne = sklearn.manifold.TSNE(n_components=dim, verbose=0,
                                     perplexity=perplexity, n_iter=n_iter,
                                     learning_rate=learning_rate, init=init)
        return tsne.fit_transform(fv)

    def _umap(self, fv: np.ndarray, dim: int):
        """
        @brief Dimensionality reduction with UMAP.
        @param[in]  fv   Feature vectors corresponding to the images. 
        @param[in]  dim  Number of dimensions to reduce the vectors to.
        @returns an array of (samples, features) of shape (N, 2).
        """
        reducer = umap.UMAP(n_components=dim)
        return reducer.fit_transform(fv)
    
    # These are the dimensionality reduction methods that we support
    methods = {
        'pca': _pca,
        'tsne': _tsne,
        'umap': _umap,
    }


if __name__ == '__main__':
    raise RuntimeError('[ERROR] This module (latentplot) cannot be run as a script.')
