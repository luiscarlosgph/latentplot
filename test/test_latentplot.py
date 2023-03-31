"""
@brief  Unit tests to test that the PCA and t-SNE plots stay the same and 
        working over time.
@author Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   16 Mar 2023.
"""
import unittest
import numpy as np
import torch
import torchvision
import tempfile
import PIL

# My imports
import videosum
import latentplot


# Global options
torch.multiprocessing.set_sharing_strategy('file_system')


def get_cifar10_samples(n: int) -> np.ndarray:
    """
    @brief Get a random list of samples from the training set of CIFAR-10.
    @param[in]  n  Number of samples to draw, max is 50K.
    @returns a list of (image, label) pairs from CIFAR-10. The images returned
             are in shape (H, W, 3), and they are BGR.
    """
    # Load the CIFAR-10 dataset
    cifar10 = torchvision.datasets.CIFAR10(root=tempfile.gettempdir(), 
        train=True, download=True)
    
    # Check that the number of samples requested is available in CIFAR-10
    assert(n < len(cifar10))

    # Loop over the images and convert them to BGR format
    samples = []
    for i in range(n):
        # Load the RGB image
        im = np.array(cifar10[i][0])

        # Make sure that the image shape is (H, W, 3)
        assert(im.shape[2] == 3)

        # Load the label
        label = cifar10[i][1]

        # Convert from RGB to BGR
        im_bgr = im[...,::-1].copy()
        
        # Add the converted image to the list
        samples.append((im_bgr, label))

    return samples


class TestVisualizationMethods(unittest.TestCase):

    def test_pca_plot(self, width: int = 15360, height: int = 8640, 
            num_images: int = 30000, path: str = 'test/data/pca.png'):
        """
        @brief Test that the PCA plot is produced without errors.
        """
        print('[INFO] Running PCA unit test ...')

        # Get samples from CIFAR-10 
        samples = get_cifar10_samples(num_images)
        images = [x[0] for x in samples]
        labels = [x[1] for x in samples]

        # Get latent vector for the images using InceptionV3
        model = videosum.InceptionFeatureExtractor('vector')
        feature_vectors = np.array([model.get_latent_feature_vector(x) for x in images])

        # Plot PCA
        plotter = latentplot.Plotter(method='pca')
        plot = plotter.plot(images, feature_vectors, labels)
        
        # Convert plot image from BGR to RGB
        plot_rgb = plot[...,::-1].copy()

        # Write image to disk
        im = PIL.Image.fromarray(plot_rgb)
        im.save(path)

        # Test that the image produced is of the expected resolution
        self.assertTrue(plot.shape[0] == height)
        self.assertTrue(plot.shape[1] == width)
        
        print('[INFO] PCA unit test completed.')

    def test_tsne_plot(self, width: int = 15360, height: int = 8640, 
            num_images: int = 30000, path: str = 'test/data/tsne.png'):
        """
        @brief Test that the t-SNE plot is produced without errors.
        """
        print('[INFO] Running t-SNE unit test ...')

        # Get samples from CIFAR-10 
        samples = get_cifar10_samples(num_images)
        images = [x[0] for x in samples]
        labels = [x[1] for x in samples]

        # Get latent vector for the images using InceptionV3
        model = videosum.InceptionFeatureExtractor('vector')
        feature_vectors = np.array([model.get_latent_feature_vector(x) for x in images])

        # Plot t-SNE
        plotter = latentplot.Plotter(method='tsne')
        plot = plotter.plot(images, feature_vectors, labels)

        # Convert plot image from BGR to RGB
        plot_rgb = plot[...,::-1].copy()

        # Write image to disk
        im = PIL.Image.fromarray(plot_rgb)
        im.save(path)

        # Test that the image produced is of the expected resolution
        self.assertTrue(plot.shape[0] == height)
        self.assertTrue(plot.shape[1] == width)

        print('[INFO] t-SNE unit test completed.')

    def test_umap_plot(self, width: int = 15360, height: int = 8640, 
            num_images: int = 30000, path: str = 'test/data/umap.png'):
        """
        @brief Test that the UMAP plot is produced without errors.
        """
        print('[INFO] Running UMAP unit test ...')

        # Get samples from CIFAR-10 
        samples = get_cifar10_samples(num_images)
        images = [x[0] for x in samples]
        labels = [x[1] for x in samples]

        # Get latent vector for the images using InceptionV3
        model = videosum.InceptionFeatureExtractor('vector')
        feature_vectors = np.array([model.get_latent_feature_vector(x) for x in images])

        # Plot UMAP
        plotter = latentplot.Plotter(method='umap')
        plot = plotter.plot(images, feature_vectors, labels)
        
        # Convert plot image from BGR to RGB
        plot_rgb = plot[...,::-1].copy()

        # Write image to disk
        im = PIL.Image.fromarray(plot_rgb)
        im.save(path)

        # Test that the image produced is of the expected resolution
        self.assertTrue(plot.shape[0] == height)
        self.assertTrue(plot.shape[1] == width)

        print('[INFO] UMAP unit test completed.')


if __name__ == '__main__':
    unittest.main()
