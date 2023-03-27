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
import cv2
import tempfile
import videosum

# My imports
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
        im_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        
        # Add the converted image to the list
        samples.append((im_bgr, label))

    return samples


def get_pascalvoc_samples(n):
    """
    @brief Get a random list of samples from the training set of 
           PascalVOC 2012.
    @param[in]  n  Number of samples to draw, max is TODO
    @returns a list of (image, label) pairs from PascalVOC 2012. The images
             returned are in shape (H, W, 3), and they are BGR.

    """
    # Load the PascalVOC dataset
    pascalvoc = torchvision.datasets.VOCClassification(
        root=tempfile.gettempdir(), image_set='train', year='2012',
        download=True)
    
    # Check that the number of samples requested is available in PascalVOC
    assert(n < len(pascalvoc))
    
    # Loop over the images and convert them to BGR format
    samples = []
    for i in range(n):
        # Load the RGB image
        im = np.array(pascalvoc[i][0])

        # Make sure that the image shape is (H, W, 3)
        assert(im.shape[2] == 3)

        # Load the label
        label = pascalvoc[i][1]

        # Convert from RGB to BGR
        im_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        
        # Add the converted image to the list
        samples.append((im_bgr, label))
    
    return samples


def preprocess_InceptionV3():
    # TODO
    pass


"""
def split_cifar10_samples(samples):
    #@brief Convert a list of samples into two lists, one containing the
    #       BGR images, and another one containing the labels.
    #@param[in]  samples  List of (image, label) pairs.
    #@returns two lists, BGR images and labels.

    # Bring samples to CPU memory
    cpu_images = [x[0][0].cpu().detach().numpy() for x in samples]

    # Convert images from shape (3, 384, 384) to (384, 384, 3) 
    rgb_images = [x.transpose((1, 2, 0)) for x in cpu_images]

    # Convert images to BGR
    bgr_images = [x[...,::-1].copy() for x in rgb_images]

    return bgr_images, [x.cpu().detach().numpy() for x in samples[1]]
"""


class TestStringMethods(unittest.TestCase):

    def test_pca_plot(self, num_images=5000):
        """
        @brief Test that the PCA plot is produced without errors.
        """
        # Get samples from CIFAR-10 
        samples = get_cifar10_samples(num_images)
        images = [x[0] for x in samples]

        # Get latent vector for the images using InceptionV3
        model = videosum.InceptionFeatureExtractor('vector')
        feature_vectors = np.array([model.get_latent_feature_vector(x) for x in images])

        # Plot PCA
        plotter = latentplot.Plotter(method='pca')
        plot = plotter.plot(images, feature_vectors)
        cv2.imwrite('test/data/pca.png', plot)
    
    def test_tsne_plot(self, num_images=5000):
        # Get samples from CIFAR-10 
        samples = get_cifar10_samples(num_images)
        images = [x[0] for x in samples]

        # Get latent vector for the images using InceptionV3
        model = videosum.InceptionFeatureExtractor('vector')
        feature_vectors = np.array([model.get_latent_feature_vector(x) for x in images])

        # Plot t-SNE
        plotter = latentplot.Plotter(method='tsne')
        plot = plotter.plot(images, feature_vectors)
        cv2.imwrite('test/data/tsne.png', plot)
        

if __name__ == '__main__':
    unittest.main()
