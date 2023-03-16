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


class TestStringMethods(unittest.TestCase):

    def test_pca_plot(self, num_images=1000, width=1920, height=1080):
        # Prepare CIFAR-10 dataloader
        preproc_tf = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.Resize(size=384),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset = torchvision.datasets.CIFAR10(root=tempfile.gettempdir(), 
                                                train=True, download=True,
                                                transform=preproc_tf)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                 shuffle=True, num_workers=1)
        dataiter = iter(dataloader)

        # Get list of random image pairs (image, label) from CIFAR-10,
        # each image of size torch.Size([1, 3, 384, 384])
        data = [dataiter.next() for _ in range(num_images)] 

        # Get a list of numpy BGR images, shape (384, 384, 3) 
        images = [x[0][0].cpu().detach().numpy().transpose((1, 2, 0))[...,::-1].copy() \
            for x in data]

        # Get latent vector for the images using InceptionV3
        model = videosum.InceptionFeatureExtractor('vector')
        feature_vectors = [model.get_latent_feature_vector(x) for x in images]

        # Plot PCA
        plotter = latentplot.Plotter(method='pca', width=1920, height=1080)
        plot = plotter.plot(images, feature_vectors)
        cv2.imsave(plot, 'test/data/pca.png')
    
    def test_tsne_plot(self, num_images=1000, width=1920, height=1080):
        # Prepare CIFAR-10 dataloader
        preproc_tf = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.Resize(size=384),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset = torchvision.datasets.CIFAR10(root=tempfile.gettempdir(), 
                                                train=True, download=True,
                                                transform=preproc_tf)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                 shuffle=True, num_workers=1)
        dataiter = iter(dataloader)

        # Get list of random image pairs (image, label) from CIFAR-10,
        # each image of size torch.Size([1, 3, 384, 384])
        data = [dataiter.next() for _ in range(num_images)] 

        # Get a list of numpy BGR images, shape (384, 384, 3) 
        images = [x[0][0].cpu().detach().numpy().transpose((1, 2, 0))[...,::-1].copy() \
            for x in data]

        # Get latent vector for the images using InceptionV3
        model = videosum.InceptionFeatureExtractor('vector')
        feature_vectors = [model.get_latent_feature_vector(x) for x in images]

        # Plot t-SNE
        plotter = latentplot.Plotter(method='tsne', width=1920, height=1080)
        plot = plotter.plot(images, feature_vectors)
        cv2.imsave(plot, 'test/data/tsne.png')
        

if __name__ == '__main__':
    unittest.main()
