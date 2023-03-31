Description
-----------

Python package to plot the latent space of a set of images with different methods.


Install with pip
----------------

```bash
$ python3 -m pip install latentplot --user
```


Install from source
-------------------

```bash
$ git clone https://github.com/luiscarlosgph/latentplot.git
$ cd latentplot
$ python3 setup.py install --user
```


Exemplary code snippet
----------------------

```python
# List of images of shape (H, W, 3) and BGR
images = [ ... ]           

# List of vectors of shape (D,), where D is the vector dimension
feature_vectors = [ ... ]  

# List of integer class labels
labels = [ ... ]           

# Produce a BGR image containing a 2D plot of the latent space with t-SNE
plotter = latentplot.Plotter(method='tsne')                              
im_tsne = plotter.plot(images, feature_vectors, labels)  # Providing labels is optional
```

The `latentplot.Plotter` constructor parameters are:

| Parameter name | Description |
| -------------- | ----------- |
| method         | Method used to reduce the feature vectors to a 2D space. Available options: **pca**, **tsne**, **umap**. |
| width          | Desired output image width. Default is 15360 pixels (16K). |
| height         | Desired output image height. Default is 8640 pixels (16K). |
| dpi            | DPI for the output image. Default is 300. |
| cell_factor    | Proportion of the reduced latent space that each cell will occupy. Default is 0.01. |                         
| dark_mode      | Set it to False to have a white background with black font. Default is True. |          
| hide_axes      | Hide axes, ticks and marks. Default is True. |  
| **kwargs       | The rest of the arguments you pass will be forwarded to the dimensionality reduction method. |


Exemplary results
-----------------

* [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html): the size of the images in this dataset is 32x32 pixels.


<!---
* [PascalVOC](http://host.robots.ox.ac.uk/pascal/VOC):

   TODO

* [Cholec80](http://camma.u-strasbg.fr/datasets):

   TODO
-->


Author
------

Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com), 2023.


License
-------

This code repository is shared under an [MIT license](LICENSE).
