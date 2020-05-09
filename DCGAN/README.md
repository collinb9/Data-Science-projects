# DCGAN

I trained a DCGAN on the [Stanford AI lab car dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html), which contains 16,185 images of cars. I used the architecture introduced in [Radford et al. 2016](https://arxiv.org/pdf/1511.06434.pdf), and trained the model using the algorithm outlined in [Goodfellow et al. 2014](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf). The model was trained for 100 epochs through the training images. Some of the generated images can be seen below.

![](https://github.com/collinb9/Data-Science-projects/blob/master/DCGAN/training_images.png "Training images")

![](https://github.com/collinb9/Data-Science-projects/blob/master/DCGAN/generated_images/generated_images_99_epochs.png "Generated images after 100 epochs")

The generator is capable of producing some pretty convincing images, such as the one on the bottom right. 

![](https://github.com/collinb9/Data-Science-projects/blob/master/DCGAN/generated_images/generated_images.gif "Generated images")
