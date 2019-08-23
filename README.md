# comp540_finalproject


### Overview
The goal of this project is to try to build a better model to automate the recognition of nuclei in images(Data science bowl 2018). Some nuclei in images are hard even for human to spot. A machine learning model that spot nuclei can help researchers improve their results. With a relativity small training data set available for this competition, overfitting is a problem on training a deep neural network. Various image augmentations are tested and applied to images and mask. We tried two different approaches to this problem, general image segmentation and instance base segmentation. An article from Ronneberger proposed [1] a fully convolutional network architecture that performs well on small data set named U-net. A slightly simplified u-net was built for the segmentation problem. Mask-RCNN is a RCNN extension network architecture proposed by He[3] have outperformed other model on Imagenet data set. A similar network architecture was trained and compare to our U-net model. 

training flow:
1. Split data set into three groups with kmeans.
2. Split training data into training set and validation set. Each group has to split evenly into training set and validation set.
3. Tune parameters with validation set.
4. Predict with the best model.
5. Run several run to build consistance result.

See final report for more inforamtion.

Reference

[1] Olaf Ronneberger, Philipp Fischer and Thomas Brox, U-Net: Convolutional Networks for Biomedical Image Segmentation, MICCAI 2015
