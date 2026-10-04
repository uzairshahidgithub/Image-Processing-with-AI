# Image Processing with Artificial Intelligence 
This task is application of AI on Image processing using Python. The aim is to modify the lfw_people datasets in sklearn in different ways with different learners, and give a conclusion on the most accurate learner. The learner in the initial datasets was replaced with some Supervised and Unsupervised machine learning models to determine this. 

## Introduction
Image processing is one of the key part of Artificial Intelligence. This can be applied in different ways ranging from identification of patterns in images, AI aided medical disgnosis, fraud detection, security and surveillance etc. 

## Getting Started
### Prerequisites
- Python 3.8+
### Installation
```bash
git clone https://github.com/Riabdulm/Image-Processing-with-AI.git
cd Image-Processing-with-AI
pip install -r requirements.txt
```
### Running Notebooks
Launch Jupyter and open any `.ipynb` file:
```bash
jupyter notebook
```

## Objective of the Project
This project aim is to fecth the lfw_people datasets contained in sklearn library by utilizing the following supervised and unsupervised machine learning models to modify the code in the faceRec.ipynb notebook - https://github.com/Riabdulm/Image-Processing-with-AI/blob/1fd6a7db945b5f7d7b0f2689a81925195a83f7cd/facerec.ipynb. 

i. Random Forest Classifier

ii. Multilayer Perceptron

iii. Deep Convolutional Neural Network

iv. KMeans Clustering.

The outcome of the application of these different models are given below with links to each of their codes. Conclusion is also given on the most accurate model.



## The results of the application of the Models:

i. **Random Forest Classifier** ---> Gave an accuracy of 63%. https://github.com/Riabdulm/Image-Processing-with-AI/blob/c67cd6203c09c5a856feb4f4753c8aebbbf066ed/random-forest-classifier.ipynb 

ii. **Multilayer Perceptron** ---> Gave an accuracy of 71%. https://github.com/Riabdulm/Image-Processing-with-AI/blob/c67cd6203c09c5a856feb4f4753c8aebbbf066ed/multilayer-perceptron.ipynb

iii. **Deep Convolutional Neural Network** ---> Gave the highest accuracy at 88%. https://github.com/Riabdulm/Image-Processing-with-AI/blob/c67cd6203c09c5a856feb4f4753c8aebbbf066ed/deep-convolutional-neural-network.ipynb

iv. **KMeans Clustering** ---> The unsupervised KMeans clustering gave the lowest accuracy at 18%. https://github.com/Riabdulm/Image-Processing-with-AI/blob/c67cd6203c09c5a856feb4f4753c8aebbbf066ed/kmeans-clustering.ipynb
## How to Use the Project for Study and Development

This repository is structured around Jupyter Notebooks (`.ipynb`), which serve as the optimal environment for interactive studying, prototyping, and tracking numerical changes. Using `.ipynb` files allows you to view the progression of the data, tweak hyperparameters in real-time, and visualize mathematical transformations step-by-step.

### Identifying Mathematical and Numerical Equations

Each model operates on specific mathematical foundations for image classification and clustering:

1. **Random Forest Classifier (`random-forest-classifier.ipynb`)**
   - **Methodology:** An ensemble learning method that constructs a multitude of decision trees during training.
   - **Mathematical Approach:** Based on Gini Impurity or Information Gain (Entropy). Gini Impurity is defined as `Gini = 1 - sum(p_i^2)`, where `p_i` is the probability of an object being classified to a particular class.

2. **Multilayer Perceptron (`multilayer-perceptron.ipynb`)**
   - **Methodology:** A class of feedforward artificial neural network (ANN) consisting of at least three layers of nodes.
   - **Mathematical Approach:** Utilizes backpropagation for training. The weight update rule minimizes the cost function (e.g., Cross-Entropy loss) using gradient descent.

3. **Deep Convolutional Neural Network (`deep-convolutional-neural-network.ipynb`)**
   - **Methodology:** A specialized type of neural network that uses convolution operations, ideal for spatial data like images.
   - **Mathematical Approach:** The convolution operation involves sliding a kernel matrix over the input image matrix to produce feature maps, capturing spatial hierarchies.

4. **KMeans Clustering (`kmeans-clustering.ipynb`)**
   - **Methodology:** An unsupervised learning algorithm that partitions data into `k` clusters based on feature similarity.
   - **Mathematical Approach:** Minimizes the within-cluster sum of squares (WCSS) by iteratively assigning data points to the nearest cluster centroid and updating the centroids.

By stepping through the cells in each notebook, you can observe how these equations translate into code, monitor the numerical outputs, inspect matrix transformations, and compare their performance interactively.


## Conclusion

Convolutional Neural Network (CNN) gave the best accuracy as they are more suited to images because neural network exploits geometric regularities that are found in images. On the other hand, KMeans clustering produced the least accuracy as unsupervised machine learning has no prior information about the dataset.


