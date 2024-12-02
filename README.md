# Fashion-MNIST CNN Classifier

A convolutional neural network (CNN) designed to classify images from the Fashion-MNIST dataset. The model achieves high accuracy on the validation set, demonstrating its ability to learn and generalize fashion-related image data effectively.


### Table of Contents

- [Description](#description)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Features](#features)
- [Setup and Installation](#setup-and-installation)
- [Implementation Details](#implementation-details)
- [Results](#results)



## Description

This project implements a deep learning model using Convolutional Neural Networks (CNNs) to classify the Fashion-MNIST dataset into 10 different categories of clothing and accessories. The project includes the dataset preprocessing pipeline, CNN architecture, training loop, validation performance evaluation, and visualizations of cost and accuracy over epochs.



## Technologies Used

The project is implemented using the following technologies:

* `Python:` Primary programming language for the project.
* `PyTorch:` Deep learning framework used for implementing and training the CNN.
* `Torchvision:` For accessing the Fashion-MNIST dataset and applying data transformations.
* `Matplotlib:` Visualization library for plotting training metrics.
* `Jupyter Notebook:` For interactive development and experimentation.




## Dataset

The project uses the Fashion-MNIST dataset, which consists of grayscale images of size 28x28 pixels, representing 10 classes of clothing and accessories. Each class contains 6,000 training images and 1,000 test images.

#### Classes:
  * T-shirt/top
  * Trouser
  * Pullover
  * Dress
  * Coat
  * Sandal
  * Shirt
  * Sneaker
  * Bag
  * Ankle boo




## Features

- A CNN with two convolutional layers, max-pooling, and one fully connected layer.
- Uses Cross-Entropy Loss for classification and Stochastic Gradient Descent (SGD) optimizer.
- Visualization of training cost and validation accuracy over epochs.
- Achieves validation accuracy of ~86% in 5 epochs.
  



## Setup and Installation

1. Clone the Repository:

```bash
git clone https://github.com/nickklos10/fashion-mnist-cnn-classifier.git
cd fashion-mnist-cnn-classifier
```

2. Create a Virtual Environment (Optional but Recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Dependencies: Install the required Python libraries using the provided requirements.txt:

```bash
pip install -r requirements.txt
```

4. Run the Training Script:

```bash
jupyter notebook Fash.ipynb
```




## Implementation Details

1. Data Preprocessing
   
  Transformations:
    * Images are resized to 16x16 pixels using transforms.Resize.
    * Converted to tensors using transforms.ToTensor.
    * The torchvision.datasets.FashionMNIST class is used to download and load the training and validation datasets.

2. Model Architecture

  The CNN consists of:
    * Convolutional Layer 1: 16 filters, kernel size 5x5, padding of 2.
    * Max Pooling Layer 1: Kernel size 2x2.
    * Convolutional Layer 2: 32 filters, kernel size 5x5, padding of 2.
    * Max Pooling Layer 2: Kernel size 2x2.
    * Fully Connected Layer: Maps the flattened feature maps to 10 output classes.

3. Loss Function
  * CrossEntropyLoss: Calculates the difference between predicted probabilities and actual class labels.

4. Optimizer
  * Stochastic Gradient Descent (SGD): Optimizer with a learning rate of 0.1.

5. Training and Validation
  * Training: Uses the training dataset to minimize the cost (Cross-Entropy Loss).
  * Validation: Evaluates the model's generalization by calculating accuracy on the validation dataset.
    Training runs for 5 epochs.




## Results

1. Performance Metrics:

* Validation Accuracy: ~86% after 5 epochs.
* Training Cost: Decreases steadily over epochs, indicating effective learning.
  
2. Training Visualization: The plot below illustrates the model's training process:

* The red line represents the decreasing training cost.
* The blue line represents the increasing validation accuracy.
