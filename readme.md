# VIN Character Classification

## About the project
The goal of this project is to make a convolutional neural network (CNN) model for classifying grayscale images with characters used in VIN codes. The characters are digits and most of capital latin letters (except from I, O and Q, which are not used in VIN codes due to their similarity with digits)

The project consists of the following files:

`train.py`: The file which loads the data from EMNIST dataset, preprocesses it, builds the neural network, trains it, evaluates its performance, and saves the model file to disk.

`inference.py`: The file which is using the trained model for making predictions on another character images and is printing results into a console.

`model.h5`: The saved model file.

`requirements.txt`: The requirements file

## Data
The neural network is using the balanced EMNIST (Extended Modified National Institute of Standards and Technology) dataset, downloaded from the extra_keras_datasets module. This dataset contains 125,000 grayscale images with handwritten characters (digits and both capital and lowercase latin letters). 

## Methods
1. The EMNIST dataset is loaded, divided into train and test datasets and preprocessed. The rows with characters not used in VIN are excluded from the dataset.

2. A convolutional neural network model is used for character classification. The model is made of two pairs of convolutional and pooling layers. Convolutional layers are extracting features from the image, pooling layers are downsampling matrix in two. The output is then flattened and passed through three dense layers, which are making predictions based on the extracted features from the earlier layers.

3. The model is trained. The loss function is the categorical cross-entropy, and the parameters are optimizes by Adam optimizer. The EarlyStopping callback is used for stopping training when a validation loss has stopped improving

4. After fitting, the model is saved to disk for later use.

5. The `inference.py` file loads the trained model and uses it for classification tasks.

## Accuracy
The model accuracy is 93.6% for test dataset and 93.4% for validation dataset.

## Usage Instructions
1. Make sure to install the libraries listed in the `requirements.txt` file
2. Open a terminal or a command prompt and navigate to the project directory.
3. Run the next code <code>python inference.py your_directory_with_pictures</code>

## Author information
This project is made by Mykhailo Olyzko (telegram: @dzhedzhalyk)

## Reference
Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. Retrieved from http://arxiv.org/abs/1702.05373
