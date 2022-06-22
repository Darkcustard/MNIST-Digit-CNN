# Usage

1. Make sure you have the dependancies found under requirements.txt - "pip -r requirements.txt"
2. Run Query.py using python.
3. Draw and manipulate the drawspace using the GUI.
4. Press the query button and the AI will output a digit to the console


# Approach

I created a jupyter notbook to handle the classifier built using Keras. The model Takes in a 28x28x1 image and outputs
an 10x1 array of probabilities for the corrisponding digits from 0-9.

The model makes use of convolutional layers and max pooling for feature extraction with a small amount of dropout to prevent
over fitting.

The classifier was trained on the MNIST-handwritten digit classification dataset.
The Dataset is split into train and test by default but I split test into validate and test
for the sake of optimising hyperparameters later.

I then used my 'pygame_menu' library to build a GUI where users can draw digits then query the classifier which returns
its output to the console.


# Changelog

*Wednesday, 22 June 2022*

* Scaled input image pixel values between 0 and 1.
* Added Dropout layer of 0.1 to combat overfitting.
* Caught logical error where 0 was ommitted from the possible responses.
* Increased filters in secondary Conv2D layer from 16 - 32.
* Split Test set into Test and Validate.
* Changed Conv2D sizes from 2x2 to 3x3.
