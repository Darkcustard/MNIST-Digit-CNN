# Usage

1. Make sure you have the dependancies found under "requirements.txt"
2. Run Query.py using python.
3. Draw and manipulate the drawspace using the GUI.
4. Press the query button and the AI will output a digit to the console


# Devlog

I created a jupyter notbook to handle the classifier built using Keras. The model Takes in a 28x28x1 image and outputs
an 9x1 array of probabilities for the corrisponding digits from 0-9.

The classifier was trained on the MNIST-handwritten digit classification dataset and uses functions to convert
betweet decimal and the 9x1 array:     3 = [0,0,0,1,0,0,0,0,0,0]

the test accuracy after 5 epochs was ~ 98.1% and I have not yet optimized the hyperparams. 


I then used my 'pygame_menu' library to build a GUI where users can draw digits then query the classifier which returns
its output to the console.

The main issue I Faced was not studying my Dataset enough as I assumed it was from 1-9 ommitting the zero which
introduced unforseen training loss etc. Once I found the issue, I modified associated functions and it worked
flawlessly.
