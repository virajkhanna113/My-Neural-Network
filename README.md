# Neural Network
# Data
Neural networks attempt to copy the human learning technique, Trial and Error. To do this, we will create something like a set of digital flashcards. Our artificial brains will attempt to guess what kind of clothing we are showing it with a flashcard, then we will give it the answer, helping the computer learn from its successes and mistakes.
![image](https://github.com/user-attachments/assets/8bd54e7e-5a32-4682-9211-a3cad3e2fd78)
The study data is often called the training dataset and the quiz data is often called the validation dataset. As Fashion MNIST is a popular dataset, it is already included with the TensorFlow library. Let's load it into our coding environment and take a look at it.

Here is the table for reference:

| Label | Description |
| --- | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

# Neuron
Neurons are the fundamental building blocks to a neural network. Just like how biological neurons send an electrical impulse under specific stimuli, artificial neural networks similarly result in a numerical output with a given numerical input.

We can break down building a neuron into 3 steps:

Defining the architecture
Intiating training
Evaluating the model

# Defining the architecture
![image](https://github.com/user-attachments/assets/d18f1969-67a3-43fc-b937-84548e730833)
Biological neurons transmit information with a mechanism similar to Morse Code. It receives electrical signals through the dendrites, and under the right conditions, sends an electrical impulse down the axon and out through the terminals.

It is theorized the sequence and timing of these impulses play a large part of how information travels through the brain. Most artificial neural networks have yet to capture this timing aspect of biological neurons, and instead emulate the phenomenon with simpler mathematical formulas.

# The Math
Computers are built with discrete 0s and 1s whereas humans and animals are built on more continuous building blocks. Because of this, some of the first neurons attempted to mimic biological neurons with a linear regression function: y = mx + b. The x is like information coming in through the dendrites and the y is like the output through the terminals. As the computer guesses more and more answers to the questions we present it, it will update its variables (m and b) to better fit the line to the data it has seen.

Neurons are often exposed to multivariate data. We're going to build a neuron that takes each pixel value (which is between 0 and 255), and assign it a weight, which is equivalent to our m. Data scientists often express this weight as w. For example, the first pixel will have a weight of w0, the second will have a weight of w1, and so on. Our full equation becomes y = w0x0 + w1x1 + w2x2 + ... + b.

The output of y = mx + b is a number, but here, we're trying to classify different articles of clothing. How might we convert numbers into categories?

Here is a simple approach: we can make ten neurons, one for each article of clothing. If the neuron assigned to "Trousers" (label #1), has the highest output compared to the other neurons, the model will guess "Trousers" for the given input image.

Keras, a deep learning framework that has been integrated into TensorFlow, makes such a model easy to build. We will use the Sequential API, which allows us to stack layers, the list of operations we will be applying to our data as it is fed through the network.

In the below model, we have two layers:

**Flatten** - Converts multidimensional data into 1 dimensional data (ex: a list of lists into a single list).
**Dense** - A "row" of neurons. Each neuron has a weight (w) for each input. In the example below, we use the number 10 to place ten neurons.
We will also define an input_shape which is the dimensions of our data. In this case, our 28x28 pixels for each image.

In academic papers, models are often represented like the picture below. In practice, modern neural networks are so large, it's impractical to graph them in this way. The below is a fraction of our entire model. There are 10 neurons on the bottom representing each of our ten classes, and 28 input nodes on the top, representing a row of our pixels. In reality, the top layer is 28 times bigger!

Each circle represents a neuron or an input, and each line represents a weight.
![image](https://github.com/user-attachments/assets/1ca79364-6f65-4c25-b1c1-0a45d161661b)

# Initiate Training
We have a model setup, but how does it learn? Just like how students are scored when they take a test, we need to give the model a function to grade its performance. Such a function is called the loss function.

In this case, we're going to use a type of function specific to classification called SparseCategoricalCrossentropy:

Sparse - for this function, it refers to how our label is an integer index for our categories
Categorical - this function was made for classification
Cross-entropy - the more confident our model is when it makes an incorrect guess, the worse its score will be. If a model is 100% confident when it is wrong, it will have a score of negative infinity!
from_logits - the linear output will be transformed into a probability which can be interpreted as the model's confidence that a particular category is the correct one for the given input.

This type of loss function works well for our case because it grades each of the neurons simultaneously. If all of our neurons give a strong signal that they're the correct label, we need a way to tell them that they can't all be right.

For us humans, we can add additional metrics to monitor how well our model is learning. For instance, maybe the loss is low, but what if the accuracy is not high?

# Evaluating the model
Now the moment of truth! The below fit method will both help our model study and quiz it.

An epoch is one review of the training dataset. Just like how school students might need to review a flashcard multiple times before the concept "clicks", the same is true of our models.

After each epoch, the model is quizzed with the validation data.

It only has 10 neurons to work with. Us humans have billions!
The accuracy should be around 80%, although there is some random variation based on how the flashcards are shuffled and the random value of the weights that were initiated.

# Prediction
Time to graduate our model and let it enter the real world. We can use the predict method to see the output of our model on a set of images, regardless of if they were in the original datasets or not.

Please note, Keras expects a batch, or multiple datapoints, when making a prediction. To make a prediction on a single point of data, it should be converted to a batch of one datapoint.

# Conclusion
While this model does significantly better than random guessing, it has a way to go before it can beat humans at recognizing clothing.
