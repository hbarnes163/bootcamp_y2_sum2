#!/usr/bin/env python
# coding: utf-8

# # Handwriting Digit Recognition
# - Based on the article by Amitrajit Bose 
# - (https://medium.com/@amitrajit_bose/handwritten-digit-mnist-pytorch-977b5338e627)
# - This code uses the MNIST digit dataset that is avaialbe through Pytorch

# ### Necessary Installs and Imports

# Python has several machine learning packages to enable the programmer to easily build machine learning models. You are familiar with Scikit-learn, however, as a result of the rise in interest in neural networks and deep learning, several new packages have been created to help developers. These include TensorFlow from Google and Pytorch from meta. Both packages offer extensive libraries to enable the developer to build interesting and innovative AI models.

# In[1]:


# Install Pytorch
get_ipython().run_line_magic('pip', 'install torch==2.6.0')
get_ipython().run_line_magic('pip', 'install torchvision==0.17.2')
get_ipython().run_line_magic('pip', 'install torchaudio==2.2.2')


# ## Please analyse and run the following commands

# In[2]:


# Import necessary packages
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
# Import packages to visaulise the data
import matplotlib.pyplot as plt
from time import time
import os

# These are the packages for Pytorch and numpy
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn
from torch import optim


# ### Download The Dataset & Define The Transforms
# - We use the transforms package to normalise and convert the data into tensor 
# - we pass this to the dataset. MNIST frunction 
# - If the data is not alrready downloaded the MNIST function will do it
# - We create test and traing data and load them using the DataLoader class

# In[3]:


# create directory to store data
my_dr = "\\MNIST\\"


# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
trainset = datasets.MNIST(my_dr, download=True, train=True, transform=transform)
valset = datasets.MNIST(my_dr, download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)


# ### Exploring The Data
# We get the next image out for the training loader and look at its shape and size

# In[5]:


dataiter = iter(trainloader)
images, labels = next(dataiter)
print(type(images))
print(images.shape)
print(labels.shape)


# In[7]:


# We show the image to see what it looks like
plt.imshow(images[0].numpy().squeeze(), cmap='gray_r');
plt.show()


# In[56]:


#we show 60 images to give an example of what we are training on
figure = plt.figure()
num_of_images = 60
for index in range(1, num_of_images + 1):
    plt.subplot(6, 10, index)
    plt.axis('off')
    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
    plt.show()


# ### Defining The Neural Network
# The picture below shows the Neural Network we will build

# ![](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/mlp_mnist.png)

# In[9]:


# Set device cuda for GPU if it's available otherwise run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ### We define the network using torch.nn.Sequential
# - The log softtmax function is the log of the softmax function 
# - When used on conjuntion with nn.NLLLoss() function This has the same effect as Cross-entropy

# In[11]:


# Layer details for the neural network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1)).to(device)

#Here we print out the network we just defined
print(model)


# ### We define the loos function as The negative log likelihood loss. 
# - It is useful to train a classification problem
# - https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html

# In[13]:


criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

logps = model(images)
loss = criterion(logps, labels)


# ### We print the weight before and after a backward
# This helps us check that our network is working

# In[15]:


print('Before backward pass: \n', model[0].weight.grad)

loss.backward()

print('After backward pass: \n', model[0].weight.grad)


# ### We use Stochastic gradient descent as the optimiser
# - You might want to try other optimsers to see how well they perform
# - Try Adam https://arxiv.org/abs/1412.6980
# - https://pytorch.org/docs/stable/generated/torch.optim.Adam.html

# In[17]:


# Optimizers require the parameters to optimize and a learning rate
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# In[19]:


print('Initial weights - ', model[0].weight)

images, labels = next(iter(trainloader))
images.resize_(64, 784)

# Clear the gradients, do this because gradients are accumulated
optimizer.zero_grad()

# Forward pass, then backward pass, then update weights
output = model(images)
loss = criterion(output, labels)
loss.backward()
print('Gradient -', model[0].weight.grad)


# In[21]:


# Take an update step and few the new weights
optimizer.step()
print('Updated weights - ', model[0].weight)


# ### Core Training Of Neural Network
# - The image is turned into one long 784 element vector. This is so the network can train on the images

# In[23]:


optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # Training pass
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        
        #This is where the model learns by backpropagating
        loss.backward()
        
        #And optimizes its weights here
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
print("\nTraining Time (in minutes) =",(time()-time0)/60)


# ### A function to show a classfied image to see if predicts correctly

# In[24]:


def view_classify(img, ps):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()


# ### Visually checking that the classfier is working

# In[25]:


images, labels = next(iter(valloader))

img = images[0].view(1, 784)
# Turn off gradients to speed up this part
with torch.no_grad():
    logps = model(img)

# Output of the network are log-probabilities, need to take exponential for probabilities
ps = torch.exp(logps)
probab = list(ps.numpy()[0])
print("Predicted Digit =", probab.index(max(probab)))
view_classify(img.view(1, 28, 28), ps)


# ### Model Evaluation
# We measure the percentage of correctly classifies images

# In[84]:


correct_count, all_count = 0, 0
for images,labels in valloader:
  for i in range(len(labels)):
    img = images[i].view(1, 784)
    # Turn off gradients to speed up this part
    with torch.no_grad():
        logps = model(img)

    # Output of the network are log-probabilities, need to take exponential for probabilities
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))


# ### Well done!

# ## Now try to build and train new ANNs using different structures and activation functions.

# In[ ]:


# Add your code here


# In[ ]:


# Add your code here


# In[ ]:


# Add your code here

