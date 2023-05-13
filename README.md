# Landmark Classification & Tagging for Social Media
Two Convolutional Neural Network architectures for classifying landmark images. One built from the scratch and the other utilizes transfer learning. 

## Project Steps

1. **Creating a CNN architecture from Scratch to Classify Landmarks**
    - **Data Loading and Preprocessing**
        - The data_transforms dictionary contains train, valid and test keys. The values are instances of transforms.Compose. 
        - At the minimum, the 3 set of transforms contains a Resize(256) step, a crop step (RandomCrop for train and CenterCrop for valid and test), a ColorJitter step, a ToTensor step and finally a Normalize step (which uses the mean and std of the dataset). 
        - The ImageFolder instances for train, valid and test use the appropriate transform from the data_transforms dictionary (using the “transform” keyword of ImageFolder)
        - The data loaders for train, valid and test use the right ImageFolder instance and use the batch_size, sampler, and num_workers that are given in input to the function.
    - **Model Architecture**
        - The model architecture is a CNN with 5 convolutional layers and 2 linear layers. The output of the model is a logit for each class. The model uses dropout and batch normalization to reduce overfitting.
    - **Training and Testing the Model**
        - The model is trained first for 70 epochs with a learning rate of 0.001 and achieves a test accuracy of 42%
        - The model is reloaded from its checkpoint and trained for second time for 30 epochs with a learning rate of 0.0005 and achieves a test accuracy of 51%
    - **Saving and Loading the Model**
        - The trained model is saved as a checkpoint along with associated hyperparameters and the class_to_idx dictionary.
        - There is a function that successfully loads a checkpoint and rebuilds the model.
    -  **Inference for Classification**
        - There is a function that successfully makes a prediction for an image. The function returns the top 𝐾 most likely classes along with the probabilities. It takes a path to an image and a checkpoint, then returns the probabilities and classes.
    - **Sanity Checking with matplotlib**
        - There is a function that takes a path to an image and a model checkpoint, then plots the image and its predicted classes. The function uses matplotlib to plot the image and its 𝐾 most likely classes with actual flower names.  
<br>



2. **Using Transfer Learning to Classify Landmarks**
    - **Model Architecture**
        - The model architecture uses the ResNet18 pre-trained CNN model. 
        - All parameters of the loaded architecture are frozen, and a linear layer at the end has been added using the appropriate input features (as returned by the backbone), and the appropriate output features, as specified by the n_classes parameter
    - **Training and Testing the Model**
        - The model is trained first for 50 epochs with a learning rate of 0.001 and achieves a test accuracy of 74%.
    - **Saving and Loading the Model**
        - The trained model is saved as a checkpoint along with associated hyperparameters and the class_to_idx dictionary.
        - There is a function that successfully loads a checkpoint and rebuilds the model.
    -  **Inference for Classification**
        - There is a function that successfully makes a prediction for an image. The function returns the top 𝐾 most likely classes along with the probabilities. It takes a path to an image and a checkpoint, then returns the probabilities and classes.
    - **Sanity Checking with matplotlib**
        - There is a function that takes a path to an image and a model checkpoint, then plots the image and its predicted classes. The function uses matplotlib to plot the image and its 𝐾 most likely classes with actual flower names. 