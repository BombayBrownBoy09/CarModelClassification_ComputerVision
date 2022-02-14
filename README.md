# Comprehensive Cars Fine-Grain Car Model Classification
**Project by Abhilash Sarnad, Bhargav Shetgaonkar, and Leo Corelli for Duke AIPI 540 Module 1**
<p align="center"><img align="center" width="800px" src="https://github.com/leocorelli/ComputerVisionProject/blob/main/images/CompCars.png"></p>


<a name="proj-stat"></a>
## 1. Problem statement
The objective of this project is to train a machine learning model to recognize the `model` of cars using images.

<a name="proj-struc"></a>
## 2. Project Structure
The project data and codes are arranged in the following manner:

```
├── README.md               <- description of project and how to set up and run it
├── requirements.txt        <- requirements file to document dependencies
├── Makefile [OPTIONAL]     <- setup and run project from command line
├── main.ipynb]             <- contains main notebook modeled using InceptionV3
├── notebooks               <- contains LogisticRegression and ResNet152 notebooks
├── .gitignore              <- git ignore file
```

_Data_: <br>
the `data` folder is not a part of this git project as it was heavy. The same can be downloaded from below link:
1) Download data [here](https://duke.box.com/s/jru8ykcob3xfhkzm3hemk4s76hxnxbaf) 
    - **Training data:** run ```unzip train_real.zip``` on command line
    - **Validation/testing data:** run ```unzip test_real.zip``` on command line
2) Download trained models [here](https://duke.box.com/s/pd98gb8064ylm8qs13khil4ndd3865w7)

```sh
http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/
```

<a name="exp"></a>
## 3. Experimentation
We tried two approaches here:

**Approach 1(Non deep learning model - Logistic regression):**
The baseline model is created using Logistic regression. The Logistic regression model is created using one linear layer in pytorch and the predictions are generated using Softmax. The accuracy with this approach is very less as shown below
<p align="center"><img align="center" width="800px" src="https://github.com/leocorelli/ComputerVisionProject/blob/main/images/logisticRegression.png"></p>

**Approach 2(With Resnet-152):**
For Deep learning approach we used pretrained Convolutional neural networks. The model is trained with Resnet-152 with no LR scheduler, Resnet-152 with OneCycle LR scheduler and Resnet-152 with Adam. The accuracy for all three models is as shown below
<p align="center"><img align="center" width="800px" src="https://github.com/leocorelli/ComputerVisionProject/blob/main/images/resNetNoLR.png"></p>
<p align="center"><img align="center" width="800px" src="https://github.com/leocorelli/ComputerVisionProject/blob/main/images/resnet1Cycle.png"></p>
<p align="center"><img align="center" width="800px" src="https://github.com/leocorelli/ComputerVisionProject/blob/main/images/resnetAdam.png"></p>

**Approach 3(With Inception V3):**
The model is trained with Incpetion V3. The training accuracy with InceptionV3 is 93.57% and test accuracy is 84.59%
<p align="center"><img align="center" width="800px" src="https://github.com/leocorelli/ComputerVisionProject/blob/main/images/Inceptionv3.png"></p>

## 4. Results

Our model is able to predict the model of the cars with a accuracy of 84.59%.
<p align="center"><img align="center" width="800px" src="https://github.com/leocorelli/ComputerVisionProject/blob/main/images/output.png"></p>


## 5. Notebooks
1) Non-Deep learning(Logistic regression) [here](https://github.com/leocorelli/ComputerVisionProject/blob/main/notebooks/LogisticRegression_Car_Prediction.ipynb) 
2) Notebook with Resnet models [here](https://github.com/leocorelli/ComputerVisionProject/blob/main/notebooks/resnet.ipynb)
3) Notebook with InceptionV3 model [here](https://github.com/leocorelli/ComputerVisionProject/blob/main/main.ipynb) 
