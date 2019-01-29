# FaceQnet
FaceQnet: Quality Assessment for Face Recognition based on Deep Learning

This repository contains the DNN FaceQnet presented in the paper: "FaceQnet: Quality Assessment for Face Recognition based on Deep Learning".

FaceQnet is a No-Reference, end-to-end Quality Assessment (QA) system for face recognition based on deep learning. 
The system consists of a Convolutional Neural Network that is able to predict the suitability of a specific input image for face recognition purposes. 
The training of FaceQnet is done using the VGGFace2 database.

-- Configuring environment in Windows:

1)Installing Conda: https://conda.io/docs/user-guide/install/windows.html

  Update Conda in the default environment:

    conda update conda
    conda upgrade --all

  Create a new environment:

    conda create -n [env-name]

  Activate the environment:

    source activate [env-name]

2) Installing dependencies in your environment:

  Install Tensorflow and all its dependencies: 
    
    pip install tensorflow
    
  Install Keras:
  
    pip install keras
    
  Install OpenCV:

    conda install -c conda-forge opencv
  
 3) If you want to use a CUDA compatible GPU for faster predictions:
  
   You will need CUDA installed in your computer: https://docs.nvidia.com/deeplearning/sdk/cudnn-install/
  
   Then, install the GPU version of Tensorflow:
    
    pip install tensorflow-gpu
  
-- Using FaceQnet for predicting scores:

  1) Download or clone the repository 
  2) Dowload the FaceQnet pretrained model (.h5 file) and place it in the /src folder.
  3) Edit and run the FaceQNet_obtainscores_Keras.py script.





