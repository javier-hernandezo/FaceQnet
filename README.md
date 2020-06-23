# FaceQnet
FaceQnet: Quality Assessment for Face Recognition based on Deep Learning

This repository currently contains two different versions of FaceQnet:

- FaceQnet v0: the version presented in the paper: <a href="https://arxiv.org/abs/1904.01740" rel="nofollow">"FaceQnet: Quality Assessment for Face Recognition based on Deep Learning"</a>.

- FaceQnet v1: the version presented in the paper: <a href="https://arxiv.org/abs/2006.03298" rel="nofollow">"Biometric Quality: Review and Application to Face Recognition with FaceQnet"</a>. This is the most recent version of FaceQnet.


FaceQnet is a No-Reference, end-to-end Quality Assessment (QA) system for face recognition based on deep learning. 
The system consists of a Convolutional Neural Network that is able to predict the suitability of a specific input image for face recognition purposes. 
The training of FaceQnet is done using the VGGFace2 database.

-- Configuring environment in Windows:

1) Installing Conda: https://conda.io/projects/conda/en/latest/user-guide/install/windows.html

  Update Conda in the default environment:

    conda update conda
    conda upgrade --all

  Create a new environment:

    conda create -n [env-name]

  Activate the environment:

    conda activate [env-name]

2) Installing dependencies in your environment:

  Install Tensorflow and all its dependencies: 
    
    pip install tensorflow
    
  Install Keras:
  
    pip install keras
    
  Install OpenCV:

    conda install -c conda-forge opencv
  
 3) If you want to use a CUDA compatible GPU for faster predictions:
  
   You will need CUDA and the Nvidia drivers installed in your computer: https://docs.nvidia.com/deeplearning/sdk/cudnn-install/
  
   Then, install the GPU version of Tensorflow:
    
    pip install tensorflow-gpu
  
-- Using FaceQnet for predicting scores:

  1) Download or clone the repository. 
  2) Due to the size of the video example, please download one of the the FaceQnet pretrained models and place the downloaded .h5 file it in the /src folder:  
  
  - <a href="https://github.com/uam-biometrics/FaceQnet/releases/download/v0/FaceQnet.h5" rel="nofollow">FaceQnet v0 pretrained model</a> 
  
  
  - <a href="https://github.com/uam-biometrics/FaceQnet/releases/download/v1.0/FaceQnet_v1.h5" rel="nofollow">FaceQnet v1 pretrained model</a> 
  
  
  3) Edit and run the FaceQNet_obtainscores_Keras.py script.
     - You will need to change the folder from which the script will try to charge the face images. It is src/Samples_cropped by default. 
     - The best results will be obtained when the input images have been cropped just to the zone of the detected face. In our experiments we have used the MTCNN face detector from <a href="https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html" rel="nofollow">here</a>, but other detector can be used.
     - FaceQnet will ouput a quality score for each input image. All the scores will are saved in a .txt file into the src folder. This file contain each filename with its associated quality metric.





