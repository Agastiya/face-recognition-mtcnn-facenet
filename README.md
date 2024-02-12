<h1 align="center">
  <br>
  <img src="https://learn.g2.com/hubfs/G2CM_FI454_Learn_Article_Images_%5BFacial_recognition%5D_V1a-1.png" alt="Markdownify" width="200">
  <br>
   Face Recognition with MTCNN and FaceNet
  <br>
</h1>

## About
This face recognition project is the project that I use as my thesis. This face recognition program uses the MTCNN method for face detection and FaceNet for face recognition. In short, after the face detection process using MTCNN, the FaceNet algorithm produces a unique vector representation for each face obtained. The process will continue by comparing the vector with a previously encoded face database to identify and verify the face.

## Getting Started
We will prepare this project, starting from preparing the dataset, installation, and running the code.

### Dependencies
* Python 3.11.6

### Preparing Dataset
You should prepare a dataset of about 5 to 10 photos for each person. The more photos you use, the better the results will be. The folder structure that can be prepared is as follows :
    
    Messi
        - photo1.jpg
        - photo2.jpg
        - ...

    Ronaldo
        - photo1.jpg
        - photo2.jpg
        - ...

### Installation
Several stages that need to be prepared at this stage are as follows :

* Clone this repository

        git clone https://github.com/Agastiya/face-recognition-mtcnn-facenet.git

* Create a directory

        mkdir processing
        mkdir processing/model
        mkdir processing/resources
        mkdir processing/resources/datasets
        mkdir processing/resources/split_dataset

* Install the required libraries

        pip install -r requirement.txt

* Put all your datasets into the **processing/resources/datasets** folder

### Executing Program
For the first time, we need to process the dataset. this section includes split and train datasets, so we will get a model that we use to identify face.

    python processing.py

This process will take time depending on the number of datasets used. After that you can run the program use this command :

    python app.py

Make sure the program runs well, you can open it in a browser at http://127.0.0.1:8090/. Upload an image from **processing/resources/split_dataset/test** or another image not from the dataset for testing.

## Acknowledgments
* [FaceNet Paper](https://www.researchgate.net/publication/273471270_FaceNet_A_Unified_Embedding_for_Face_Recognition_and_Clustering)
* [Article](https://medium.com/@culuma/face-recognition-with-facenet-and-mtcnn-11e77240adb6)
* [MTCNN](https://github.com/ipazc/mtcnn)
* [Facenet](https://github.com/davidsandberg/facenet)