from utils.utils import face_detection, feature_extraction
import os
import splitfolders
import shutil
import numpy as np

def split_dataset():
    
    # Split the dataset into train and test folders
    splitfolders.ratio('processing/resources/datasets',
                       output="processing/resources/split_dataset",
                       seed=1337,
                       ratio=(.7, 0, .3),
                       group_prefix=None,
                       move=False)
    
    # Remove val split dataset
    shutil.rmtree('processing/resources/split_dataset/val')


def train(folder, file):

    labels = []
    embeddings = []
    for subdir in os.listdir(folder):
        label = subdir
        path = folder + subdir + '/'
        for imageFile in os.listdir(path):
            image = path + imageFile
            imageDetection = face_detection(image)
            if imageDetection is None:
                continue
            embedding = feature_extraction(imageDetection)
            labels.append(label)
            embeddings.append(embedding)

    labels = np.array(labels)
    embeddings = np.array(embeddings)
    np.savez_compressed(file, embeddings, labels)


print('Split Dataset Start ...')
split_dataset()
print('Split Dataset Finish ...')

print('\n---------------------------------\n')

print('Train Model Start ...')
train('processing/resources/split_dataset/train/', 'processing/model/model.npz')
print('Train Model Finish ...')
