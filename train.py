import os
from PIL import Image
import numpy as np


def load_dataset(target_path):
    data_path = target_path

    classes = sorted(os.listdir(data_path))
    classes = {i:item for i, item in enumerate(classes)}
    classes_rev = {value:key for key,value in classes.items()}
    print(classes)
    
    data, labels = [], []

    for key in classes_rev.keys() :
        img_path = os.path.join(data_path, key)
        images = os.listdir(img_path)
        for image in images:
            img = Image.open(os.path.join(img_path, image)).convert("RGB").resize((64,64))
            data.append(np.array(img).astype('float32')/255)
            labels.append(classes_rev[key])

    data = np.array(data)
    labels = np.array(labels)
    
    print(data.shape)
    print(labels.shape)
    

if __name__ == "__main__":
    data_path = "./data/Train"
    
    load_dataset(data_path)
    

    