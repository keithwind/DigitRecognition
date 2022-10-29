import numpy as np

def training_data_loader():
    images_path = 'training_data/images.ubyte' # Images Data Location
    labels_path = 'training_data/labels.ubyte' # Labels Data Location

    images_data = []
    labels_data = []

    with open(images_path,'rb') as f: # rb -> read in binary 
        f.read(16) #useless magic number and imazge specs
        for i in range(60000): # 60,000 images
            images_data.append(list(f.read(784))) #read next 784 bytes of single image data
    
    
    with open(labels_path,'rb') as f:
        f.read(8)
        for i in range(60000):
            labels_data.append(int.from_bytes(f.read(1),"big")) #read 1 byte of label data

    images_data = [np.reshape(x,(784,1))/256.0 for x in images_data] # convert image data to numpy arrays of size 784x1
    labels_data = [vectorize(x) for x in labels_data] # vectorize labels data as 10x1 array

    return (images_data,labels_data)

def test_data_loader():
    images_path = 'test_data/images_t.ubyte' # Images Data Location
    labels_path = 'test_data/labels_t.ubyte' # Labels Data Location

    images_data = []
    labels_data = []

    with open(images_path,'rb') as f: # rb -> read in binary 
        f.read(16) #useless magic number and imazge specs
        for i in range(10000): # 60,000 images
            images_data.append(list(f.read(784))) #read next 784 bytes of single image data
    
    
    with open(labels_path,'rb') as f:
        f.read(8)
        for i in range(10000):
            labels_data.append(int.from_bytes(f.read(1),"big")) #read 1 byte of label data

    images_data = [np.reshape(x,(784,1))/256.0 for x in images_data] # convert image data to numpy arrays of size 784x1
    labels_data = [vectorize(x) for x in labels_data] # vectorize labels data as 10x1 array

    return (images_data,labels_data)
def vectorize(x):
    vec = np.zeros((10,1))
    vec[x] = [1]
    return vec