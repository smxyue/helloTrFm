import os
import pickle
import tarfile
import matplotlib.pyplot as plt
import numpy as np
import pickle
import ssl
import urllib.request
ssl._create_default_https_context = ssl._create_unverified_context
def unpickle(file):
    
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        
    return dict[b'data'], dict[b'labels']

train_files = ['data/cifar-10-batches-py/data_batch_1',
               'data/cifar-10-batches-py/data_batch_2',
               'data/cifar-10-batches-py/data_batch_3',
               'data/cifar-10-batches-py/data_batch_4',
               'data/cifar-10-batches-py/data_batch_5']
test_file = 'data/cifar-10-batches-py/test_batch'
def is_data_unpacked():
    for file in train_files + [test_file]:
        if not os.path.exists(file):
            return False
    if not os.path.exists(test_file):
        return False
    return True
def get_data():
    if not is_data_unpacked():
        if not os.path.exists('data/cifar-10-python.tar.gz'):
            print('Downloading data...')
            import urllib.request
            url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
            urllib.request.urlretrieve(url, 'data/cifar-10-python.tar.gz')
        print('Unpacking data...')
        with tarfile.open('data/cifar-10-python.tar.gz', 'r:gz') as tar:
            tar.extractall('data')
    train_datas = []
    train_labels = []
    for file in train_files:
        datas,lables = unpickle(file)
        train_datas.append(datas)
        train_labels.append(lables)
        
    train_datas=np.concatenate(train_datas,axis=0)
    train_datas=train_datas/255.0
    train_labels=np.concatenate(train_labels,axis=0)
    test_datas,test_labels = unpickle(test_file)
    test_datas=test_datas/255.0
    return (train_datas, train_labels), (test_datas, test_labels)
def get_categories():
    """
    Load CIFAR-10 category names from the meta file.
    
    Returns:
        list: List of 10 category names
    """
    meta_file = 'data/cifar-10-batches-py/batches.meta'
    
    # Check if meta file exists, if not try to unpack data
    if not os.path.exists(meta_file):
        # Trigger data unpacking if needed
        get_data()
    
    with open(meta_file, 'rb') as f:
        meta_data = pickle.load(f, encoding='bytes')
        categories = [label.decode('utf-8') for label in meta_data[b'label_names']]
    
    return categories
def show_sample():
    categories = get_categories()
    (train_datas, train_labels),(test_datas,test_labels) = get_data()
    fig,axes=plt.subplots(6,6, figsize=(10,10))
    axes=axes.ravel()
    startIndex = np.random.randint(0, len(train_datas[0]) - 36)
    for i in range(36):
        img = train_datas[startIndex + i].reshape(3, 32, 32).transpose(1, 2, 0)
        axes[i].imshow(img,vmin=0,vmax=0.2)
        axes[i].set_title(categories[train_labels[startIndex+i]])
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
    img =train_datas[startIndex].reshape(3, 32, 32).transpose(1, 2, 0)
    print('image shape', img.shape)
    print('Image sample:', img)
if __name__ == '__main__':
    show_sample()
    