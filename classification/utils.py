import numpy as np
import requests

from os.path import join
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from tensorflow.keras.applications import xception

from mpl_toolkits.axes_grid1 import ImageGrid

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 数据集切分
def split_train_val(labels, num_classes, seed=1234):
    selected_breed_list = list(labels.groupby('breed').count().sort_values(by='id', ascending=False).head(num_classes).index)
    labels = labels.loc[labels['breed'].isin(selected_breed_list)].reset_index()
    labels['target'] = 1
    labels.loc[:, 'rank'] = labels[['breed', 'index']].groupby('breed')['index'].rank()
    labels_pivot = labels.pivot('id', 'breed', 'target').reset_index().fillna(0)
    np.random.seed(seed=seed)
    rnd = np.random.random(len(labels))
    train_idx = rnd < 0.8
    valid_idx = rnd >= 0.8
    y_train = labels_pivot[selected_breed_list].values
    ytr = y_train[train_idx]
    yv = y_train[valid_idx]
    return (train_idx, valid_idx, ytr, yv, labels, selected_breed_list)

# 读取图像
def read_img(img_id, data_dir, train_or_test, size):
    """Read and resize image.
    # Arguments
        img_id: string
        train_or_test: string 'train' or 'test'.
        size: resize the original image.
    # Returns
        Image as numpy array.
    """
    img = image.load_img(join(data_dir, train_or_test, '%s.jpg' % img_id), target_size=size)
    img = image.img_to_array(img)
    return img

# 图像显示
def show_images(num_classes, labels, data_dir):
    j = int(np.sqrt(num_classes))
    i = int(np.ceil(1. * num_classes / j))
    fig = plt.figure(1, figsize=(16, 16))
    grid = ImageGrid(fig, 111, nrows_ncols=(i, j), axes_pad=0.05)
    for i, (img_id, breed) in enumerate(labels.loc[labels['rank'] == 1, ['id', 'breed']].values):
        ax = grid[i]
        img = read_img(img_id, data_dir, 'train', (224, 224))
        ax.imshow(img / 255.)
        ax.text(10, 200, 'LABEL: %s' % breed, color='k', backgroundcolor='w', alpha=0.8)
        ax.axis('off')
    plt.show()

# 数据增强
def data_augmentation_example(input_path, count):
    # load image to array
    image = img_to_array(load_img(input_path))

    # reshape to array rank 4
    image = image.reshape((1,) + image.shape)

    # let's create infinite flow of images
    train_datagen = ImageDataGenerator(rotation_range=45,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.25,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
    images_flow = train_datagen.flow(image, batch_size=1)

    plt.figure(figsize=(9,9))
    for idx, new_images in enumerate(images_flow):
        if idx < count:
            plt.subplot(330 + 1 + idx)
            new_image = array_to_img(new_images[0], scale=True)
            plt.imshow(new_image)
            plt.axis('off')
        else:
            plt.show()
            break
            
 # 预测，即：给定测试图像，对其类别进行预测
def prediction_from_url(url, model, selected_breed_list):
    test_image_path = '/tmp/test.jpg'
    response = requests.get(url)
    if response.status_code == 200:
        with open(test_image_path, 'wb') as f:
            f.write(response.content)
    img = read_img('test', '/', 'tmp', (224, 224))
    x = xception.preprocess_input(np.expand_dims(img.copy(), axis=0))
    preds = model.predict(x)
    pred_idx = np.argmax(preds, axis=1)[0]
    plt.title('Prediction: %s (%.2f)' % (selected_breed_list[pred_idx] , preds[0][pred_idx]*100))
    plt.imshow(img / 255.)
    plt.axis('off')
    plt.show()        
    


def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig
