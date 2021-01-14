import sys
sys.path.append('..')
import requests
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import os
from os.path import join
from os import listdir, makedirs

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from tensorflow.keras.applications import xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.models import Model

from mpl_toolkits.axes_grid1 import ImageGrid
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from utils import split_train_val, show_images, read_img, print_confusion_matrix


BATCH_SIZE = 16
EPOCHS = 3


# 数据集比较大，这里只保留10种狗
NUM_CLASSES = 10 

data_dir = '/training/colin/Github/Computer-Vision/data/dog-breed-identification'
labels = pd.read_csv(join(data_dir, 'labels.csv')) # EDIT WITH YOUR LABELS FILE NAME
# print("Total number of images in the dataset: {}".format(len(listdir(join(data_dir, 'train')))))
# print("Top {} labels (sorted by number of samples)".format(NUM_CLASSES))
# print(labels
#  .groupby("breed")
#  .count()
#  .sort_values("id", ascending=False)
#  .head(NUM_CLASSES)
# )


SEED = 2019

def get_model():
    # 下载预训练的xception模型，注意到include_top=False表示没有1000分类的head
    base_model = xception.Xception(weights='imagenet', include_top=False)

    # 在基模型的基础上添加几层
    x = base_model.output
    # BN层
    x = BatchNormalization()(x)
    # 全局池化
    x = GlobalAveragePooling2D()(x)
    # 添加dropout提高泛化
    x = Dropout(0.5)(x)
    # 添加全连接增加表达能力
    x = Dense(1024, activation='relu')(x)
    # 继续dropout
    x = Dropout(0.5)(x)
    # 分成NUM_CLASSES这么多类
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    # 给定“头”和“尾”以定义好整个模型
    model = Model(inputs=base_model.input, outputs=predictions)

    # 我们把前面的层次固定住(学习率变为0)
    for layer in base_model.layers:
        layer.trainable = False

    # 指定优化器与损失函数，进行编译
    optimizer = RMSprop(lr=0.001, rho=0.9)
    model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=["accuracy"])
    # 输出模型结构
    # model.summary()
    return model


if __name__ == "__main__":
    # Data Loader
    train_datagen = ImageDataGenerator(rotation_range=45,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.25,
                                    horizontal_flip=True,
                                    fill_mode='nearest')

    test_datagen = ImageDataGenerator()

    # 切分数据
    (train_idx, valid_idx, ytr, yv, labels, selected_breed_list) = split_train_val(labels, NUM_CLASSES, seed=SEED)

    INPUT_SIZE = 299 # width/height of image in pixels (as expected by Xception model)

    x_train = np.zeros((len(labels), INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')

    for i, img_id in tqdm(enumerate(labels['id'])):
        img = read_img(img_id, data_dir, 'train', (INPUT_SIZE, INPUT_SIZE))
        x = xception.preprocess_input(np.expand_dims(img.copy(), axis=0))
        x_train[i] = x
    print('\nTotal Images shape: {}'.format(x_train.shape))

    Xtr = x_train[train_idx]
    Xv = x_train[valid_idx]
    print('Train (images, H, W, C):', Xtr.shape,
          '\nVal (images, H, W, C):', Xv.shape, 
          '\n\nTrain samples (images, labels)', ytr.shape,
          '\nValidation samples (images, labels)', yv.shape)


    model = get_model()

    # TRAINING
    hist = model.fit_generator(train_datagen.flow(Xtr, ytr, batch_size=BATCH_SIZE),
                            steps_per_epoch=train_idx.sum() // BATCH_SIZE,
                            epochs=EPOCHS,
                            validation_data=test_datagen.flow(Xv, yv, batch_size=BATCH_SIZE),
                            validation_steps=valid_idx.sum() // BATCH_SIZE,
                            verbose=2)
    print('Done!')
    # 评估
    test_gen = test_datagen.flow(Xv, yv, batch_size=BATCH_SIZE, shuffle=False)
    probabilities = model.predict_generator(test_gen, steps=len(yv)//BATCH_SIZE+1)
    # 绘制多分类混淆矩阵
    cnf_matrix = confusion_matrix(np.argmax(yv, axis=1), np.argmax(probabilities,axis=1))
    _ = print_confusion_matrix(cnf_matrix, selected_breed_list)

    report = classification_report(np.argmax(probabilities,axis=1), np.argmax(yv, axis=1), target_names=selected_breed_list)
    print(report)

    # 存储训练得到的模型权重
    # !mkdir models
    # model.save_weights('../tmp/models/tl_xception_weights.h5')
    save_path = '/home/colin/Github/Computer-Vision/models/'    # 保存路径
    if not os.path.exists(save_path): # 判断路径是否存在，不存在就创建
        os.makedirs(save_path)
    # !mkdir models
    model.save_weights('/home/colin/Github/Computer-Vision/models/tl_xception_weights.h5') # 保存
    print('Done !')