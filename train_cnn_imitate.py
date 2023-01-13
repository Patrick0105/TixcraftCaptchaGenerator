import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import os

# 讀入訓練資料
train_df = pd.read_csv("Training/TrainingSet.csv")

# 讀入圖片並轉成矩陣
train_images = []
for img_name in train_df['file_name']:
    img = tf.keras.preprocessing.image.load_img("Training/Training_Img/"+img_name, target_size=(120, 100))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    train_images.append(img_array)

# 將圖片矩陣轉成 numpy array 並標準化
train_images = np.array(train_images) / 255

# 建立模型
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(120, 100, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='softmax')
])

# 編譯模型
model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# 將驗證碼文字轉換成 one-hot encoding
code_map = dict(zip('abcdefghijklmnopqrstuvwxyz', range(26)))
train_df['code'] = train_df['code'].map(code_map)
train_labels = tf.keras.utils.to_categorical(train_df['code'])

# 訓練模型
history = model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

predictions = model.predict(test_images)
