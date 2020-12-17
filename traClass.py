from scipy.sparse.construct import random
from sklearn.model_selection import train_test_split
from glob import glob
from PIL import Image
import os
import numpy as np
import keras
import keras_resnet.models
from keras_radam import RAdam
from keras.callbacks import TensorBoard
import time
import math
from utils.myTools import buildData

ima1 = os.listdir('./im1')
X,Y=buildData(ima1)



s_time = time.strftime("%Y%m%d%H%M%S", time.localtime())  # 时间戳
train_X,test_X,train_y,test_y=train_test_split(X,Y,test_size=0.2,random_state=0)


shape,classes=(44,68,3),5
x=keras.layers.Input(shape)
model=keras_resnet.models.ResNet50(x,classes=classes)
model.compile(RAdam(), "categorical_crossentropy", ["accuracy"])
model.summary()
#logs文件路径
logs_path = './logs/log_%s' % (s_time)
try:
    os.makedirs(logs_path)
except:
    pass
#将loss ，acc， val_loss ,val_acc记录tensorboard
tensorboard = TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True)
#,write_batch_performance=True)

training_y = keras.utils.np_utils.to_categorical(train_y)
test_y = keras.utils.np_utils.to_categorical(test_y)
history = model.fit(train_X, training_y, shuffle=True,
                    verbose=1,         
                    epochs=20, callbacks=[tensorboard])
score = model.evaluate(test_X, test_y, batch_size=50)
print(score)
