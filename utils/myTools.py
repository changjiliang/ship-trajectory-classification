import numpy as np
from PIL import Image


def read_image1(filename):
    img = Image.open('./im1/'+filename).convert('RGB')
    img = np.array(img)
    # img=np.resize(img,(80,120,3))
    return img


def buildData(ima1):
    # 将训练集图片转换成数组
    X = []
    for i in ima1:
        X.append(read_image1(i))
    X = np.array(X)
    # 根据文件名提取标签
    Y = []
    for filename in ima1:
        m = int(filename.split('-')[0])
        Y.append(m)

    Y = np.array(Y)
    X = X.astype('float32')
    X /= 255
    return X,Y


