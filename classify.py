import os
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data, io,data_dir,filters, feature
from skimage.color import label2rgb
import skimage
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from tensorflow.contrib.learn import SVM

path_train = 'training/'
path_test = 'testing/'
def loadTrainImage():
    i = 0
    train_label = np.zeros(60)
    train_data = np.zeros((60,150,150)) #相当于60*(150*150)
    for filename in os.listdir("training/"):
        if (i == 60):
            break
        cur_label = int(filename.split('_',1)[0])
        train_label[i] = cur_label
        image = cv2.imread(path_train + filename, cv2.IMREAD_GRAYSCALE) #只读取图片的灰度
        # cv2.namedWindow('1', 0)
        # cv2.imshow('1',image)
        # cv2.waitKey()
        im = np.array(image)
        train_data[i] = im
        i = i + 1

    return train_data, train_label

def loadTestImage():
    i = 0
    test_label = np.zeros(60)
    test_data = np.zeros((60,150,150)) #相当于60*(150*150)
    for filename in os.listdir("testing/"):
        if (i == 60):
            break
        cur_label = int(filename.split('_',1)[0])
        test_label[i] = cur_label
        image = cv2.imread(path_test + filename, cv2.IMREAD_GRAYSCALE) #只读取图片的灰度,cv2.IMREAD_GRAYSCALE
        im = np.array(image)
        # print(im.shape)
        test_data[i] = im
        i = i + 1
    return test_data, test_label

# settings for HOG
ori = 12
per_cell = (8, 8)
per_block = (2, 2)
norm = 'L2-Hys'


def feature_abstract(train_data, test_data):

    list = []
    #对训练集处理

    for i in range(60):

        #使用Hog方法提取图像的纹理特征
        ft = feature.hog(train_data[i],  # input image
                         orientations=ori,  # number of bins
                         pixels_per_cell=per_cell,  # pixel per cell
                         cells_per_block=per_block,  # cells per blcok
                         block_norm=norm,  # block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}
                         transform_sqrt=True,  # power law compression (also known as gamma correction)
                         feature_vector=True,  # flatten the final vectors
                         visualise=False)  # return HOG map

        list.append(ft)
        train_hist = np.array(list, 'float64')

    list = []
    #对测试集处理
    for i in range(60):
        # 使用Hog方法提取图像的纹理特征

        fb = feature.hog(test_data[i],  # input image
                         orientations=ori,  # number of bins
                         pixels_per_cell=per_cell,  # pixel per cell
                         cells_per_block=per_block,  # cells per blcok
                         block_norm=norm,  # block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}
                         transform_sqrt=True,  # power law compression (also known as gamma correction)
                         feature_vector=True,  # flatten the final vectors
                         visualise=False)  # return HOG map

        list.append(fb)
        test_hist = np.array(list, 'float64')

    return train_hist, test_hist

#计算两个向量的余弦相似度
def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    sim = num / denom
    return sim

def predict(train_hist, test_hist):
    predict_label = np.zeros(60)
    index = 0
    mmax = float(-1)
    for i in range(60):
        for j in range(60):
            res = cos_sim(test_hist[i], train_hist[j])
            if (res > mmax):
                mmax = res
                index = j

        predict_label[i] = train_label[index]
        mmax = float(-1)
    return predict_label

def Accuracy(predict_label, test_label):
    ans = 0
    for i in range(60):
        if predict_label[i] == test_label[i]:
            ans += 1
    print("使用Hog + Cosine Similarity:accuracy:%.2f%%" %(float(ans/60) *100))

if __name__ == '__main__':

    train_data, train_label = loadTrainImage()
    test_data, test_label = loadTestImage()
    train_hist, test_hist = feature_abstract(train_data, test_data)

    # svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1);
    # res = OneVsRestClassifier(svr_rbf, -1).fit(train_hist, train_label).score(test_hist, test_label)
    model = OneVsRestClassifier(svm.SVC(kernel='linear'))
    clf = model.fit(train_hist, train_label)
    accuracy = clf.score(test_hist, test_label)
    print("使用OneVsRestClassifier中的svm.SVC实现多分类:accurcay:%.2f%%" %(100*accuracy))

    predict_label = predict(train_hist, test_hist)
    Accuracy(predict_label, test_label)



    model_svm = SVC(kernel='linear')
    model_svm.fit(train_hist, train_label)
    ss = model_svm.score(test_hist, test_label)
    print("使用SVM中的SVC:accurcay:%.2f%%" % (100 * ss))



