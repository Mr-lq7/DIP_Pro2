# 用LBP+SVM算法实现

import os
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data, io, data_dir, filters, feature
from skimage.color import label2rgb
import skimage
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVR, SVC

path_train = 'training/'
path_test = 'testing/'


def loadTrainImage():
    i = 0
    train_label = np.zeros(60)
    train_data = np.zeros((60, 150, 150))  # 相当于60*(150*150)
    for filename in os.listdir("training/"):
        if (i == 60):
            break
        cur_label = int(filename.split('_', 1)[0])
        train_label[i] = cur_label
        image = cv2.imread(path_train + filename, cv2.IMREAD_GRAYSCALE)  # 只读取图片的灰度
        print(image.shape)
        # cv2.namedWindow('1', 0)
        # cv2.imshow('1',image)
        # cv2.waitKey()
        im = np.array(image)
        print(im.shape)
        train_data[i] = im
        print(train_data[i].shape)
        i = i + 1

    return train_data, train_label


def loadTestImage():
    i = 0
    test_label = np.zeros(60)
    test_data = np.zeros((60, 150, 150))  # 相当于60*(150*150)
    for filename in os.listdir("testing/"):
        if (i == 60):
            break
        cur_label = int(filename.split('_', 1)[0])
        test_label[i] = cur_label
        image = cv2.imread(path_test + filename, cv2.IMREAD_GRAYSCALE)  # 只读取图片的灰度,cv2.IMREAD_GRAYSCALE
        im = np.array(image)
        # print(im.shape)
        test_data[i] = im
        i = i + 1

    return test_data, test_label


# settings for LBP
radius = 1  # LBP算法中范围半径的取值
n_points = 8 * radius  # 领域像素点数


# 注意:LBP的应用中,如纹理分类,人脸分析等,一般都不将LBP图谱作为特征向量用于分类识别,而是采用LBP特征谱的统计直方图作为特征向量用于分类识别
def feature_abstract(train_data, test_data):
    train_hist = np.zeros((60, 256))
    test_hist = np.zeros((60, 256))

    # 对训练集处理

    for i in range(60):
        # 使用LBP方法提取图像的纹理特征

        lbp = local_binary_pattern(train_data[i], n_points, radius)
        # 统计图像的直方图
        # edges = filters.sobel(train_data[i])
        # lbp = lbp - edges
        # print(lbp.reshape)
        # lbp = lbp.reshape(256)
        max_bins = int(lbp.max() + 1)
        train_hist[i], bin_edges = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))

    # test_hist, bin_edges = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
    # print(test_hist)
    # print(bin_edges)

    # 对测试集处理
    for i in range(60):
        # 使用LBP方法提取图像的纹理特征
        lbp = local_binary_pattern(test_data[i], n_points, radius)
        # cv2.imshow('1',lbp)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # 统计图像的直方图
        # edges = filters.sobel(train_data[i])
        # lbp = lbp - edges
        # lbp = lbp.reshape(256)
        max_bins = int(lbp.max() + 1)
        test_hist[i], bin_edges = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))

    return train_hist, test_hist


# 计算两个向量的余弦相似度
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
    print("Accuracy:", float(ans / 60))


if __name__ == '__main__':

    train_data, train_label = loadTrainImage()
    test_data, test_label = loadTestImage()
    train_hist, test_hist = feature_abstract(train_data, test_data)

    predict_label = predict(train_hist, test_hist)
    print(predict_label)
    Accuracy(predict_label, test_label)

