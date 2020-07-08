import torchvision.datasets as dset
import torchvision.transforms as transforms
import scipy.io as scio
from tqdm import tqdm
import torch
from coco.model import ImgNet
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py

# coco_train = dset.CocoDetection(root='E:/datasets/coco2017/train2017',
#                                annFile='E:/datasets/coco2017/annotations/instances_train2017.json',
#                                transform=transforms.ToTensor())
#
# print('Number of samples: ', len(coco_train))
# img, target = coco_train[3]
# # 原本的image数组形状是[3,height,width],转换成[height,width,3]才能用matplotlib展示
# img = img.numpy().transpose(1,2,0)
# for i in target:
#     print(i['category_id'])
# # 不知道前面的哪个方法会改变matplotlib打印图片的方式，只能先这样解决了
# # 切换matplotlib内核
# matplotlib.use('TkAgg')
# plt.imshow(img)
# plt.show()


# 提取coco2017的标签并保存为mat文件
def createlabelmat():
    coco_train = dset.CocoDetection(root='E:/datasets/coco2017/train2017',
                                    annFile='E:/datasets/coco2017/annotations/instances_train2017.json',
                                    transform=transforms.ToTensor())

    print('Number of samples: ', len(coco_train))
    result = []
    for i in tqdm(range(len(coco_train))):
        _, target = coco_train[i]
        labels = []
        for t in target:
            label = t['category_id']
            # 去除重复的标签
            if label not in labels:
                labels.append(label)
        result.append(labels)
    scio.savemat('cocomat/labels.mat', {'labels': result})
    print('create labels finish!')

# 提取coco2017的标签并保存为mat文件
def createonehotlabelmat():
    coco_train = dset.CocoDetection(root='E:/datasets/coco2017/train2017',
                                    annFile='E:/datasets/coco2017/annotations/instances_train2017.json',
                                    transform=transforms.ToTensor())

    print('Number of samples: ', len(coco_train))
    onehot = np.zeros((len(coco_train), 91))
    for i in tqdm(range(len(coco_train))):
        _, target = coco_train[i]
        labels = []
        for t in target:
            label = t['category_id']
            # 去除重复的标签
            if label not in labels:
                onehot[i][label - 1] = 1
    # scio.savemat('cocomat/labels.mat', {'labels': onehot})
    matfile = 'cocomat/coco.mat'
    temp = scio.loadmat(matfile)
    scio.savemat(matfile, {'T_bow1': temp['T_bow1'], 'T_bow2': temp['T_bow2'],'L':onehot})
    print('create labels finish!')

def createlabeltxt():
    coco_train = dset.CocoDetection(root='E:/datasets/coco2017/train2017',
                                    annFile='E:/datasets/coco2017/annotations/instances_train2017.json',
                                    transform=transforms.ToTensor())
    f = open('cocotxt/labels.txt', 'w')
    for i in tqdm(range(len(coco_train))):
        _, target = coco_train[i]
        labels = []
        for t in target:
            label = t['category_id']
            # 去除重复的标签
            if label not in labels:
                labels.append(label)
        f.write(str(labels).replace('[', '').replace(']', '').replace(' ', ''))
        f.write('\n')
        labels.clear()
    f.close()
    print('create labels txt finish!')


def loadlabels():
    dataFile = 'cocomat/labels.mat'
    data = scio.loadmat(dataFile)
    print(data['labels'])


def reshapeImage(img):
    res = cv2.resize(img, (224, 224))
    # dst = np.zeros(res.shape, dtype=np.float32)
    # cv2.normalize(res, dst=dst, alpha=1.0, beta=0, norm_type=cv2.NORM_L2)
    res *= 127.0
    res = res.astype(np.uint8)
    # matplotlib.use('TkAgg')
    # plt.imshow(res)
    # plt.show()
    res = res.transpose((2, 0, 1))
    # 用于提取cnn特征
    # return torch.from_numpy(res)
    # 用于压缩图片
    return res
# 提取图像的cnn特征并保存mat
def imagesfeature():
    coco_train = dset.CocoDetection(root='E:/datasets/coco2017/train2017',
                                    annFile='E:/datasets/coco2017/annotations/instances_train2017.json',
                                    transform=transforms.ToTensor())

    print('Number of samples: ', len(coco_train))
    imgs = torch.zeros(1000, 3, 224, 224)
    feats = np.zeros(shape=(len(coco_train), 4096))
    ind = 0
    alexnet = ImgNet()
    alexnet.cuda()
    for i in tqdm(range(len(coco_train))):
        img, _ = coco_train[i]
        img = np.array(img)
        img = img.transpose((1, 2, 0))
        img = reshapeImage(img)
        imgs[ind,:] = img
        if ind % 999 == 0 and ind is not 0:
            feats[i-999:i+1,:] = alexnet(imgs.cuda()).cpu().detach().numpy()
            ind = 0
        else:
            ind = ind + 1
    scio.savemat('cocomat/feature.mat', {'feature': feats})
    print('finish!')

#将图像压缩到224*224并保存mat
def imagesresize():
    coco_train = dset.CocoDetection(root='E:/datasets/coco2017/train2017',
                                    annFile='E:/datasets/coco2017/annotations/instances_train2017.json',
                                    transform=transforms.ToTensor())

    print('Number of samples: ', len(coco_train))
    imgs = np.zeros((len(coco_train), 3, 224, 224),dtype=np.uint8)
    for i in tqdm(range(len(coco_train))):
        img, _ = coco_train[i]
        img = np.array(img)
        img = img.transpose((1, 2, 0))
        img = reshapeImage(img)
        imgs[i,:] = img
    # scio.savemat('cocomat/imageresize224.mat', {'': imgs})
    f = h5py.File("cocomat/img-size224-numall-value127.hdf5", "w")
    d1 = f.create_dataset("IAll", data=imgs)
    print('finish!')
# 将coco数据集中的caption数据提取并保存为txt文件,每行为一个图像的caption
def createcaptiontxt():
    coco_train = dset.CocoCaptions(root='E:/datasets/coco2017/train2017',
                                   annFile='E:/datasets/coco2017/annotations/captions_train2017.json',
                                   transform=transforms.ToTensor())

    print('Number of samples: ', len(coco_train))
    f = open('cocotxt/caption.txt', 'w')
    for i in tqdm(range(len(coco_train))):
        _, target = coco_train[i]
        f.write(''.join(target).replace("\n", ""))
        f.write('\n')
    print('create caption txt finish!')

def txttomat():
    res = np.zeros((118287,4096),dtype=float)
    f = open('cocotxt/cnnfeature.txt')
    lines = f.readlines()
    row = 0

    for line in lines:
        list = line.strip('\n').split(' ')
        res[row:] = list[:]
        row += 1
        print(row)
    print(res)


if __name__ == '__main__':
    imagesresize()
    # createonehotlabelmat()
    # txttomat()