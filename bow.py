# 用于将coco数据集的caption提取bow特征向量
import numpy as np
import scipy.io as scio
from sklearn.feature_extraction.text import CountVectorizer
# 参数详见 https://blog.csdn.net/weixin_38278334/article/details/82320307
# max_features：单词的维度。
# binary=FALSE：单词出现了几次就是几，binary=True：单词只要出现就是1。
count = CountVectorizer(max_features=2000,binary=True)
data = []
for line in open("cocotxt/caption.txt", "r"):
    data.append(line)
docs = np.array(data)
bag = count.fit_transform(docs)
result = bag.toarray()
# 保存为txt文件
# np.savetxt('cocotxt/bowresult.txt',result,fmt='%d')
# 保存为mat文件
matfile = 'cocomat/coco.mat'
# 写入第一个数据
# scio.savemat(matfile, {'T_bow1':result})
# 附加写入其他数据
temp  = scio.loadmat(matfile)['T_bow1']
scio.savemat(matfile, {'T_bow1':temp,'T_bow2':result})

#检查mat文件
# check = scio.loadmat(matfile)
# print(check['T_bow2'])
