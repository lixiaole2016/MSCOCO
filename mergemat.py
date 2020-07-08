import scipy.io as scio
import numpy as np
import h5py

matfile1 = 'E:/datasets/djsrhmirflickr/mirflickr25k-iall.mat'
matfile2= 'cocomat/djsrh/coco-iall.hdf5'
# data = h5py.File('E:/datasets/djsrhmirflickr/mirflickr25k-iall.mat')['IAll'][0:1000]
# iall = data['IAll']
# scio.savemat(matfile2, {'IAll': iall[:1000]})
# mat = scio.loadmat(matfile1)
# scio.savemat(matfile2, {'LAll': mat['L'][:10000,:]})
# mat = scio.loadmat(matfile2)
# scio.savemat(matfile2, {'LAll': mat['YAll'].astype('int32')})
mat =  h5py.File(matfile2)['IAll'].value
print(mat[0,0,0,0])
print('finish!')
