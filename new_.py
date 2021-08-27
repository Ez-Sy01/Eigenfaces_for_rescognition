import cv2
import numpy as np
from scipy import linalg 

def viewing(data_set,image):
    for i in range(data_set):
        cv2.imshow(str(i),image[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_default.xml')
data_set = 20

filename = list()
for i in range(data_set):
    filename.append('E:\python\python_world/faces/' + str(i+1) + '.jpg')

images = list()
gray_img = list()
for i in range(data_set):
    images.append(cv2.imread(filename[i]))
    images[i] = cv2.resize(images[i],dsize = (256,256),interpolation = cv2.INTER_AREA)
    images[i] = cv2.cvtColor(images[i],cv2.COLOR_BGR2GRAY)
    
# viewing(data_set,images)

G_img = list()
for i in range(data_set):
    G_img.append(np.ravel(images[i]))
G_img = np.transpose(G_img)

mean_vec = np.mean(G_img,axis = 1)

phi_img = np.transpose(G_img)

for i in range(data_set):
    phi_img[i] = phi_img[i] - mean_vec
phi_img = np.transpose(phi_img)

A = phi_img 

C = np.dot(np.transpose(A),A)

eigen_val,eigen_vec = linalg.eig(C)
sort_data = np.argsort(eigen_val)[::-1]
eig_vec = list()
for i in range(data_set):
    eig_vec.append(eigen_vec[sort_data[i]])
eig_val = np.sort(eigen_val)[::-1]

# viewing eigenfaces

# for i in range(data_set):
#     face = np.dot(A,eigen_vec[i]).reshape((256,256))
#     cv2.imshow('Eigenfaces ' + str(i+1) + ' image',face)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

Av_eig = list()
for i in range(data_set):
    Av_eig.append(np.dot(A,eigen_vec[i]))

W = list()

W = np.dot(Av_eig,A)

test_filename = list()

test_data_set = 8

for i in range(test_data_set):
    test_filename.append('E:/python/python_world/faces/test_' + str(i+1) + '.jpg')

test_images = list()
for i in range(test_data_set):
    test_images.append(cv2.imread(test_filename[i]))
    test_images[i] = cv2.resize(test_images[i],dsize = (256,256),interpolation = cv2.INTER_AREA)
    test_images[i] = cv2.cvtColor(test_images[i],cv2.COLOR_BGR2GRAY)
    
test_G_img = list()
for i in range(test_data_set):
    test_G_img.append(np.ravel(test_images[i]))    

test_G_img = np.transpose(test_G_img)
test_mean_vec = np.mean(test_G_img, axis = 1)

test_G_img = np.transpose(test_G_img)

for i in range(test_data_set):
    test_G_img[i] = test_G_img[i] - mean_vec

test_G_img = np.transpose(test_G_img)

test_phi_img = test_G_img
test_A = test_phi_img

test_W = np.dot(Av_eig,test_A)

Omega = np.transpose(W)
test_Omega = np.transpose(test_W)


#Euclidean-distance

result_Euclidean = list()
Euclidean_num = list()
Euclidean_distance = list()
for i in range(test_data_set):
    for k in range(data_set):
        result_Euclidean.append(np.linalg.norm(Omega[k] - test_Omega[i]))
    Euclidean_num.append(np.argmin(result_Euclidean))
    Euclidean_distance.append(np.min(result_Euclidean))
    result_Euclidean = list()

print('==== Euclidean distance ====')
for i in range(test_data_set):
    print(str(i+1) + '번째 사진은 ' + str(Euclidean_distance[i]) + '로 ' + str(Euclidean_num[i] + 1) + "번째 사진과 가장 가깝습니다.")

print(Euclidean_distance)
print(Euclidean_num)


#Manhattan-distance
