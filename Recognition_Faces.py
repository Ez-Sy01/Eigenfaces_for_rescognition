import cv2
import numpy as np
from numpy.core.fromnumeric import _trace_dispatcher
from scipy import linalg

def viewing(data_set,image_I):
    for i in range(data_set):
        cv2.imshow(str(i+1),image_I[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def viewing_Eigenface(k,C_vectors):
    for i in range(k):
        eigenface = np.dot(A,C_vectors[i]).reshape((256,256))
        cv2.imshow('eigenfaces' + str(i+1),eigenface)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# intitial: setting path
def fn(data_set,name = ''):
    filename = list()
    for i in range(data_set):
        filename.append('E:\python\python_world/faces/' + name + str(i+1 ) + '.jpg')
    return filename

def I_image(data_set,filename):
    I = list()
    for i in range(data_set):
        I.append(cv2.imread(filename[i]))
        I[i] = cv2.resize(I[i],dsize = (256,256),interpolation = cv2.INTER_AREA)
        I[i] = cv2.cvtColor(I[i],cv2.COLOR_BGR2GRAY)
    return I

def gamma_image(data_set,image_I):
    g_I = list()
    for i in range(data_set):
        g_I.append(np.ravel(image_I[i]))
    g_I = np.transpose(g_I)
    return g_I

def phi_image(data_set,g_I,p_I):
    phi_I = list()
    test_I = np.transpose(g_I)
    for i in range(data_set):
        phi_I.append(test_I[i] - p_I)
    return np.transpose(phi_I)

def eigen_image(C):
    eigen_val,eigen_vec = linalg.eig(C)
    masking_sort = np.argsort(eigen_val)[::-1]
    C_values,C_vectors = np.sort(eigen_val)[::-1],list()
    for i in masking_sort:
        C_vectors.append(eigen_vec[i])
    return C_values,C_vectors

def Euclidean_distance_weight(weight,weight_test):
    weight,weight_test = np.transpose(weight),np.transpose(weight_test)
    Euclidean_result,distance_val = np.zeros(np.size(weight_test,0),dtype = int),np.zeros(np.size(weight_test,0))
    distance = list()
    for i in range(test_set):
        for k in range(data_set):
            distance.append(np.linalg.norm(weight_test[i] - weight[k]))
        distance_val[i] = np.min(distance)
        Euclidean_result[i] = np.argmin(distance)
        distance = list()
    return Euclidean_result,distance_val

def Manhattan_distance_weight(weight,weight_test):
    weight,weight_test = np.transpose(weight),np.transpose(weight_test)
    Manhattan_result, Manhattan_val = np.zeros(np.size(weight_test,0),dtype = int),np.zeros(np.size(weight_test,0))
    distance = list()
    for i in range(test_set):
        for k in range(data_set):
            distance.append(np.sum(np.abs(weight[k] - weight_test[i])))
        Manhattan_val[i] = np.min(distance)
        Manhattan_result[i] = np.argmin(distance)
        distance = list()
    return Manhattan_result,Manhattan_val

data_set = 20

test_set = 8

#register_images
filename = fn(data_set) # image_name

I = I_image(data_set,filename) # cv2_image(original)

#resgister_input(test)_image
testname = fn(test_set,'test_')

t_I = I_image(test_set,testname)

# # viewing_origin_image
# viewing(data_set,I)

# # viewing_testing_image
# viewing(test_set,t_I)

# image_data_size -> 256 by 256 /
dim = np.size(I,1) ** 2

#creative_gamma_image
g_I = gamma_image(data_set,I)

t_g_I = gamma_image(test_set,t_I)

#creative_mean_vector-phsi vector
p_I  = np.mean(g_I,axis = 1)

#creative mean-zero-vector_phi_vector
phi_I = phi_image(data_set,g_I,p_I)

t_phi_I = phi_image(test_set,t_g_I,p_I)

#phi_image = A
A = phi_I

t_A = t_phi_I

#creative covariance_matrix (AT * A)
C = np.dot(np.transpose(A),A)

#creative eigen_values & eigen_vectors -> size sort (big -> small)
C_values, C_vectors = eigen_image(C)

#viewing_eigenFaces
k = 18
viewing_Eigenface(k,C_vectors)

#make weight -> train_set
A_eigenvectors = list()
for i in range(k):
    A_eigenvectors.append(np.dot(A,C_vectors[i]))
weight = np.dot(A_eigenvectors,A)

#make weight -> input image
weight_test = np.dot(A_eigenvectors,t_A)

# Euclidean distance - weight_base - face recognition

# Euclidean distance -> classify k-th face image

Euclidean_result,distance_val = Euclidean_distance_weight(weight,weight_test)
count = 0
print('k가 {}입니다.'.format(k))
print('==== Euclidean distance ====')
for i in Euclidean_result:
    print('test_' + str(count+1) + 'th - face image => {}'.format(i+1))
    count += 1
print(Euclidean_result)
print(distance_val)

# Manhattan distance - weight_base - face recognition

# Manhattan distance -> classify k-th face image

Manhattan_result, Manhattan_val = Manhattan_distance_weight(weight,weight_test)
count = 0

print('==== Manhattan distance ====')
for i in Manhattan_result:
    print('test_' + str(count+1) + 'th - face image => {}'.format(i+1))
    count += 1
print(Manhattan_result)
print(Manhattan_val)