import keras
import numpy as np
from sklearn.utils import resample
import os
import cv2

LEARNING_RATE = 0.001
#EPOCH = 100
EPOCH = 10
DATA_SIZE = 5000
TRAINING_SIZE = 5000
TEST_SIZE = 1000
NUM_TARGET = 1
#NUM_SHADOW = 10
NUM_SHADOW = 3
IN = 1
OUT = 0
VERBOSE = 1

def sample_data(train_data,test_data,num_sets):
    (x_train, y_train), (x_test, y_test) = train_data, test_data
    new_x_train, new_y_train = [], []
    new_x_test, new_y_test = [], []
    for i in range(num_sets):
        x_temp, y_temp = resample(x_train, y_train, n_samples=TRAINING_SIZE, random_state=0)
        new_x_train.append(x_temp)
        new_y_train.append(y_temp)
        x_temp, y_temp = resample(x_test, y_test, n_samples=TEST_SIZE, random_state=0)
        new_x_test.append(x_temp)
        new_y_test.append(y_temp)
    return (new_x_train, new_y_train), (new_x_test, new_y_test)

def build_fcnn_model():
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    # build the model
    model = Sequential()
    model.add(Flatten(input_shape=(28,28)))
    model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    model.summary()
    return model

def get_trained_keras_models(keras_model, train_data, test_data, num_models):
    (x_train, y_train), (x_test, y_test) = train_data, test_data
    models = []
    for i in range(num_models):
        models.append(keras.models.clone_model(keras_model))
        models[i].compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        models[i].fit(x_train[i], y_train[i], batch_size=32, epochs=EPOCH, verbose=VERBOSE, shuffle=True)
        score = models[i].evaluate(x_test[i], y_test[i], verbose=VERBOSE)
        print('\n', 'Model ', i, ' test accuracy:', score[1])
    return models

def get_attack_dataset(models, train_data, test_data, num_models, data_size):
    # generate dataset for the attack model
    (x_train, y_train), (x_test, y_test) = train_data, test_data
    num_classes = len(y_train[0][0])
    x_data, y_data = [[] for i in range(num_classes)], [[] for i in range(num_classes)]
    for i in range(num_models):
        # IN data
        x_temp, y_temp = resample(x_train[i], y_train[i], n_samples=data_size, random_state=0)
        for j in range(data_size):
            y_idx = np.argmax(y_temp[j])
            x_data[y_idx].append(models[i].predict(x_temp[j:j+1])[0])
            y_data[y_idx].append(IN)
        # OUT data
        x_temp, y_temp = resample(x_test[i], y_test[i], n_samples=data_size, random_state=0)
        for j in range(data_size):
            y_idx = np.argmax(y_temp[j])
            x_data[y_idx].append(models[i].predict(x_temp[j:j+1])[0])
            y_data[y_idx].append(OUT)
    return x_data, y_data

def get_trained_svm_models(train_data, test_data, num_models):
    from sklearn import svm
    (x_train, y_train), (x_test, y_test) = train_data, test_data
    models = []
    for i in range(num_models):
        print('Training svm model : ', i)
        models.append(svm.SVC(gamma='scale',kernel='linear',verbose=VERBOSE))
        models[i].fit(x_train[i], y_train[i])
        score = models[i].score(x_test[i],y_test[i])
        print('SVM model ', i, 'score : ',score)
    return models

def permutate_dataset(dataset, y, name='Training'):
    perm = np.random.permutation(dataset.shape[0])
    perm_dataset = dataset[perm]
    perm_y = y[perm]
    print('%s set' % name, perm_dataset.shape)
    return perm_dataset, perm_y

def main():
    print('Hello World!')
    # load the pre-shuffled train and test data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    print(y_train.shape,x_train.shape)
    train_set_x = []
    train_set_y = []
    for i in range(10):
        for filename in os.listdir("/home/jingwezh/Data/ml_mbr_inf/mnist/"+str(i)+"/"):  
            img1 = cv2.imread("/home/jingwezh/Data/ml_mbr_inf/mnist/"+str(i)+"/"+filename,cv2.IMREAD_GRAYSCALE)   #读取图片，第二个参数表示以灰度图像读入
            if img1 is None:                   #判断读入的img1是否为空，为空就继续下一轮循环
                continue
            res1= cv2.resize(img1,(28,28))              #对图片进行缩放，第一个参数是读入的图片，第二个是制定的缩放大小
            res1_1 = res1.reshape(1,784)/255       #将表示图片的二维矩阵转换成一维
            res1_1_1 = res1_1.tolist()                    #将numpy.narray类型的矩阵转换成list
            train_set_x.append(res1_1_1)
            train_set_y.append(i)
    
    train_set_x = np.array(train_set_x).reshape(-1,28,28)
    train_set_y = np.array(train_set_y)
    # print(train_set_x.shape,train_set_y.shape)
    # rescale [0,255] --> [0,1]
    # x_train = x_train.astype('float32')/255
    x_train = train_set_x
    y_train = train_set_y
    x_train, y_train = permutate_dataset(x_train, y_train)
    x_test = x_train
    y_test = y_train
    # x_test = x_test.astype('float32')/255
    # one-hot encoding for the labels
    # num_classes = len(np.unique(y_train))
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # split the data for each model
    target_train = (x_train[:TRAINING_SIZE*NUM_TARGET],y_train[:TRAINING_SIZE*NUM_TARGET])
    target_test = (x_test[:TEST_SIZE*NUM_TARGET],y_test[:TEST_SIZE*NUM_TARGET])
    target_train_data, target_test_data = sample_data(target_train, target_test, NUM_TARGET)

    shadow_train = (x_train[TRAINING_SIZE*NUM_TARGET:],y_train[TRAINING_SIZE*NUM_TARGET:])
    shadow_test = (x_test[TEST_SIZE*NUM_TARGET:],y_test[TEST_SIZE*NUM_TARGET:])
    shadow_train_data, shadow_test_data = sample_data(shadow_train, shadow_test, NUM_SHADOW)

    cnn_model = build_fcnn_model()
    # compile the target model
    target_models = get_trained_keras_models(cnn_model, target_train_data, target_test_data, NUM_TARGET)
    # compile the shadow models
    shadow_models = get_trained_keras_models(cnn_model, shadow_train_data, shadow_test_data, NUM_SHADOW)

    # get train data for the attack model
    attack_train = get_attack_dataset(shadow_models, shadow_train_data, shadow_test_data, NUM_SHADOW, TEST_SIZE)
    # get test data for the attack model
    attack_test = get_attack_dataset(target_models, target_train_data, target_test_data, NUM_TARGET, TEST_SIZE)

    # training the attack model
    attack_model = get_trained_svm_models(attack_train, attack_test, num_classes)

    # TODO generate the report


if __name__== '__main__':
    main()
