import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.metrics import confusion_matrix , classification_report
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers, models
import random

CATEGORIES = ["a", "b", "c", "d", "e", "f", "g", "h","i", "j", "k", "l", "m", "n", "o", "p",
              "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

DATADIR = "dataset"
VALDIR = "valset"
TESTDIR = "testset"

training_data = []
validation_data = []
test_data = []

def plot_history(history):
    fig, axs = plt.subplots(2)
    
    #create accuracy subplot
    axs[0].plot(history.history["accuracy"], label = "train accuracy")
    axs[0].plot(history.history["val_accuracy"], label = "test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_title("Accuracy eval")
    axs[0].legend(loc = "lower right")
    
    axs[1].plot(history.history["loss"], label = "train error")
    axs[1].plot(history.history["val_loss"], label = "test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].set_title("Error eval")
    axs[1].legend(loc = "upper right")

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) #gets us to the path in categories
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try: #incase the picture cannot be read
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (80, 80))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
        
def modify_training_data():
    X_train = [] #feature set
    y_train = [] #labels

    for features, label in training_data:
        X_train.append(features)
        y_train.append(label)
        
    X_train = np.array(X_train).reshape(-1, 80, 80,1)
    return X_train, y_train


def create_validation_data():
    for category in CATEGORIES:
        path = os.path.join(VALDIR, category) #gets us to the path in categories
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try: #incase the picture cannot be read
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (80, 80))
                validation_data.append([new_array, class_num])
            except Exception as e:
                pass
        
def modify_validation_data():
    X_val = [] #val set
    y_val = [] #labels

    for features, label in validation_data:
        X_val.append(features)
        y_val.append(label)
        
    X_val = np.array(X_val).reshape(-1, 80, 80, 1) 
    return X_val, y_val

def create_test_data():
    for category in CATEGORIES:
        path = os.path.join(TESTDIR, category) #gets us to the path in categories
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try: #incase the picture cannot be read
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (80, 80))
                test_data.append([new_array, class_num])
            except Exception as e:
                pass
        
def modifiy_test_data():
    X_test = [] #test set
    y_test = [] #labels

    for features, label in test_data:
        X_test.append(features)
        y_test.append(label)
        
    X_test = np.array(X_test).reshape(-1, 80, 80, 1)
    X_test = X_test/255
    return X_test, y_test
    

def arrayify(X_train, y_train, X_val, y_val):
    X_train = X_train/255
    X_val = X_val/255
    X_train_prediction = X_train
    X_val_prediction = X_val

    X_train = np.array(X_train)
    X_val = np.array(X_val)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    return X_train, y_train, X_val, y_val, X_train_prediction, X_val_prediction

def model(X_train, y_train, X_val, y_val):
    data_augmentation = keras.Sequential([layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(80, 80,1)),layers.experimental.preprocessing.RandomContrast(0.6)])
    cnn = models.Sequential([
    data_augmentation,
    layers.AveragePooling2D(6,3),
    layers.Conv2D(filters = 16, kernel_size = (5,5), activation = 'relu',padding='same', kernel_regularizer=keras.regularizers.l2(0.001),input_shape = (80, 80, 1)), 
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
    layers.Dropout(0.3),
    layers.Conv2D(filters = 32, kernel_size = (5,5), activation = 'relu',padding='same', kernel_regularizer=keras.regularizers.l2(0.001)), 
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
    layers.Dropout(0.3),
    layers.Conv2D(filters = 64, kernel_size = (5,5), activation = 'relu',padding='same', kernel_regularizer=keras.regularizers.l2(0.001)), 
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
    
    layers.Flatten(),
    
    layers.Dense(32, activation = 'relu'),
    
    layers.Dense(26, activation = 'softmax')
    ])

    cnn.compile(optimizer = keras.optimizers.Adam(learning_rate=1e-4), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    cnn.fit(X_train, y_train,validation_data=(X_val, y_val), epochs = 5)
    return cnn

def retrain_model_train(X_train_prediction, y_train, cnn):
    signs = []
    letters = []

    sign_prediction = cnn.predict(X_train_prediction)

    for i in range(len(sign_prediction)):
        
        letter = y_train[i]
        
        letter_prediction = np.argmax(sign_prediction[i])
        
        if letter_prediction != letter:
            #print("predicting: ", CATEGORIES[letter], ", with: ", CATEGORIES[letter_prediction])
            signs.append(X_train_prediction[i])
            letters.append(letter)
            
    print("Length of Data: ", len(X_train_prediction))
    print("Length of data wrong: ", len(signs))
    print("Accuracy: ", (1-len(signs)/len(X_train_prediction)))

    for i in range (26):
        find = letters.count(i)
        print(CATEGORIES[i], ":", find, "({}%)".format(round((find/len(signs)*100), 1)))
    
    signs = np.array(signs).reshape(-1, 80, 80, 1)
    signs = np.array(signs)
    letters = np.array(letters)

    cnn.fit(signs, letters, epochs = 5)

def retrain_model_val(X_val_prediction, y_val, cnn):
    signs_val = []
    letters_val = []

    sign_prediction = cnn.predict(X_val)

    for i in range(len(sign_prediction)):
        
        letter = y_val[i]
        
        letter_prediction = np.argmax(sign_prediction[i])
        
        if letter_prediction != letter:
            #print("predicting: ", CATEGORIES[letter], ", with: ", CATEGORIES[letter_prediction])
            signs_val.append(X_val[i])
            letters_val.append(letter)
            
    print("Length of Training Data: ", len(X_val))
    print("Length of data wrong: ", len(signs_val))
    print("Accuracy: ", (1-len(signs_val)/len(X_val)))

    for i in range (26):
        find = letters_val.count(i)
        print(CATEGORIES[i], ":", find, "({}%)".format(round((find/len(signs_val)*100), 1)))

    signs_val = np.array(signs_val).reshape(-1, 80, 80, 1)

    signs_val = np.array(signs_val)
    letters_val = np.array(letters_val)

    cnn.fit(signs_val, letters_val, epochs = 5)

def model_testing(X_test, y_test, cnn):
    signs_test = []
    letters_test = []

    sign_prediction = cnn.predict(X_test)

    for i in range(len(sign_prediction)):
        
        letter = y_test[i]
        
        letter_prediction = np.argmax(sign_prediction[i])
        
        if letter_prediction != letter:
            #print("predicting: ", CATEGORIES[letter], ", with: ", CATEGORIES[letter_prediction])
            signs_test.append(X_test[i])
            letters_test.append(letter)
            
    print("Length of Training Data: ", len(X_test))
    print("Length of data wrong: ", len(signs_test))
    print("Accuracy: ", (1-len(signs_test)/len(X_test)))

    for i in range (26):
        find = letters_test.count(i)
        print(CATEGORIES[i], ":", find, "({}%)".format(round((find/len(signs_test)*100), 1)))

if __name__ == "__main__":
    create_training_data()
    random.shuffle(training_data)
    X_train, y_train = modify_training_data()

    create_validation_data()
    random.shuffle(validation_data)
    X_val, y_val = modify_training_data()

    create_test_data()
    X_test, y_test = modifiy_test_data()    

    X_train, y_train, X_val, y_val, X_train_prediction, X_val_prediction = arrayify(X_train, y_train, X_val, y_val)

    model(X_train, y_train, X_val, y_val) #Creates the model

    retrain_model_train(X_train_prediction, y_train, cnn) #Retrains the model on values that it got wrong originally

    retrain_model_val(X_val_prediction, y_val, cnn) #Retrains it against the validation model

    model_testing(X_test, y_test, cnn)
    #------------------- From this point, you can modify how the model is retrained or not ---------------------------#