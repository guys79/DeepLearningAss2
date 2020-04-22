import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D,Flatten,Dense,Input,Lambda
from keras.layers.pooling import MaxPool2D
from keras.regularizers import l2
from keras import backend
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
import cv2
import pathlib


def get_train_set():
    """
    This function will return the train set in the form of:
    Same person - ('folder_name',ing_num1,img_num2,1)
    Not the same person - ('folder_name1',ing_num1,'folder_name2',img_num2,0)
    :return: The train set
    """
    train = []
    file = open("pairsDevTrain.txt", "r")
    first = True

    for line in file:
        if first:
            first = False
        else:
            split_line = line.split('\t')
            if len(split_line) == 3: # The same person
                train_tuple = (split_line[0],int(split_line[1]),int(split_line[2]),1)
            elif len(split_line) == 4:  # not the same person
                train_tuple = (split_line[0], int(split_line[1]), split_line[2],int(split_line[3]),0)
            else:
                raise Exception("Invalid inout")
            train.append(train_tuple)

    return train


def get_test_set():
    """
    This function will return the test set in the form of:
    Same person - ('folder_name',ing_num1,img_num2,1)
    Not the same person - ('folder_name1',ing_num1,'folder_name2',img_num2,0)
    :return: The test set
    """
    test = []
    file = open("pairsDevTest.txt", "r")
    first = True
    for line in file:
        if first:
            first = False
        else:
            split_line = line.split('\t')
            if len(split_line) == 3:  # The same person
                test_tuple = (split_line[0], int(split_line[1]), int(split_line[2]),1)
            elif len(split_line) == 4:  # not the same person
                test_tuple = (split_line[0], int(split_line[1]), split_line[2], int(split_line[3]),0)
            else:
                raise Exception("Invalid inout")
            test.append(test_tuple)
    return test

def get_single_image(name,number):
    format_num = format_number(number)
    path = r'%s/lfw2/%s/%s_%s.jpg' % (pathlib.Path(__file__).parent.absolute(),name,name,format_num)
    img = cv2.imread(path)
    return img

def format_number(num):
    str_num = str(num)
    while len(str_num)<4:
        str_num = "0%s" % str_num
    return str_num
def get_train_test_sets():
    """
    This function will return the train and test sets in the form of:
    Same person - ('folder_name',ing_num1,img_num2,1)
    Not the same person - ('folder_name1',ing_num1,'folder_name2',img_num2,0)
    :return: The train & test sets
    """
    train = get_train_set()
    test = get_test_set()
    return train,test

def get_conv_weight_initializer():
    """
    This function will return the initializer of the Convolution layers
    :return: Initializer of the Convolution layers
    """
    return keras.initializers.RandomNormal(mean=0,stddev=0.01)

def get_bias_weight_initializer():
    """
        This function will return the initializer of the Convolution layers
        :return: Initializer of the bias
        """
    return keras.initializers.RandomNormal(mean=0.5,stddev=0.01)

def get_fc_weight_initializer():
    """
        This function will return the initializer of the Convolution layers
        :return: Initializer of the FC layers
        """
    return keras.initializers.RandomNormal(mean=0,stddev=0.1)


def build_model(shape):
    """
    This function will assemble a Siamese Neural Networks that compares between inputs with the shape of 'shape'
    :param shape: The shape of the input
    :return: A Siamese Neural Network
    """
    conv_initializer = get_conv_weight_initializer() # The wight initializer for the Convolution layers
    bias_initializer = get_bias_weight_initializer() # The bias initializer for the biases
    fc_initializer = get_fc_weight_initializer() # The fc initializer for the fully connected layers
    n_features = 4096
    l2_penalty = 0.001 # The penalty for the L2 regularization

    model = Sequential()

    # Add a convolution layer with 64 10x10 filters. The activation function is RELU
    model.add(Conv2D(64,(10,10), activation = 'relu', input_shape=shape,kernel_initializer=conv_initializer,
                     bias_initializer = bias_initializer,kernel_regularizer=l2(l2_penalty)))
    # MaxPooling (2,2)
    model.add(MaxPool2D(pool_size=(2,2)))

    # Add a convolution layer with 128 7x7 filters. The activation function is RELU
    model.add(Conv2D(128, (7, 7), activation='relu', kernel_initializer=conv_initializer,
                     bias_initializer=bias_initializer, kernel_regularizer=l2(l2_penalty)))
    # MaxPooling (2,2)
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Add a convolution layer with 128 4x4 filters. The activation function is RELU
    model.add(Conv2D(128, (4,4), activation='relu', kernel_initializer=conv_initializer,
                     bias_initializer=bias_initializer, kernel_regularizer=l2(l2_penalty)))
    # MaxPooling (2,2)
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Add a convolution layer with 256 4x4 filters. The activation function is RELU
    model.add(Conv2D(256, (4, 4), activation='relu', kernel_initializer=conv_initializer,
                     bias_initializer=bias_initializer, kernel_regularizer=l2(l2_penalty)))
    # Flatten the
    model.add(Flatten())

    # Add a fully connected layer with 4096 units (the number of features in the feature map)
    model.add(Dense(n_features,activation='sigmoid',kernel_initializer=fc_initializer,bias_initializer=bias_initializer
                    ,kernel_regularizer=l2(l2_penalty)))

    # Creating the two twins based in the model
    twin1_input = Input(shape)
    twin2_input = Input(shape)
    twin1 = model(twin1_input)
    twin2 = model(twin2_input)

    # Creating a custom layer rule to merge the twin results using l1 siamese distance
    #
    merge_layer_rule = Lambda(lambda output : backend.abs(output[0]-output[1]))

    # Create the merge layer with twin1 and twin2 as the inputs
    merge_layer = merge_layer_rule([twin1,twin2])

    # Creating the output layer (only one output neuron)
    output_layer = Dense(1,activation='sigmoid',kernel_initializer= fc_initializer
                         ,bias_initializer =bias_initializer)(merge_layer)

    # Creating the final network
    final_network = Model(inputs = [twin1_input,twin2_input],outputs = output_layer)

    # Compile network with the loss and optimizer
    final_network.compile(loss=binary_crossentropy,optimizer=Adam(lr = 0.00001))

    return final_network

#f = build_model((105,105,1))
#f.summary()

train,test = get_train_test_sets()
print(train)
print(len(train))
print(test)
print(len(test))
cv2.imshow('image',get_single_image('Marsha_Thomason',1))
cv2.waitKey()