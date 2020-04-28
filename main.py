import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, Flatten, Dense, Input, Lambda, Dropout
from keras.layers.pooling import MaxPool2D
from keras.regularizers import l2
from keras.metrics import accuracy, binary_accuracy
from keras import backend
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
import cv2
import pathlib
import random
import numpy as np
import tensorflow as tf

#threads = 1
#tf.config.threading.set_intra_op_parallelism_threads(threads)
#tf.config.threading.set_inter_op_parallelism_threads(threads)


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
            if len(split_line) == 3:  # The same person
                train_tuple = (split_line[0], int(split_line[1]), int(split_line[2]), 1)
            elif len(split_line) == 4:  # not the same person
                train_tuple = (split_line[0], int(split_line[1]), split_line[2], int(split_line[3]), 0)
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
                test_tuple = (split_line[0], int(split_line[1]), int(split_line[2]), 1)
            elif len(split_line) == 4:  # not the same person
                test_tuple = (split_line[0], int(split_line[1]), split_line[2], int(split_line[3]), 0)
            else:
                raise Exception("Invalid inout")
            test.append(test_tuple)
    return test


def get_img_set(tuple_list, shape=(105, 105, 1)):
    """
    This function receives a set of image tuples and a shape
    The function will return the set of image objects with the shape of the given shape
    :param tuple_list: a list of ('folder_name', img1,img2) or ('folder_name1',img1,'folder_name2',img2)
    :param shape: the image's shape
    :return: A set of image objects with the shape of the given shape
    """

    x1 = []
    x2 = []
    y = []
    random.shuffle(tuple_list)
    for tple in tuple_list:
        if len(tple) == 4:
            img1 = get_single_image(tple[0], tple[1], shape=shape)
            img2 = get_single_image(tple[0], tple[2], shape=shape)
            y.append(tple[3])
        elif len(tple) == 5:
            img1 = get_single_image(tple[0], tple[1], shape=shape)
            img2 = get_single_image(tple[2], tple[3], shape=shape)
            y.append(tple[4])
        else:

            raise Exception("invalid tuple size %d" % len(tple))
        x1.append(img1)
        x2.append(img2)

    return [np.array(x1), np.array(x2), np.reshape(np.array(y), (len(y),))]


def get_single_image(name, number, shape=(105, 105, 1), margin=0.25):
    """
    This function receives a image name, number and shape.
    The function will return a image object with the given shape
    :param name: The image's name
    :param number: The image's number
    :param shape: The image's shape
    :param margin: portion of img to crop from each border
    :return: The image object with the given shap
    """
    format_num = format_number(number)
    path = r'%s/lfw2/%s/%s_%s.jpg' % (pathlib.Path(__file__).parent.absolute(), name, name, format_num)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    width = img.shape[0]
    height = img.shape[1]
    width_margin = int(margin * width)
    height_margin = int(margin * height)
    cropped_img = img[width_margin:width - width_margin, height_margin:height - height_margin]
    resized_img = cv2.resize(cropped_img, (shape[0],shape[1]), interpolation=cv2.INTER_AREA) / 255
    resized_img = np.reshape(resized_img,shape)
    return resized_img


def format_number(num):
    """
    Thus function will transform a number to the **** format
    for example, 12 -> 0012
    :param num: The number
    :return: The number in the right format
    """
    str_num = str(num)
    while len(str_num) < 4:
        str_num = "0%s" % str_num
    return str_num


def get_train_test_sets(shape=(105, 105, 1)):
    """
    This function will return the train and test sets in the form of:
    Same person - ('folder_name',ing_num1,img_num2,1)
    Not the same person - ('folder_name1',ing_num1,'folder_name2',img_num2,0)
    :return: The train & test sets
    """
    train = get_train_set()
    img_train = get_img_set(train, shape=shape)
    test = get_test_set()
    img_test = get_img_set(test, shape=shape)
    return img_train, img_test


def get_conv_weight_initializer():
    """
    This function will return the initializer of the Convolution layers
    :return: Initializer of the Convolution layers
    """
    return keras.initializers.RandomNormal(mean=0, stddev=0.01)


def get_bias_weight_initializer():
    """
        This function will return the initializer of the Convolution layers
        :return: Initializer of the bias
        """
    return keras.initializers.RandomNormal(mean=0.5, stddev=0.01)


def get_fc_weight_initializer():
    """
        This function will return the initializer of the Convolution layers
        :return: Initializer of the FC layers
        """
    return keras.initializers.RandomNormal(mean=0, stddev=0.1)


def build_model(shape):
    """
    This function will assemble a Siamese Neural Networks that compares between inputs with the shape of 'shape'
    :param shape: The shape of the input
    :return: A Siamese Neural Network
    """
    #with tf.device('/CPU:0'):
    conv_initializer = get_conv_weight_initializer()  # The wight initializer for the Convolution layers
    bias_initializer = get_bias_weight_initializer()  # The bias initializer for the biases
    fc_initializer = get_fc_weight_initializer()  # The fc initializer for the fully connected layers
    n_features = 4096
    l2_penalty = 0.001  # The penalty for the L2 regularization
    dropout_prob = 0.2
    dropout = dropout_prob!=0
    model = Sequential()

    # Add a convolution layer with 64 10x10 filters. The activation function is RELU
    model.add(Conv2D(64, (10, 10), activation='relu', input_shape=shape, kernel_initializer=conv_initializer,
                     bias_initializer=bias_initializer, kernel_regularizer=l2(l2_penalty)))
    # MaxPooling (2,2)
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Add a dropout layer
    if dropout:
        model.add(Dropout(dropout_prob))

    # Add a convolution layer with 128 7x7 filters. The activation function is RELU
    model.add(Conv2D(128, (7, 7), activation='relu', kernel_initializer=conv_initializer,
                     bias_initializer=bias_initializer, kernel_regularizer=l2(l2_penalty)))
    # MaxPooling (2,2)
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Add a dropout layer
    if dropout:
        model.add(Dropout(dropout_prob))

    # Add a convolution layer with 128 4x4 filters. The activation function is RELU
    model.add(Conv2D(128, (4, 4), activation='relu', kernel_initializer=conv_initializer,
                     bias_initializer=bias_initializer, kernel_regularizer=l2(l2_penalty)))
    # MaxPooling (2,2)
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Add a dropout layer
    if dropout:
        model.add(Dropout(dropout_prob))

    # Add a convolution layer with 256 4x4 filters. The activation function is RELU
    model.add(Conv2D(256, (4, 4), activation='relu', kernel_initializer=conv_initializer,
                     bias_initializer=bias_initializer, kernel_regularizer=l2(l2_penalty)))
    # Flatten the
    model.add(Flatten())

    # Add a fully connected layer with 4096 units (the number of features in the feature map)
    model.add(
        Dense(n_features, activation='sigmoid', kernel_initializer=fc_initializer, bias_initializer=bias_initializer
              , kernel_regularizer=l2(l2_penalty)))

    # Add a dropout layer
    if dropout:
        model.add(Dropout(dropout_prob))

    # Creating the two twins based in the model
    twin1_input = Input(shape)
    twin2_input = Input(shape)
    twin1 = model(twin1_input)
    twin2 = model(twin2_input)

    # Creating a custom layer rule to merge the twin results using l1 siamese distance
    #
    merge_layer_rule = Lambda(lambda output: backend.abs(output[0] - output[1]))

    # Create the merge layer with twin1 and twin2 as the inputs
    merge_layer = merge_layer_rule([twin1, twin2])

    # Creating the output layer (only one output neuron)
    output_layer = Dense(1, activation='sigmoid', kernel_initializer=fc_initializer
                         , bias_initializer=bias_initializer)(merge_layer)


    # Creating the final network
    final_network = Model(inputs=[twin1_input, twin2_input], outputs=output_layer)

    # Compile network with the loss and optimizer
    final_network.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.00001),
                          metrics=[binary_accuracy])
    # final_network.compile(loss=binary_crossentropy,optimizer=Adam(lr = 0.00001))

    return final_network


def get_train_validation(train, portion):
    """
    This function will divide the train set into validation/train sets
    :param train: The train set
    :param portion: The portion of the validation set
    :return: The validation/train set
    """
    x_train1 = train[0]
    x_train2 = train[1]
    y_train = train[2]
    validation_len = int(len(x_train1) * portion)
    validation_set_indices = random.sample(range(len(x_train1)), validation_len)
    validation_set_indices.sort()
    new_x1_train = []
    new_x2_train = []
    new_y_train = []
    x1_validation = []
    x2_validation = []
    y_validation = []

    validation_index = 0
    for i in range(len(x_train1)):
        if validation_index != validation_len and validation_set_indices[validation_index] == i:
            x1_validation.append(x_train1[i])
            x2_validation.append(x_train2[i])
            y_validation.append(y_train[i])
            validation_index += 1
        else:
            new_x1_train.append(x_train1[i])
            new_x2_train.append(x_train2[i])
            new_y_train.append(y_train[i])
    return [np.array(new_x1_train), np.array(new_x2_train), np.array([new_y_train])], [np.array(x1_validation)
        , np.array(x2_validation), np.array(y_validation)]


def get_batches(batch_size, x1_train, x2_train, y_train):
    """
    Get the list of batches for training
    :param batch_size: size of batch
    :param x_train: train instances
    :param y_train: train labels
    :return: list of batches
    """
    instance_count = len(x1_train)
    batch_num = int(instance_count / batch_size)
    batches = []
    for batch in range(batch_num + 1):
        batch_start = batch * batch_size
        if batch_start == instance_count:
            break  # in case the len of train set is a multiple of batch size
        batch_end = min((batch + 1) * batch_size, instance_count)  # avoid index out of bounds
        x_batch = [x1_train[batch_start:batch_end], x2_train[batch_start:batch_end]]
        y_batch = y_train[:, batch_start:batch_end]
        y_batch = y_batch.reshape(y_batch.shape[1], )
        batches.append([x_batch, y_batch])
    return batches


def train_model(siamese_model, train, validation, batch_size, num_iterations, max_no_improve=100, max_epochs=25):
    x1_train = train[0]
    x2_train = train[1]
    y_train = train[2]
    #x1_validation = validation[0]
    #x2_validation = validation[1]
    #x_validation = [x1_validation, x2_validation]
    #y_validation = validation[2]

    #batches = get_batches(batch_size, x1_train, x2_train, y_train)
    iteration = 0
    best_val_loss = -1
    no_improve = 0
    epoch = 0
    """
    while True:

        for x_batch, y_batch in batches:
            iteration += 1
            print("iteration - %d epoch - %d" % (iteration, epoch))
            history = siamese_model.fit(x=x_batch, y=y_batch)
            print("loss - %s" % history.history["loss"])
            val_lost = siamese_model.evaluate(x=x_validation, y=y_validation)
            y_rrr = siamese_model.predict(x = x_validation)
            print(val_lost)
           # print(y_rrr)
            if best_val_loss == -1 or best_val_loss > val_lost:
                best_val_loss = val_lost
                no_improve = 0
            else:
                no_improve += 1

            if iteration == num_iterations or max_no_improve == no_improve:
                break
        epoch += 1
        if iteration == num_iterations or max_no_improve == no_improve or epoch == max_epochs:
            break
    """
    X_train = [x1_train,x2_train]

    history = siamese_model.fit(x=X_train, y=y_train,validation_split=0.2,epochs=6,batch_size = 32,verbose=2)






def test_prediction(siamese_model, test):
    x1_test = test[0]
    x2_test = test[1]
    x_test = [x1_test, x2_test]
    y_test = test[2]
    val_lost = siamese_model.evaluate(x=x_test, y=y_test)
    print(val_lost)


def test_model():
    # todo: changed the shape since img can be greyscale (much smaller dimensions)
    # img_shape = (105, 105, 3)
    img_shape = (105, 105,1)
    #img_shape = (250,250,1)
    validation_portion = 0.2
    batch_size = 32
    num_of_iterations = 100

    print("Fetching train/test sets...")
    train, test = get_train_test_sets(shape=img_shape)

    print("Fetching validation set...")
  #  train, validation = get_train_validation(train, validation_portion)

    print("Building model...")
    siamese_model = build_model(shape=img_shape)
    siamese_model.summary()

    print("Training model...")
    #train_model(siamese_model, train, validation, batch_size, num_of_iterations)
    train_model(siamese_model, train, None, batch_size, num_of_iterations)
    test_prediction(siamese_model, test)



test_model()
