import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, Flatten, Dense, Input, Lambda, Dropout
from keras.layers.pooling import MaxPool2D
from keras.regularizers import l2
from keras import backend
from keras.optimizers import SGD, RMSprop,Adam
from keras.losses import binary_crossentropy
import cv2
import pathlib
import random
import numpy as np




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
    resized_img = np.reshape(resized_img, shape)
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
    return keras.initializers.RandomNormal(mean=0, stddev=0.01,seed=1)


def get_bias_weight_initializer():
    """
        This function will return the initializer of the Convolution layers
        :return: Initializer of the bias
        """
    return keras.initializers.RandomNormal(mean=0.5, stddev=0.01,seed=1)


def get_fc_weight_initializer():
    """
        This function will return the initializer of the Convolution layers
        :return: Initializer of the FC layers
        """
    return keras.initializers.RandomNormal(mean=0, stddev=0.2,seed=1)


def build_model(shape):
    """
    This function will assemble a Siamese Neural Networks that compares between inputs with the shape of 'shape'
    :param shape: The shape of the input
    :return: A Siamese Neural Network
    """


    conv_initializer = get_conv_weight_initializer()  # The wight initializer for the Convolution layers
    bias_initializer = get_bias_weight_initializer()  # The bias initializer for the biases
    fc_initializer = get_fc_weight_initializer()  # The fc initializer for the fully connected layers

    n_features = 4096
    l2_penalty = 0.00001  # The penalty for the L2 regularization
    dropout_prob = 0.1
    dropout = dropout_prob!=0
    optimizer = Adam(lr=0.00001)
    #optimizer = SGD(lr=0.00001)
    #optimizer = RMSprop(lr=0.00001)
    model = Sequential()

    # Add a convolution layer with 64 10x10 filters. The activation function is RELU
    model.add(Conv2D(64, (10, 10), activation='relu', input_shape=shape, kernel_initializer=conv_initializer,
                     bias_initializer=bias_initializer, kernel_regularizer=l2(l2_penalty)))

    # Add a dropout layer
    if dropout:
     model.add(Dropout(dropout_prob))

    # MaxPooling (2,2)
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Add a convolution layer with 128 7x7 filters. The activation function is RELU
    model.add(Conv2D(128, (7, 7), activation='relu', kernel_initializer=conv_initializer,
                     bias_initializer=bias_initializer, kernel_regularizer=l2(l2_penalty)))

    # Add a dropout layer
    if dropout:
        model.add(Dropout(dropout_prob))

    # MaxPooling (2,2)
    model.add(MaxPool2D(pool_size=(2, 2)))



    # Add a convolution layer with 128 4x4 filters. The activation function is RELU
    model.add(Conv2D(128, (4, 4), activation='relu', kernel_initializer=conv_initializer,
                     bias_initializer=bias_initializer, kernel_regularizer=l2(l2_penalty)))

    # Add a dropout layer
    if dropout:
        model.add(Dropout(dropout_prob))


    # MaxPooling (2,2)
    model.add(MaxPool2D(pool_size=(2, 2)))


    # Add a convolution layer with 256 4x4 filters. The activation function is RELU
    model.add(Conv2D(256, (4, 4), activation='relu', kernel_initializer=conv_initializer,
                     bias_initializer=bias_initializer, kernel_regularizer=l2(l2_penalty)))
    # Flatten the
    model.add(Flatten())

    # Add a dropout layer
    #if dropout:
     #   model.add(Dropout(dropout_prob))

    # Add a fully connected layer with 4096 units (the number of features in the feature map)
    model.add(
        Dense(n_features, activation='sigmoid', kernel_initializer=fc_initializer, bias_initializer=bias_initializer
              , kernel_regularizer=l2(l2_penalty)))

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

    final_network.compile(loss=binary_crossentropy,
                          metrics=["accuracy"],optimizer=optimizer)


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
        y_batch = y_batch.reshape(y_batch.shape[1],1 )
        batches.append([x_batch, y_batch])
    return batches


def train_model(siamese_model, train, validation_split, batch_size, epochs):
    """
    This function will train the model
    :param siamese_model: The given model
    :param train: The train set
    :param validation_split: The validation_split (the portion of the validation out of the train set)
    :param batch_size: The batch size
    :param epochs: The number of epochs
    :return: The history of the model training
    """
    x1_train = train[0]
    x2_train = train[1]
    y_train = train[2]
    X_train = [x1_train,x2_train]

    history = siamese_model.fit(x=X_train, y=y_train,validation_split=validation_split,batch_size=batch_size,epochs=epochs,verbose=2)

    y_r = siamese_model.predict(X_train)
    print(binary_crossentropy(y_true=y_train,y_pred=y_r))
    return history.history




def test_prediction(siamese_model, test):
    """
    This function will evaluate the model using the test set
    :param siamese_model: The given model
    :param test: The given test set
    :return: The test loss & accuracy
    """
    x1_test = test[0]
    x2_test = test[1]
    x_test = [x1_test, x2_test]
    y_test = test[2]
    score = siamese_model.evaluate(x=x_test, y=y_test)
    loss = score[0]
    acc = score[1]
    return loss,acc

def write_to_file(history,test_loss,test_acc):
    """
    This function will write the results to the file
    :param history: The history of the model training
    :param test_loss: The test loss
    :param test_acc: The test accuracy
    :return:
    """
    to_file = []
    header = "epoch"
    scores = []
    epoch = 0
    for key in history:
        header = "%s,%s" % (header, key)
        scores.append(history[key])

    to_file.append(header)
    for i in range(len(scores[0])):
        epoch += 1
        line = "%d" % epoch
        for j in range(len(scores)):
            line = "%s,%s" % (line, scores[j][i])
        to_file.append(line)
    test_line = "test_loss,%s,test_accuracy,%s" % (test_loss, test_acc)
    to_file.append(test_line)

    file = open('results.csv', 'w')
    for i in range(len(to_file)):
        file.write("%s\n" % to_file[i])

def test_model():
    """
    This funciton will test the model
    The function will fetch the data, build the model, train it and test it
    :return:
    """
    img_shape = (105, 105,1)
    validation_portion = 0.2
    batch_size = 32
    epochs = 10

    print("Fetching train/test sets...")
    train, test = get_train_test_sets(shape=img_shape)


    print("Building model...")
    siamese_model = build_model(shape=img_shape)
    siamese_model.summary()

    print("Training model...")
    history = train_model(siamese_model, train, validation_portion, batch_size, epochs)
    test_loss, test_acc = test_prediction(siamese_model, test)
    write_to_file(history,test_loss,test_acc)





test_model()
