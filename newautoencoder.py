import seaborn as sns
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Conv2D, Input, Dense, Reshape, Conv2DTranspose, Activation, BatchNormalization, ReLU, Concatenate,Flatten,Dropout,Lambda
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import MaxPooling2D, UpSampling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import cifar100, cifar10
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras import regularizers
from keras.objectives import mean_squared_error
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import umap
from scipy.spatial.distance import cosine
from sklearn.utils import class_weight
import os
os.environ["PATH"] += os.path.expanduser("~/Desktop/somedir/somefile.txt")

#
# Dataset Path
#
cifar10_dataset_folder_path = 'cifar-10-batches-py'

#
# Dataset Load
#
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

#
# Normalize pixel values to be between 0 and 1
#
train_images, test_images = train_images / 255.0, test_images / 255.0

#
# Class names of cifar10 data set.
#
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']




def visualizeResults(history):
    #
    # Plot the results of training.
    #
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    # Accuracy of the full model with encoder and decoder.
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    test_loss, test_acc = feat_extractor.evaluate(test_images, test_labels, verbose=2)

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('pics/lossgraph.png')
    plt.show()

# Adds noise.
def add_noise_and_clip_data(data):
   noise = np.random.normal(loc=0.0, scale=0.1, size=data.shape)
   data = data + noise
   data = np.clip(data, 0., 1.)
   return data

train_images_noisy = add_noise_and_clip_data(train_images)
test_images_noisy = add_noise_and_clip_data(test_images)

# Normalizes the values of features for visualization purposes.
def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (20*(x-min_val) / (max_val-min_val)) -10
    return x

def loss_function(y_true, y_pred):  ## loss function for using in autoencoder models
    mses = mean_squared_error(y_true, y_pred)
    return K.sum(mses, axis=(1,2))

def calc_euc_distance(x1,y1,x2,y2):
    d = np.square(x2 - x1) + np.square(y2 - y1)
    distance = np.sqrt(d)

    return distance

def calc_manhattan_distance(x1,y1,x2,y2):
    d = abs(x1-x2)+abs(y1-y2)

    return d

def calc_cosine_distance(x1,y1,x2,y2):
    vector1 = [x1, y1]
    vector2 = [x2, y2]
    cosine_similarity = 1 - cosine(vector1, vector2)

    return cosine_similarity

# Function for checking the original image and the noisy image
def check_noisy_image(idx):
   plt.subplot(1,2,1)
   plt.imshow(train_images[idx])
   plt.title('Original image')
   plt.subplot(1,2,2)
   plt.imshow(train_images_noisy[idx])
   plt.title('Image with noise')
   plt.show()

check_noisy_image(1)
check_noisy_image(2)


#
# Architecture
#
# Encoder
inp = Input((32, 32,3))
e = Conv2D(32, (3, 3),activation='relu')(inp)
e = MaxPooling2D((2, 2))(e)
e = Conv2D(64, (3, 3),activation='relu')(e)
e = MaxPooling2D((2, 2))(e)
e = Conv2D(128, (3, 3),activation='relu')(e)
l = Flatten()(e)
# Bottleneck layer
bl = Dense(10, name='nrns')(l)
# Decoder
d = Dense(512, activation='relu')(bl)
d = Dense(1024, activation='relu')(d)
d = Dense(2048, activation='relu')(d)
d = Dense(3072, activation='sigmoid')(d)
decoded = Reshape((32,32,3))(d)

#
# Compile the model with Encoder and Decoder
#
model = Model(inp, decoded)
feat_extractor = Model(inp,bl)
# Show the architecture.
model.summary()
feat_extractor.summary()
model.compile(optimizer="adam", loss=loss_function, metrics=['accuracy'])

#
# Checkpoint for training model.
#
filepath="modeldenoisecontfrom50plusepoch5.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]

#
# Checkpoint for encoder model.
#
filepath_encoder="savedencoder25wnormalization.h5"
checkpoint_encoder = ModelCheckpoint(filepath_encoder, monitor='val_accuracy', verbose=1, save_best_only=False, mode='max')
callbacks_list_encoder = [checkpoint_encoder]


#
# Checkpoint loader
#
new_model = tf.keras.models.load_model('savedmodel50.h5', compile=False)
new_model.summary()
model = new_model
model.compile(optimizer="adam", loss='mean_squared_error', metrics=['accuracy'])


#
# Load the saved model for encoder.
#
enc_m = tf.keras.models.load_model('savedencoder25.h5')
enc_m.summary()
feat_extractor = enc_m

#
# Train the model. Given sets depend on tasks.
# For autoencoder:
# history = model.fit(train_images,train_images, epochs=50,callbacks=callbacks_list,validation_data=(test_images,test_images))
# For decoder
# history = feat_extractor.fit(train_images,train_labels, epochs=5, batch_size = 512,callbacks=callbacks_list,validation_data=(test_images,test_labels))
# For noisy:
# history = model.fit(train_images_noisy,train_images, epochs=5,callbacks=callbacks_list,validation_data=(test_images_noisy,test_images))
#


#
# Remove comment for saving the trained model.
#
# model.save('modeldenoisecontfrom50plusepoch5.h5')

#
# Remove comment if there is training and want to visualize results of traing.
#
# visualizeResults(history)


#
# SNS & UMAP
#
print("SNS & UMAP results")
pred_imgs = feat_extractor.predict(test_images[:50])
nmzed = normalize(pred_imgs)
nmzed = pred_imgs
g = sns.pairplot(pd.DataFrame(nmzed))
plt.savefig('pics/scattermatrix.png')
plt.show()
x_test_encoded = feat_extractor.predict(test_images)
plt.figure(figsize=(15, 15))
x_test_encoded = umap.UMAP().fit_transform(x_test_encoded)
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=test_labels)
plt.savefig('pics/umapscatter.png')
plt.show()

#
#Denoise results
#
decoded_imgs = model.predict(test_images_noisy[:20])
plt.figure(figsize =(10,4))
for i in range(0,10):
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(test_images_noisy[i].reshape(32, 32, 3))
    # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(2, 10, i + 1 + 10)
    plt.imshow(decoded_imgs[i].reshape(32, 32, 3))
    # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('pics/denoise.png')
plt.show()
#
# Encode and decode some images
#
decoded_imgs = model.predict(test_images)

#
# Visualize the reconstructed inputs with the original ones from the dataset.
#
# n is the number of images to compare.
n = 20
plt.figure(figsize =(20,4))
for i in range(n):
    # Original images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(test_images[i].reshape(32,32,3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstructed images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(32,32,3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('pics/reconstruct.png')
plt.show()

#
# Query images.
#

#
# Data prep for queries.
#
print("Calculating distances...")
t_images = test_images[:20]
encoded_test_images_20 = feat_extractor.predict(t_images)

encoded_test_images = feat_extractor.predict(test_images[:500])
encoded_train_images = feat_extractor.predict(train_images[:500])
encoded_all_images = np.concatenate((encoded_train_images,encoded_test_images))

encoded_test_images_20 = umap.UMAP().fit_transform(encoded_test_images_20)
encoded_all_images = umap.UMAP().fit_transform(encoded_all_images)

encoded_test_images_20 = pd.DataFrame(encoded_test_images_20)
encoded_all_images = pd.DataFrame(encoded_all_images)

original_all_images = np.append(train_images,test_images,axis=0)

distances = pd.DataFrame(columns={'original_id','compared_id','distance'})

plt.figure(figsize =(20,4))
which_row = 0
for i in range (0,9):
    x1 = encoded_all_images.loc[i, 0]
    y1 = encoded_all_images.loc[i, 1]

    for k in range (0,len(encoded_all_images)):
        x2 = encoded_all_images.loc[k, 0]
        y2 = encoded_all_images.loc[k, 1]

        dist = calc_euc_distance(x1, y1, x2, y2)
        m_dist = calc_manhattan_distance(x1, y1, x2, y2)
        cos_dist = calc_cosine_distance(x1, y1, x2, y2)
        distances.loc[which_row,('distance')] = dist
        distances.loc[which_row, ('mdistance')] = m_dist
        distances.loc[which_row, ('cdistance')] = cos_dist
        distances.loc[which_row,('compared_id')] = k
        distances.loc[which_row,('original_id')] = i
        which_row += 1


n=10
plt.figure(figsize =(10,2))
for k in range(0,9):
    closest_20_euc = distances.loc[distances['original_id'] == k]
    closest_20_euc = closest_20_euc.apply(pd.to_numeric, errors='coerce')
    closest_20_euc = closest_20_euc.nsmallest(20, ('distance'))
    closest_20_euc = closest_20_euc['compared_id']
    closest_20_euc = closest_20_euc.apply(pd.to_numeric, errors='coerce')
    for z in range(0,9):
        ax = plt.subplot(2, n, z + 1)
        plt.imshow(original_all_images[k].reshape(32, 32, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax = plt.subplot(2, n, z + 1 + n)
        getpic = closest_20_euc.iloc[z]
        plt.imshow(original_all_images[getpic].reshape(32, 32, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig('pics/euc'+str(k)+'.png')

for k in range(0,9):
    closest_20_c = distances.loc[distances['original_id'] == k]
    closest_20_c = closest_20_c.apply(pd.to_numeric, errors='coerce')
    closest_20_c = closest_20_c.nsmallest(20, ('cdistance'))
    closest_20_c = closest_20_c['compared_id']
    closest_20_c = closest_20_c.apply(pd.to_numeric, errors='coerce')
    for z in range(0,9):
        ax = plt.subplot(2, n, z + 1)
        plt.imshow(original_all_images[k].reshape(32, 32, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax = plt.subplot(2, n, z + 1 + n)
        getpic = closest_20_c.iloc[z]
        plt.imshow(original_all_images[getpic].reshape(32, 32, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig('pics/cosine'+str(k)+'.png')


for k in range(0,9):
    closest_20_m = distances.loc[distances['original_id'] == k]
    closest_20_m = closest_20_m.apply(pd.to_numeric, errors='coerce')
    closest_20_m = closest_20_m.nsmallest(20, ('mdistance'))
    closest_20_m = closest_20_m['compared_id']
    closest_20_m = closest_20_m.apply(pd.to_numeric, errors='coerce')
    for z in range(0,9):
        # Original images
        ax = plt.subplot(2, n, z + 1)
        plt.imshow(original_all_images[k].reshape(32, 32, 3))
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstructed images
        ax = plt.subplot(2, n, z + 1 + n)
        getpic = closest_20_m.iloc[z]
        plt.imshow(original_all_images[getpic].reshape(32, 32, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig('pics/manhattan'+str(k)+'.png')















