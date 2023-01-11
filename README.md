This project is based on the autoencoder deep learning architecture to extract features from a given dataset. This project was for a course assignement and developed with another student.

Please check [readme.pdf](https://github.com/Giraytd/autoencoder_deep_learning/blob/b3a47c03e6d22f04c9587753f6e11ee1557fc906/readme.pdf) file for the details.

Here are some parts and results of the project.

**Convolutional Architecture for the Autoencoder**
```python
#
# Architecture
#
# Encoder
2
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
```


**Ten most similar images with Manhattan distance**:
![manhattanexample1](https://github.com/Giraytd/autoencoder_deep_learning/blob/b3a47c03e6d22f04c9587753f6e11ee1557fc906/pics/manhattan7.png)

**Ten most similar images with Euclidean distance**:
![eucexample1](https://github.com/Giraytd/autoencoder_deep_learning/blob/b3a47c03e6d22f04c9587753f6e11ee1557fc906/pics/euc8.png)

**UMAP scatter projection of the latent values of the test dataset**:
![umap](https://github.com/Giraytd/autoencoder_deep_learning/blob/b3a47c03e6d22f04c9587753f6e11ee1557fc906/pics/umapscatter.png)
