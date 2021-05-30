import numpy as np
import os

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt

class imdbReviews:
    def __init__(self):
        train_data, test_data = tfds.load(name="imdb_reviews", split=["train", "test"], 
                                  batch_size=-1, as_supervised=True)

        train_examples, train_labels = tfds.as_numpy(train_data)
        test_examples, test_labels = tfds.as_numpy(test_data)
        self.train_examples = train_examples
        self.train_labels = train_labels
        self.test_examples = test_examples
        self.test_labels = test_labels
        self.train_data = train_data
        self.test_data = test_data

        print("Training entries: {}, test entries: {}".format(len(train_examples), len(test_examples)))

    


class RedeNeural:
    def __init__(self,imdb):
        print("Version: ", tf.__version__)
        print("Eager mode: ", tf.executing_eagerly())
        print("Hub version: ", hub.__version__)
        print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

        self.checkpoint_path = "training_1/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)


        self.imdb = imdb

        self.CreateModel()
        
    
    def CreateModel(self):
        model = "https://tfhub.dev/google/nnlm-en-dim50/2"
        hub_layer = hub.KerasLayer(model, input_shape=[], dtype=tf.string, trainable=True)
        hub_layer(self.imdb.train_examples[:3])

        self.model = tf.keras.models.Sequential()
        self.model.add(hub_layer)
        self.model.add(tf.keras.layers.Dense(16, activation='relu'))
        self.model.add(tf.keras.layers.Dense(1))

        self.model.summary()

        self.model.compile(optimizer='adam',
              loss=tf.losses.BinaryCrossentropy(from_logits=True),
              metrics=[tf.metrics.BinaryAccuracy(threshold=0.0, name='accuracy')])


    def TrainNetwork(self):

        x_val = self.imdb.train_examples[:10000]
        partial_x_train = self.imdb.train_examples[10000:]

        y_val = self.imdb.train_labels[:10000]
        partial_y_train = self.imdb.train_labels[10000:]

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

        self.history = self.model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1,
                    callbacks=[cp_callback])

        
    def PlotTraining(self):
        history_dict = self.history.history
        history_dict.keys()

        acc = history_dict['accuracy']
        val_acc = history_dict['val_accuracy']
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']

        epochs = range(1, len(acc) + 1)

        # "bo" is for "blue dot"
        plt.plot(epochs, loss, 'bo', label='Training loss')
        # b is for "solid blue line"
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()


        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()

    def Evaluate(self):
        results = self.model.evaluate(self.imdb.test_data, self.imdb.test_labels)

        print(results)
    
    def Reload(self):
        latest = tf.train.latest_checkpoint(self.checkpoint_dir)
        self.CreateModel()
        self.model.load_weights(self.checkpoint_path)





