import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
import os
import pathlib

max_length = 128      # Maximum length of input sentence to the model.
batch_size = 32
epochs = 2

phase1 = "Phase 1"
phase2 = "Phase 2"

# Labels in our dataset.
labels = ["contradiction", "entailment", "neutral"]

class TrainingData:
    def __init__(self):
        self.LoadTrainingData()

    def LoadTrainingData(self):
        self.train_df = pd.read_csv("SNLI_Corpus/snli_1.0_train.csv", nrows=100000)
        self.valid_df = pd.read_csv("SNLI_Corpus/snli_1.0_dev.csv")
        self.test_df = pd.read_csv("SNLI_Corpus/snli_1.0_test.csv")

        # Shape of the data
        print(f"Total train samples : {self.train_df.shape[0]}")
        print(f"Total validation samples: {self.valid_df.shape[0]}")
        print(f"Total test samples: {self.valid_df.shape[0]}")

        print(f"Sentence1: {self.train_df.loc[1, 'sentence1']}")
        print(f"Sentence2: {self.train_df.loc[1, 'sentence2']}")
        print(f"Similarity: {self.train_df.loc[1, 'similarity']}")

        print("Number of missing values")
        print(self.train_df.isnull().sum())
        self.train_df.dropna(axis=0, inplace=True)

        print("Train Target Distribution")
        print(self.train_df.similarity.value_counts())

        print("Validation Target Distribution")
        print(self.valid_df.similarity.value_counts())

        self.train_df = (
            self.train_df[self.train_df.similarity != "-"]
            .sample(frac=1.0, random_state=42)
            .reset_index(drop=True)
        )
        self.valid_df = (
            self.valid_df[self.valid_df.similarity != "-"]
            .sample(frac=1.0, random_state=42)
            .reset_index(drop=True)
        )

        self.train_df["label"] = self.train_df["similarity"].apply(
            lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
        )
        self.y_train = tf.keras.utils.to_categorical(self.train_df.label, num_classes=3)

        self.valid_df["label"] = self.valid_df["similarity"].apply(
            lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
        )
        self.y_val = tf.keras.utils.to_categorical(self.valid_df.label, num_classes=3)

        self.test_df["label"] = self.test_df["similarity"].apply(
            lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
        )
        self.y_test = tf.keras.utils.to_categorical(self.test_df.label, num_classes=3)



class BertSemanticDataGenerator(tf.keras.utils.Sequence):
    """Generates batches of data.

    Args:
        sentence_pairs: Array of premise and hypothesis input sentences.
        labels: Array of labels.
        batch_size: Integer batch size.
        shuffle: boolean, whether to shuffle the data.
        include_targets: boolean, whether to incude the labels.

    Returns:
        Tuples `([input_ids, attention_mask, `token_type_ids], labels)`
        (or just `[input_ids, attention_mask, `token_type_ids]`
         if `include_targets=False`)
    """

    def __init__(
        self,
        sentence_pairs,
        labels,
        batch_size=batch_size,
        shuffle=True,
        include_targets=True,
    ):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_targets = include_targets
        # Load our BERT Tokenizer to encode the text.
        # We will use base-base-uncased pretrained model.
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.indexes = np.arange(len(self.sentence_pairs))
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch.
        return len(self.sentence_pairs) // self.batch_size

    def __getitem__(self, idx):
        # Retrieves the batch of index.
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        sentence_pairs = self.sentence_pairs[indexes]

        # With BERT tokenizer's batch_encode_plus batch of both the sentences are
        # encoded together and separated by [SEP] token.
        encoded = self.tokenizer.batch_encode_plus(
            sentence_pairs.tolist(),
            add_special_tokens=True,
            max_length=max_length,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors="tf",
        )

        # Convert batch of encoded features to numpy array.
        input_ids = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

        # Set to true if data generator is used for training/validation.
        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="int32")
            return [input_ids, attention_masks, token_type_ids], labels
        else:
            return [input_ids, attention_masks, token_type_ids]

    def on_epoch_end(self):
        # Shuffle indexes after each epoch if shuffle is set to True.
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)


class SemanticComparer:
    def __init__(self):
        self.backuppath = pathlib.Path(__file__).parent.joinpath("semantic_comparer_backup")
        self.modelpath = self.backuppath.joinpath("semantic_comparer_model").absolute()
        self.checkpointfile = self.backuppath.joinpath("checkpoint.txt").absolute()
        
        self.CreateModel()

    def CreateModel(self):
        print("Creating Model")
        # Create the model under a distribution strategy scope.
        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            # Encoded token ids from BERT tokenizer.
            input_ids = tf.keras.layers.Input(
                shape=(max_length,), dtype=tf.int32, name="input_ids"
            )
            # Attention masks indicates to the model which tokens should be attended to.
            attention_masks = tf.keras.layers.Input(
                shape=(max_length,), dtype=tf.int32, name="attention_masks"
            )
            # Token type ids are binary masks identifying different sequences in the model.
            token_type_ids = tf.keras.layers.Input(
                shape=(max_length,), dtype=tf.int32, name="token_type_ids"
            )
            # Loading pretrained BERT model.
            self.bert_model = transformers.TFBertModel.from_pretrained("bert-base-uncased")
            # Freeze the BERT model to reuse the pretrained features without modifying them.
            self.bert_model.trainable = False

            sequence_output, pooled_output = self.bert_model(
                input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
            )
            # Add trainable layers on top of frozen layers to adapt the pretrained features on the new data.
            bi_lstm = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(64, return_sequences=True)
            )(sequence_output)
            # Applying hybrid pooling approach to bi_lstm sequence output.
            avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_lstm)
            max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_lstm)
            concat = tf.keras.layers.concatenate([avg_pool, max_pool])
            dropout = tf.keras.layers.Dropout(0.3)(concat)
            output = tf.keras.layers.Dense(3, activation="softmax")(dropout)
            self.model = tf.keras.models.Model(
            inputs=[input_ids, attention_masks, token_type_ids], outputs=output
            )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss="categorical_crossentropy",
            metrics=["acc"],
        )


        print(f"Strategy: {strategy}")
        self.model.summary
        print("Model Created")

    def Train(self, trainingData):
        print("Start Model Training")
        phase = self.GetPhaseFromCheckpoint()
        if not phase:
            self.TrainPhase1(trainingData)
            self.TrainPhase2(trainingData)
        else:
            self.Reload()
            if phase == phase1:
                print("Training Phase 1 already performed")
                self.TrainPhase2(trainingData)
            elif phase == phase2:
                print("Network Already Trained")
        print("Training Completed")

    def GetPhaseFromCheckpoint(self):
        if not os.path.exists(self.checkpointfile):
            return None
        
        phase = ""
        with open(self.checkpointfile,'r') as f:
            for line in f.readlines():
                phase = line
            f.close()
        return phase


    def SetCheckPoint(self, phase):
        with open(self.checkpointfile,'w') as f:
            f.write(phase)
            f.close()
    
    def TrainPhase1(self, trainingData):
        print("Starting Training Phase 1")
        train_data = BertSemanticDataGenerator(
            trainingData.train_df[["sentence1", "sentence2"]].values.astype("str"),
            trainingData.y_train,
            batch_size=batch_size,
            shuffle=True,
        )
        valid_data = BertSemanticDataGenerator(
            trainingData.valid_df[["sentence1", "sentence2"]].values.astype("str"),
            trainingData.y_val,
            batch_size=batch_size,
            shuffle=False,
        )

        self.history = self.model.fit(
            train_data,
            validation_data=valid_data,
            epochs=epochs,
            use_multiprocessing=True,
            workers=-1,
        )

        self.Save()
        self.SetCheckPoint(phase1)
        print("Training Phase 1 Completed")

    def TrainPhase2(self, trainingData):
        print("Starting Training Phase 2")
        train_data = BertSemanticDataGenerator(
            trainingData.train_df[["sentence1", "sentence2"]].values.astype("str"),
            trainingData.y_train,
            batch_size=batch_size,
            shuffle=True,
        )
        valid_data = BertSemanticDataGenerator(
            trainingData.valid_df[["sentence1", "sentence2"]].values.astype("str"),
            trainingData.y_val,
            batch_size=batch_size,
            shuffle=False,
        )

        # Unfreeze the bert_model.
        self.bert_model.trainable = True
        # Recompile the model to make the change effective.
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.model.summary()

        history = self.model.fit(
            train_data,
            validation_data=valid_data,
            epochs=epochs,
            use_multiprocessing=True,
            workers=-1,
        )

        self.Save()

        test_data = BertSemanticDataGenerator(
            trainingData.test_df[["sentence1", "sentence2"]].values.astype("str"),
            trainingData.y_test,
            batch_size=batch_size,
            shuffle=False,
        )
        self.model.evaluate(test_data, verbose=1)

        self.Save()
        self.SetCheckPoint(phase2)
        print("Training Phase 2 Completed")


    def check_similarity(self,sentence1, sentence2):
        sentence_pairs = np.array([[str(sentence1), str(sentence2)]])
        test_data = BertSemanticDataGenerator(
            sentence_pairs, labels=None, batch_size=1, shuffle=False, include_targets=False,
        )

        proba = self.model.predict(test_data)[0]
        idx = np.argmax(proba)
        proba = f"{proba[idx]: .2f}%"
        pred = labels[idx]
        return pred, proba

    def TestNetwork(self):
        sentence1 = "Two women are observing something together."
        sentence2 = "Two women are standing with their eyes closed."
        print(self.check_similarity(sentence1, sentence2))

        sentence1 = "A smiling costumed woman is holding an umbrella"
        sentence2 = "A happy woman in a fairy costume holds an umbrella"
        print(self.check_similarity(sentence1, sentence2))

        sentence1 = "A soccer game with multiple males playing"
        sentence2 = "Some men are playing a sport"
        print(self.check_similarity(sentence1, sentence2))
    
    def Reload(self):
        print("Traying to Reload Model")
        if os.path.exists(self.modelpath):
            self.model = tf.keras.models.load_model(str(self.modelpath))

            # Check its architecture
            self.model.summary()
            print("Model Successfully Reloaded")
        else:
            print("No model found")

    def Save(self):
        print("Saving Model")
        if not os.path.exists(self.modelpath):
            os.makedirs(self.modelpath,exist_ok=True)
        self.model.save(str(self.modelpath))
        print("Model Successfully Saved")
        
            


def main():
    trainingData = TrainingData()

    semanticComparer = SemanticComparer()
    try:
        semanticComparer.Reload()
    except:
        print("It was not possible to reload data from previous training")
    
    semanticComparer.Train(trainingData)

    semanticComparer.TestNetwork()

if __name__ == '__main__':
    main()
