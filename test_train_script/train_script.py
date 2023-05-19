import numpy as np
import tensorflow as tf
import pandas as pd

from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from tensorflow.keras.optimizers import Adam


from datasets import load_dataset

import datetime
from datasets import Dataset


dataframe = pd.read_csv("../data/trainingAppreciative.csv", sep="|")
dataframe_test = dataframe[:100]
dataframe_train = dataframe[100:]

dataset = Dataset.from_pandas(dataframe_train)
dataset_test = Dataset.from_pandas(dataframe_test)



tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
tokenized_data = tokenizer(dataset["sentence"], return_tensors="np", padding=True)
tokenized_test_data = tokenizer(dataset_test["sentence"], return_tensors="np", padding=True)

tokenized_data = dict(tokenized_data)
tokenized_test_data = dict(tokenized_test_data)



log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

labels = np.array(dataset["appreciative"])
labels_test = np.array(dataset_test["appreciative"])


model = TFAutoModelForSequenceClassification.from_pretrained("neuralmind/bert-base-portuguese-cased")
model.compile(optimizer=Adam(3e-5))

model.fit(tokenized_data, labels,validation_data=(tokenized_test_data, labels_test),callbacks=[tensorboard_callback])


model.save_pretrained("tensorflow-test/appreciative")

