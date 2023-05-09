from transformers import AutoTokenizer, TFAutoModel
from typing import Any, Callable
from transformers import TFAutoModelForSequenceClassification
from tensorflow.keras.optimizers import Adam, AdamW

import tensorflow_models as tfm
import tensorflow as tf
import os


class Tokenizer:
    """
        This class hold the tokenizer and helper functions to make
        it easier and more intuitive for the use case in this work
    """
    def __init__(self, tokenizer : AutoTokenizer | Callable | Any = None):
        """
            initialize with the main tokenizer object

            params:
                tokenizer : AutoTokenizer from transformers

            return:
                A Tokenizer object.
        """
        if tokenizer is not None:
            self.tokenizer : AutoTokenizer = tokenizer

    def build_tokenizer(self,model_name: str):
        """
            load the chosen model from Hugging face gallery
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        

    def get_tokenizer(self) -> AutoTokenizer:
        """
            Get the AutoTokenizer Object stored here
            return:
                The tokenizer object
        """
        return self.tokenizer

    def tokenize_function(self, data : list):
        """
            Responsible for tokenizing the input, this is the steps where the 
            words, sentences and punctuation are converted into tokens.
            It's important to be sure that the steps below are compatible the
            model that you've chosen.

            params:
                - data : a python list containing the data to be tokenized
            returns: 
                - a dictionary containing the tokenized data, ready for training.
        """
        tokenized_data =  self.tokenizer(
            data,
            return_tensors="np",
            max_length=128, #hyperparameter, apparently
            truncation=True,
            padding=True
        )
        return dict(tokenized_data)


class TFModel:
    def __init__(self, model_name: str):
        self.model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
        self.metrics = [tf.keras.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def compile(
        self,
        optimizer : str | Any ,
        # loss_func: str | Any,
        # metrics: Any
        ):
        self.model.compile(optimizer, loss=self.loss, metrics=self.metrics)

    def fit(self, tokenized_data : Any, labels: Any,callbacks : list = []):
        self.model.fit(
            x=tokenized_data,
            y=labels,
            validation_split=0.2,
            callbacks=callbacks
        )

    def evaluate(self, validation_data : Any):
        self.model.evaluate(validation_data)



class Optimizer:
    def __init__(self,
                 epochs : int,
                 batch_size : int, 
                 eval_batch_size : int,
                 train_data_size : int,
                 initial_learning_rate : float = 2e-5):

        self.epochs = epochs,
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.initial_learning_rate = initial_learning_rate
        self.train_data_size = train_data_size
        self.warmup_steps = int(0.1*train_data_size)
        self.steps_per_epoch = int(train_data_size)
        self.num_train_steps = self.steps_per_epoch * epochs


    def get_linear_decay(self):
        linear_decay = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate = self.initial_learning_rate,
            end_learning_rate = 0,
            decay_steps=self.num_train_steps
        )
        return linear_decay

    def get_warmup_schedule(self):
        warmup_schedule = tfm.optimization.lr_schedule.LinearWarmup(
            warmup_learning_rate = 0,
            after_warmup_lr_sched = self.get_linear_decay(),
            warmup_steps = self.warmup_steps
        )
        return warmup_schedule

    def optimizer(self) -> Any:
        optimizer = tf.keras.optimizers.experimental.AdamW(
            learning_rate = self.get_warmup_schedule()
        )
        return optimizer


class ExportModel(tf.Module):
    def __init__(self, input_processor, classifier):
        self.input_processor = input_processor
        self.classifier = classifier

    @tf.function(
        input_signature = [{
            "sentence" : tf.TensorSpec(shape=[None], dtype=tf.string)
        }]
    )
    def __call__(self,inputs,labels):
        packed = self.input_processor(inputs)
        logits = self.classifier(packed, training=False)
        result_cls_ids = tf.argmax(logits)

        return {
            'logits' : logits,
            'class_id' : result_cls_ids,
            "class" : tf.gather(
                tf.constant(labels),
                result_cls_ids
            )
        }






def export_model(model,model_name):
    export_dir = os.mkdir(f"../result/tf_saved_model_{model_name}")
    tf.saved_model.save(export_model, export_dir=export_dir,
                        signatures={"serving_default" : export_model.__call__}] 


