from BerTimbau import TFModel, Tokenizer, Optimizer #ExportModel, export_model
from DataLoading import ModelData
import pandas as pd
import os
import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint



TENSORBOARD_DIR : str = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
CHECKPOINT_PATH : str = "../results/models/checkpoint_best"
IMAGES_PATH : str = "./results/figures"
DATA_PATH : str = "./data/"
CHECKPOINT_PATH = "../results/models/checkpoint_best"

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    """
        This function takes the current figure inside the matplotlib
        buffer and saves it to a file,
        cleaning the buffer afterwards.

        Usage:
            Do your matplotlib.pyplot pipeline as usual, but instead of
            calling plt.plot(), you just call save_fig(<name_of_the_file>)

    """
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)



def main():


    #Data loading
    main_data = ModelData(
        os.path.join(DATA_PATH, "trainingAppreciative.csv")
    )
    train_X, test_X,train_labels, test_labels = main_data.get_train_test_splits()
    train_X = np.array(train_X)
    train_labels = np.array(train_labels)


    train_data = []
    optimizer = Optimizer(
        epochs=5,
        batch_size=32,
        eval_batch_size=32,
        train_data_size=len(train_data),
    ).optimizer()


    checkpoint = ModelCheckpoint(
        filepath = CHECKPOINT_PATH,
        save_weights_only=False,
        save_freq="epoch",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_DIR, histogram_freq=1)


    #create the wrapper hugging face transformer model
    model = TFModel("neuralmind/bert-base-portuguese-cased")
    #compile wrapper
    model.compile(optimizer)
    #training step
    history = model.fit(train_X,train_labels,callbacks=[checkpoint,tensorboard_callback])


    train_df = pd.DataFrame(history.history)
    train_df.plot(y=['accuracy', 'val_accuracy'])
    save_fig("history_final_model")


if __name__ == "__main__":
    main()
