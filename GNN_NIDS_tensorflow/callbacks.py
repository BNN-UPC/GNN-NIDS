import tensorflow as tf

def get_model_checkpoint(checkpoint_dir, save_freq):
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir, save_freq=save_freq, save_best_only=True
    )
