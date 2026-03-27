import tensorflow as tf
import os
import pandas as pd
from models import ImageCaptioningModel
from utils import load_image, create_masks
import json

# --- Configuration ---
DATA_PATH = 'd:/DataScience/data' # Adjust if needed
CAPTIONS_FILE = f'{DATA_PATH}/captions.txt'
IMAGES_DIR = f'{DATA_PATH}/Images'

VOCAB_SIZE = 10000
MAX_LENGTH = 40
BATCH_SIZE = 32
EMBED_DIM = 256
NUM_LAYERS = 4
NUM_HEADS = 8
FF_DIM = 512
EPOCHS = 20

# --- Data Preparation ---
def preprocess_text(text):
    text = text.lower()
    text = tf.strings.regex_replace(text, r'[^\w\s]', '')
    text = tf.strings.join(['[start]', text, '[end]'], separator=' ')
    return text

def load_dataset():
    if not os.path.exists(CAPTIONS_FILE):
        print(f"Error: Captions file not found at {CAPTIONS_FILE}")
        return None
    
    df = pd.read_csv(CAPTIONS_FILE)
    df['image'] = df['image'].apply(lambda x: f'{IMAGES_DIR}/{x}')
    
    tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        standardize=None,
        output_sequence_length=MAX_LENGTH)
    
    captions = df['caption'].apply(lambda x: '[start] ' + x.lower() + ' [end]').values
    tokenizer.adapt(captions)
    
    def map_func(img_path, cap):
        img = load_image(img_path)
        cap = tokenizer(cap)
        return img, cap

    dataset = tf.data.Dataset.from_tensor_slices((df['image'].values, captions))
    dataset = dataset.map(map_func, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return dataset, tokenizer

# --- Model & Training Setup ---
dataset, tokenizer = load_dataset()

model = ImageCaptioningModel(
    num_layers=NUM_LAYERS,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
    vocab_size=VOCAB_SIZE,
    max_len=MAX_LENGTH
)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

# Learning Rate Scheduler: Cosine Decay
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.0001,
    decay_steps=10000)

optimizer = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# --- Training Loop ---
@tf.function
def train_step(img_tensor, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    combined_mask = create_masks(tar_inp)

    with tf.GradientTape() as tape:
        predictions = model(img_tensor, tar_inp, training=True, look_ahead_mask=combined_mask, padding_mask=None)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

def train():
    for epoch in range(EPOCHS):
        total_loss = 0
        for (batch, (img_tensor, tar)) in enumerate(dataset):
            batch_loss = train_step(img_tensor, tar)
            total_loss += batch_loss

            if batch % 100 == 0:
                print(f'Epoch {epoch+1} Batch {batch} Loss {batch_loss.numpy():.4f}')

        print(f'Epoch {epoch+1} Total Loss {total_loss/len(dataset):.4f}')
        model.save_weights(f'weights_epoch_{epoch+1}.weights.h5')

if __name__ == "__main__":
    if dataset:
        # Save vocabulary early so UI can use it during training
        vocab = tokenizer.get_vocabulary()
        with open('vocab.json', 'w') as f:
            json.dump(vocab, f)
        print("Vocabulary saved to vocab.json")
        
        print("Starting training...")
        train()
        # Save final weights
        model.save_weights('weights.weights.h5')
        print("Training complete. Weights saved.")
