# Save this in D:\DataScience\generate_vocab.py
import tensorflow as tf
import os
import pandas as pd
import json

DATA_PATH = 'd:/DataScience/data'
CAPTIONS_FILE = f'{DATA_PATH}/captions.txt'

if os.path.exists(CAPTIONS_FILE):
    df = pd.read_csv(CAPTIONS_FILE)
    tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=10000,
        standardize=None,
        output_sequence_length=40)
    
    captions = df['caption'].apply(lambda x: '[start] ' + x.lower() + ' [end]').values
    tokenizer.adapt(captions)
    
    vocab = tokenizer.get_vocabulary()
    with open('vocab.json', 'w') as f:
        json.dump(vocab, f)
    print("✅ vocab.json generated successfully!")
else:
    print("❌ Could not find captions.txt")
