import tensorflow as tf

class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # Using InceptionV3 as the base model
        self.inception_v3 = tf.keras.applications.InceptionV3(
            include_top=False,
            weights='imagenet'
        )
        # Unfreeze the last 50 layers for fine-tuning
        for layer in self.inception_v3.layers[:-50]:
            layer.trainable = False
        for layer in self.inception_v3.layers[-50:]:
            layer.trainable = True

        self.dense = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        features = self.inception_v3(x)
        # shape: (batch_size, 8, 8, 2048) -> (batch_size, 64, 2048)
        features = tf.reshape(features, (features.shape[0], -1, features.shape[3]))
        return self.dense(features)

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training):
        attn_output = self.mha(query=x, value=x, key=x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.mha2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.dropout3 = tf.keras.layers.Dropout(dropout)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(query=x, value=x, key=x, attention_mask=look_ahead_mask, training=training, return_attention_scores=True)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(query=out1, value=enc_output, key=enc_output, attention_mask=padding_mask, training=training, return_attention_scores=True)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2

class ImageCaptioningModel(tf.keras.Model):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, vocab_size, max_len, dropout=0.1):
        super().__init__()
        self.cnn_encoder = CNN_Encoder(embed_dim)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.pos_encoding = self.positional_encoding(max_len, embed_dim)
        
        self.enc_layers = [TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        self.dec_layers = [TransformerDecoderLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def positional_encoding(self, position, d_model):
        import numpy as np
        def get_angles(pos, i, d_model):
            angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
            return pos * angle_rates

        angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, img, caption, training, look_ahead_mask, padding_mask):
        # Encoder
        enc_output = self.cnn_encoder(img) # (batch, 64, embed_dim)
        for i in range(len(self.enc_layers)):
            enc_output = self.enc_layers[i](enc_output, training=training)

        # Decoder
        seq_len = tf.shape(caption)[1]
        x = self.embedding(caption) # (batch, seq_len, embed_dim)
        x *= tf.math.sqrt(tf.cast(tf.shape(x)[-1], tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(len(self.dec_layers)):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training=training, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)

        return self.final_layer(x)
