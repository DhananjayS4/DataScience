import tensorflow as tf
import numpy as np

def create_masks(caption):
    # Padding mask for the caption sequence
    # Explicitly cast to int32 to avoid type mismatch in tf.equal
    caption_idx = tf.cast(caption, tf.int32)
    padding_mask = tf.cast(tf.math.equal(caption_idx, 0), tf.float32)
    padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :] # (batch, 1, 1, seq_len)

    # Look ahead mask to prevent decoder from seeing future tokens
    seq_len = tf.shape(caption)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    look_ahead_mask = tf.cast(look_ahead_mask, tf.float32)

    # Combined mask
    combined_mask = tf.maximum(padding_mask, look_ahead_mask)
    return combined_mask

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img

class CaptionPredictor:
    def __init__(self, model, tokenizer, max_length):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.word2idx = tf.keras.layers.StringLookup(
            mask_token="",
            vocabulary=tokenizer.get_vocabulary())
        self.idx2word = tf.keras.layers.StringLookup(
            mask_token="",
            vocabulary=tokenizer.get_vocabulary(),
            invert=True)

    def greedy_search(self, image):
        image = tf.expand_dims(image, 0)
        start_token = tf.cast(self.word2idx('[start]'), tf.int32)
        end_token = tf.cast(self.word2idx('[end]'), tf.int32)

        output = tf.expand_dims([start_token], 0)

        for i in range(self.max_length):
            combined_mask = create_masks(output)
            # For simplicity in inference, we assume 1 encoder layer or just pass None for padding mask if not used in cross-attention
            # In our models.py, padding_mask in decoder is used for cross-attention over encoder output
            # Since encoder output is image patches, we don't really have a padding mask for it (all patches are valid)
            predictions = self.model(image, output, training=False, look_ahead_mask=combined_mask, padding_mask=None)
            
            # shape (batch_size, seq_len, vocab_size)
            predictions = predictions[:, -1:, :]
            predicted_id = tf.argmax(predictions, axis=-1)
            predicted_id = tf.cast(predicted_id[0, 0], tf.int32)

            if int(predicted_id) == int(end_token):
                break

            output = tf.concat([output, tf.expand_dims([predicted_id], 0)], axis=-1)

        result = []
        for i in range(output.shape[1]):
            word = self.idx2word(output[0, i]).numpy().decode('utf-8')
            if word not in ['[start]', '[end]', '']:
                result.append(word)
        
        return ' '.join(result)

    def beam_search(self, image, beam_width=3):
        image = tf.expand_dims(image, 0)
        start_token = tf.cast(self.word2idx('[start]'), tf.int32)
        end_token = tf.cast(self.word2idx('[end]'), tf.int32)

        # (score, sequence)
        beams = [(0, [start_token])]

        for _ in range(self.max_length):
            new_beams = []
            for score, seq in beams:
                # Use int() for reliable comparison in Python list
                if int(seq[-1]) == int(end_token):
                    new_beams.append((score, seq))
                    continue

                output = tf.expand_dims(seq, 0)
                combined_mask = create_masks(output)
                
                predictions = self.model(image, output, training=False, look_ahead_mask=combined_mask, padding_mask=None)
                predictions = tf.nn.log_softmax(predictions[0, -1, :]) # Log probabilities

                # Get top K candidates
                top_k = tf.math.top_k(predictions, k=beam_width)
                for i in range(beam_width):
                    next_id = tf.cast(top_k.indices[i], tf.int32)
                    next_score = float(top_k.values[i])
                    new_beams.append((score + next_score, seq + [next_id]))

            # Sort and keep top K
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]

            # If all top beams have [end], stop
            if all(int(seq[-1]) == int(end_token) for _, seq in beams):
                break

        # Return best sequence
        best_seq = beams[0][1]
        result = []
        for idx in best_seq:
            word = self.idx2word(idx).numpy().decode('utf-8')
            if word not in ['[start]', '[end]', '']:
                result.append(word)
        
        return ' '.join(result)
