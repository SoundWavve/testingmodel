import tensorflow as tf
import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc, delta

def extract_features(wav_path, num_features=25, max_length=1220):
    assert num_features % 3 == 1, "num_features must be set so that num_features % 3 == 1"
    rate, sig = wavfile.read(wav_path)
    num_mfcc_features = (num_features - 1) // 3
    mfcc_feat = mfcc(sig, rate, numcep=num_mfcc_features)
    d_mfcc_feat = delta(mfcc_feat, 2)
    dd_mfcc_feat = delta(d_mfcc_feat, 2)
    features = np.concatenate((mfcc_feat, d_mfcc_feat, dd_mfcc_feat), axis=1)
    extra_feature = np.zeros((features.shape[0], 1))
    features = np.concatenate((features, extra_feature), axis=1)
    print("Original feature shape:", features.shape)
    if features.shape[0] > max_length:
        features = features[:max_length, :]
    else:
        pad_width = max_length - features.shape[0]
        features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
    print("Padded/truncated feature shape:", features.shape)
    return features.reshape(1, max_length, num_features), rate

def write_wav(wav_path, rate, data):
    data = np.asarray(data, dtype=np.int16)
    print("Audio data range:", data.min(), data.max())
    print("Audio data mean:", data.mean())
    print("Audio data length:", len(data))
    wavfile.write(wav_path, rate, data)

input_wav = 'source.wav'
output_wav = 'output.wav'
meta_path = 'mfcc_model_epoch_3129.meta'
data_path = 'mfcc_model_epoch_3129.data-00000-of-00001'

with tf.compat.v1.Session() as sess:
    input_features, rate = extract_features(input_wav)
    saver = tf.compat.v1.train.import_meta_graph(meta_path)
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    graph = tf.compat.v1.get_default_graph()
    input_tensor = graph.get_tensor_by_name("Placeholder:0")
    output_tensor = graph.get_tensor_by_name("Tanh_1:0")
    output_features = sess.run(output_tensor, feed_dict={input_tensor: input_features})
    
    print("Output features shape:", output_features.shape)
    print("Output features range:", output_features.min(), output_features.max())
    
    # Ensure output features have a proper shape for audio
    if output_features.ndim == 3:
        output_features = np.mean(output_features, axis=-1)
    
    # Flatten the output features and process them as an audio signal
    output_features = output_features.flatten()
    print("Flattened output features shape:", output_features.shape)

    # Normalize the output to the range of int16
    output_features = np.int16(output_features / np.max(np.abs(output_features)) * 32767)
    
    # Ensure the length matches the expected 1220 samples for 3 seconds duration
    expected_length = 1220
    if len(output_features) > expected_length:
        output_features = output_features[:expected_length]
    elif len(output_features) < expected_length:
        pad_width = expected_length - len(output_features)
        output_features = np.pad(output_features, (0, pad_width), 'constant')

    write_wav(output_wav, rate, output_features)

    sess.close()

