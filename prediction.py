import tensorflow as tf
import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc, delta

def extract_features(wav_path, num_features=25, max_length=1220):
    # Ensure that the total number of features after concatenation is 25
    assert num_features % 3 == 1, "num_features must be set so that num_features % 3 == 1"
    
    rate, sig = wavfile.read(wav_path)
    num_mfcc_features = (num_features - 1) // 3
    
    # Extract MFCC features
    mfcc_feat = mfcc(sig, rate, numcep=num_mfcc_features)
    d_mfcc_feat = delta(mfcc_feat, 2)
    dd_mfcc_feat = delta(d_mfcc_feat, 2)
    
    # Ensure that the concatenated features produce the correct number of features
    features = np.concatenate((mfcc_feat, d_mfcc_feat, dd_mfcc_feat), axis=1)
    
    # Add an extra dimension to match the expected 25 features
    extra_feature = np.zeros((features.shape[0], 1))
    features = np.concatenate((features, extra_feature), axis=1)
    
    # Ensure the features have the required shape (max_length, num_features)
    print("Original feature shape:", features.shape)  # Debugging line
    if features.shape[0] > max_length:
        features = features[:max_length, :]  # Truncate
    else:
        pad_width = max_length - features.shape[0]
        features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
    
    print("Padded/truncated feature shape:", features.shape)  # Debugging line
    return features.reshape(1, max_length, num_features), rate

def write_wav(wav_path, rate, data):
    # Ensure the data is in the correct format
    data = np.asarray(data, dtype=np.int16)
    
    # Reshape the data to be 1D if necessary
    if data.ndim > 1:
        data = data.flatten()
    
    # Write the WAV file
    wavfile.write(wav_path, rate, data)

input_wav = 'input.wav'
output_wav = 'output.wav'
meta_path = 'mfcc_model_epoch_3129.meta'
data_path = 'mfcc_model_epoch_3129.data-00000-of-00001'

# Start a session
with tf.compat.v1.Session() as sess:
    input_features, rate = extract_features(input_wav)

    # Import the meta graph
    saver = tf.compat.v1.train.import_meta_graph(meta_path)
    # Restore the values of the variables
    saver.restore(sess, tf.train.latest_checkpoint('./'))

    # Get the graph
    graph = tf.compat.v1.get_default_graph()
    # for op in graph.get_operations():
    #     print(op.name)
    input_tensor = graph.get_tensor_by_name("Placeholder:0")
    output_tensor = graph.get_tensor_by_name("Tanh_1:0")
    output_features = sess.run(output_tensor, feed_dict={input_tensor: input_features})

    if output_features.ndim > 1:
        output_features = output_features.flatten()

    write_wav(output_wav, rate, output_features.astype(np.int16))

    sess.close()
