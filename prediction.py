import tensorflow as tf
import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc, delta


def extract_features(wav_path, num_features=25, max_length=1220):
    rate, sig = wavfile.read(wav_path)
    mfcc_feat = mfcc(sig, rate, numcep=num_features)
    d_mfcc_feat = delta(mfcc_feat, 2)
    dd_mfcc_feat = delta(d_mfcc_feat, 2)
    features = np.concatenate((mfcc_feat, d_mfcc_feat, dd_mfcc_feat), axis=1)
    
    # Ensure the features have the required shape (max_length, num_features)
    if features.shape[0] > max_length:
        features = features[:max_length, :]  # Truncate
    else:
        pad_width = max_length - features.shape[0]
        features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
    
    return features.reshape(1, max_length, num_features), rate


def write_wav(wav_path, rate, data):
    wavfile.write(wav_path, rate, data)




input_wav = 'source.wav'
output_wav = 'output.wav'
meta_path = 'mfcc_model_epoch_3129.meta'
data_path = 'mfcc_model_epoch_3129.data-00000-of-00001'

# Start a session
with tf.compat.v1.Session() as sess:

    input_features, rate = extract_features(input_wav)

    # Import the meta graph
    saver = tf.compat.v1.train.import_meta_graph('mfcc_model_epoch_3129.meta')
    # Restore the values of the variables
    saver.restore(sess, tf.train.latest_checkpoint('./'))

    # Get the graph
    graph = tf.compat.v1.get_default_graph()
    # for op in graph.get_operations():
    #     print(op.name)
    input_tensor = graph.get_tensor_by_name("Placeholder_1:0")
    output_tensor = graph.get_tensor_by_name("Tanh_1:0")
    output_features = sess.run(output_tensor, feed_dict={input_tensor: input_features})
    write_wav(output_wav, rate, output_features.astype(np.int16))





    sess.close()








# # Function to read a .wav file and extract MFCC features
# def extract_features(wav_path):
#     rate, sig = wavfile.read(wav_path)
#     mfcc_feat = mfcc(sig, rate)
#     d_mfcc_feat = delta(mfcc_feat, 2)
#     dd_mfcc_feat = delta(d_mfcc_feat, 2)
#     features = np.concatenate((mfcc_feat, d_mfcc_feat, dd_mfcc_feat), axis=1)
#     return features, rate

# # Function to write a .wav file
# def write_wav(wav_path, rate, data):
#     wavfile.write(wav_path, rate, data)

# # Load the model from checkpoint
# def load_model_from_checkpoint(checkpoint_path):
#     tf.compat.v1.disable_eager_execution()
#     sess = tf.compat.v1.Session()
#     saver = tf.compat.v1.train.import_meta_graph(checkpoint_path + '.meta')
#     saver.restore(sess, checkpoint_path)
#     graph = tf.compat.v1.get_default_graph()
#     return sess, graph

# # Process the input and get the output
# def process_input(sess, graph, input_features):
#     input_tensor = graph.get_tensor_by_name("input:0")
#     output_tensor = graph.get_tensor_by_name("output:0")
#     output_features = sess.run(output_tensor, feed_dict={input_tensor: input_features})
#     return output_features

# # Main function
# def main(input_wav, output_wav, checkpoint_path):
#     input_features, rate = extract_features(input_wav)
#     sess, graph = load_model_from_checkpoint(checkpoint_path)
#     output_features = process_input(sess, graph, input_features)
#     # Assuming the output is in the same shape as input for simplicity
#     write_wav(output_wav, rate, output_features.astype(np.int16))
#     sess.close()

# if __name__ == "__main__":
#     input_wav = 'input.wav'
#     output_wav = 'output.wav'
#     checkpoint_path = 'checkpoint'  # Adjust to the base name of your checkpoint files
#     main(input_wav, output_wav, checkpoint_path)

