import tensorflow as tf
import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc, delta
import matlab.engine

# Function to read a .wav file and extract MFCC features
def extract_features(wav_path, num_features=25, max_length=1220):
    rate, sig = wavfile.read(wav_path)
    num_mfcc_features = (num_features - 1) // 3
    mfcc_feat = mfcc(sig, rate, numcep=num_mfcc_features)
    d_mfcc_feat = delta(mfcc_feat, 2)
    dd_mfcc_feat = delta(d_mfcc_feat, 2)
    features = np.concatenate((mfcc_feat, d_mfcc_feat, dd_mfcc_feat), axis=1)
    extra_feature = np.zeros((features.shape[0], 1))
    features = np.concatenate((features, extra_feature), axis=1)
    if features.shape[0] > max_length:
        features = features[:max_length, :]
    else:
        pad_width = max_length - features.shape[0]
        features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
    return features.reshape(1, max_length, num_features), rate

# Function to write a .wav file
def write_wav(wav_path, rate, data):
    data = np.asarray(data, dtype=np.float32)
    wavfile.write(wav_path, rate, data)

# Main function
def main(input_wav, output_wav, meta_path, data_path):
    # Start MATLAB engine
    eng = matlab.engine.start_matlab()
    eng.addpath('../model/invMFCCs_new')

    tf.compat.v1.disable_eager_execution()
    input_features, rate = extract_features(input_wav)
    
    with tf.compat.v1.Session() as sess:
        saver = tf.compat.v1.train.import_meta_graph(meta_path)
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        graph = tf.compat.v1.get_default_graph()
        input_tensor = graph.get_tensor_by_name("Placeholder:0")
        output_tensor = graph.get_tensor_by_name("Tanh_1:0")
        
        output_features = sess.run(output_tensor, feed_dict={input_tensor: input_features})
        
        print("Output features shape:", output_features.shape)
        
        # Process each batch of predicted MFCC features
        for i in range(output_features.shape[0]):
            predicted_mfccs_transposed = np.transpose(output_features[i,:,:])
            
            # Convert MFCC features to waveform using invmelfcc function in MATLAB
            inverted_wav_data = eng.invmelfcc(matlab.double(predicted_mfccs_transposed.tolist()), rate, 25)
            inverted_wav_data = np.squeeze(np.array(inverted_wav_data))

            # Normalize the waveform data to be between -1 and 1
            maxVec = np.max(inverted_wav_data)
            minVec = np.min(inverted_wav_data)
            inverted_wav_data = ((inverted_wav_data - minVec) / (maxVec - minVec) - 0.5) * 2
            
            # Write the waveform data to a .wav file
            wavfile.write(output_wav, int(rate), inverted_wav_data.astype(np.float32))

if __name__ == "__main__":
    input_wav = 'input.wav'
    output_wav = 'output.wav'
    meta_path = 'mfcc_model_epoch_3129.meta'
    data_path = 'mfcc_model_epoch_3129.data-00000-of-00001'
    main(input_wav, output_wav, meta_path, data_path)
