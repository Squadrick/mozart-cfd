import audio_to_midi_melodia as atomm
import rnn_predict as rpred
import pretty_midi
import numpy as np
import librosa

def process_midi(file_name, length):
    _, sr = atomm.audio_to_midi_melodia(file_name, 'test.mid')
    duet = pretty_midi.PrettyMIDI('test.mid')
    output, midi = rpred.generate_midi(duet, length) 
    audio_data = midi.synthesize()
    librosa.output.write_wav("final.wav", audio_data, sr) 
    midi.write('final.mid')

process_midi('guitar.wav', 16)
