import audio_to_midi_melodia as atomm
import rnn_predict as rpred

def process_midi(file_name, length):
    duet = atomm.audio_to_midi_melodia(file_name, 'test.mid')
    output = rpred(duet, length)

    return output
