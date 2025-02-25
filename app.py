import os
import torchaudio

TEST_AUDIO_DIR = "./data/test"


def load_audio(audio_directory: str):
    """
    Read audio files from audio directory (folder that contains audio files) and then return
    a dictionary with filename as key and value of a tuple of the file's waveform and sample rate
    """

    audio_data = {}

    if not os.path.isdir(audio_directory):
        raise ValueError(f"The provided path '{audio_directory}' is not a directory.")

    for filename in os.listdir(audio_directory):
        filepath = os.path.join(audio_directory, filename)

        if filename.lower().endswith((".wav", ".mp3")):
            try:
                waveform, sample_rate = torchaudio.load(filepath)
                audio_data[filename] = (waveform, sample_rate)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        else:
            print(f"[LOG]: {filename} is not readable")

    return audio_data
