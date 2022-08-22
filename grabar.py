import pyaudio
import wave 
import os

def grabar(ruta):
    """
    ejemplo:
    grabar(ruta = "C:/Users/andre/OneDrive/Documentos/Ingenieria de sistemas/Sexto semestre/Procesos Estocasticos/Proyecto/grabacion.wav")
    """
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    duracion = 3
    archivo = ruta

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("Se inicio la grabacion")
    frames = []

    for i in range(0, int(RATE/CHUNK*duracion)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Grabacion finalizada")


    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(archivo, "wb")
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

ruta = "ejemplo: C:/Users/andre/OneDrive/Documentos/Ingenieria de sistemas/Sexto semestre/Procesos Estocasticos/Proyecto/grabacion.wav"
grabar(ruta)