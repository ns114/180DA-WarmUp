# Following tutorial from Real Python https://realpython.com/python-speech-recognition/#the-recognizer-class
import speech_recognition as sr
r = sr.Recognizer()
audio_file = sr.AudioFile('harvard.wav')
with audio_file as source:
    audio = r.record(source)
r.recognize_google(audio)