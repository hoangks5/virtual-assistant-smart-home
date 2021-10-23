import speech_recognition as sr
import pyaudio
r = sr.Recognizer()
with sr.Microphone() as source:
    print('You: ')
    audio = r.listen(source)
    try:
        text = r.recognize_google(audio,language="vi-VI")
        print('You --> : {}'.format(text))
    except:
        print("Hmm.....")