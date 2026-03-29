import pyttsx3
import threading
import queue

engine = pyttsx3.init()
engine.setProperty('rate', 150)

voice_queue = queue.Queue()

def voice_worker():
    while True:
        text = voice_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        voice_queue.task_done()

threading.Thread(target=voice_worker, daemon=True).start()

def speak(text):
    if voice_queue.empty():  # prevents overlapping speech
        voice_queue.put(text)