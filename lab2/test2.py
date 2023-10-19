import cv2
import pyaudio
import numpy as np
import wave
import subprocess

# Параметри аудіо
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 2048
AUDIO_FILENAME = "output_audio2.wav"
VIDEO_FILENAME = "output_video2.avi"
OUTPUT_FILENAME = "output_combined2.avi"

# Ініціалізація OpenCV
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(VIDEO_FILENAME, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Ініціалізація PyAudio
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=0, frames_per_buffer=CHUNK)


audio_frames = []

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        out.write(frame)
        cv2.imshow('Recording...', frame)

        # Запис аудіо
        # data = stream.read(CHUNK)
        data = stream.read(stream.get_read_available(), exception_on_overflow=False)

        audio_frames.append(data)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    pass

print("Saving...")
# Закінчити відео запис
cap.release()
out.release()
cv2.destroyAllWindows()

# Закінчити аудіо запис
stream.stop_stream()
stream.close()
audio.terminate()

# Зберегти аудіо в файл
with wave.open(AUDIO_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(audio_frames))

# Об'єднати аудіо та відео за допомогою FFmpeg
cmd = f"ffmpeg -i {VIDEO_FILENAME} -i {AUDIO_FILENAME} -c:v copy -c:a aac -strict experimental {OUTPUT_FILENAME}"
subprocess.call(cmd, shell=True)

print("Done!")
