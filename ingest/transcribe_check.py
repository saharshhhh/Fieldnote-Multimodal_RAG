from faster_whisper import WhisperModel

model = WhisperModel("small", compute_type="int8")
segments, info = model.transcribe("data/calls/call_01_ac_frozen_coil.mp3", vad_filter=True)
for s in segments:
    print(f"[{s.start:6.1f}s - {s.end:6.1f}s] {s.text.strip()}")