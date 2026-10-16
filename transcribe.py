#!/usr/bin/env python3
"""Speech to text for the Voice plugin, with faster-whisper.

Runs with the Python of the plugin's own venv, never the system's: main.py
starts it and reads its stdout, one JSON object per line.

  transcribe.py --download --model base --models-dir DIR
      fetch a model (the one-time setup does this), print {"path": "..."}
  transcribe.py --model base --models-dir DIR [--language it] ...
      load the model, read raw 16 kHz mono s16le audio on stdin until EOF,
      print {"loaded": true}, {"partial": "..."} after each segment, then
      {"text": "...", "language": "it"}

Failures are printed as {"error": "..."}. The audio is read in a thread
from the start, so the model loads while main.py is still recording and
the pipe never fills up.
"""
import argparse
import json
import os
import sys
import threading

# Our stdout is the protocol. Anything a library prints goes to stderr.
OUT = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)


def say(msg):
    OUT.write(json.dumps(msg) + "\n")
    OUT.flush()


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="base")
    p.add_argument("--models-dir", required=True)
    p.add_argument("--language", default="")
    p.add_argument("--device", default="cpu")
    p.add_argument("--compute-type", default="int8")
    p.add_argument("--download", action="store_true")
    return p.parse_args(argv)


def transcribe(args):
    audio = []
    reader = threading.Thread(target=lambda: audio.append(sys.stdin.buffer.read()), daemon=True)
    reader.start()
    from faster_whisper import WhisperModel
    import numpy as np

    # local_files_only: the setup downloaded the model; never go online here.
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type,
                         download_root=args.models_dir, local_files_only=True)
    say({"loaded": True})
    reader.join()
    data = audio[0] if audio else b""
    samples = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
    # The VAD filter drops the silence around the words, where Whisper
    # would otherwise make up a "Thank you."
    segments, info = model.transcribe(samples, language=args.language or None,
                                      vad_filter=True, beam_size=5)
    text = ""
    for segment in segments:
        text += segment.text
        say({"partial": text.strip()})
    say({"text": text.strip(), "language": info.language})


def main():
    args = parse_args(sys.argv[1:])
    try:
        if args.download:
            from faster_whisper import download_model
            say({"path": download_model(args.model, cache_dir=args.models_dir)})
        else:
            transcribe(args)
    except Exception as e:  # main.py shows it to the user
        print(f"transcribe: {e!r}", file=sys.stderr)
        say({"error": str(e) or type(e).__name__})
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
