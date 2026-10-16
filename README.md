# Voice: dictation in the Parsec launcher

A [Parsec](https://github.com/abidibo/parsec) plugin. Say something and
Parsec types it into the window you were in. The speech is turned into text
on your computer by [faster-whisper](https://github.com/SYSTRAN/faster-whisper),
a fast build of OpenAI's Whisper: no account, no cloud, offline after the
first use.

| Type | What happens |
|---|---|
| `say` | Enter starts listening. Speak, then pause: it stops by itself after a second and a half of silence. |
| then Enter | Pastes what you said into the window you came from. |
| then Tab | Search: puts the text in Parsec's search box, as if you had typed it. Copy: copies it. |

The usual flow: put the cursor where the text should go, open Parsec, type
`say`, Enter, speak, pause, Enter. While it listens the row shows the
seconds and a level meter, so you can see the microphone hears you. It
stops after 30 seconds anyway, or after 8 if you say nothing.

## Requirements

- Parsec 1.4.0 or later (slow answers for plugins); the Search action needs
  Parsec 1.8.0, older versions just don't show it
- PipeWire (`pw-record`, standard on GNOME 46) or PulseAudio (`parecord`)
- Python 3.8+ with `venv`. Debian and Ubuntu ship it separately:
  `sudo apt install python3-venv` (the plugin tells you if it's missing)
- The GNOME Shell extension that comes with Parsec, to paste; without it
  the text is copied instead

## Install

```sh
parsec plugin install https://github.com/abidibo/parsec-voice
```

## First use

The first `say` sets up the speech engine, once:

- a Python venv with faster-whisper, about 130 MB to download, 460 MB on disk
- the model you chose, 145 MB for `base`

Everything goes in `~/.local/share/parsec/plugin-data/voice/`, nothing is
installed system-wide, and plugin updates leave it alone. On a good
connection it takes under a minute. Parsec shows the progress; if you close
it, the setup carries on in the background and resumes if it is
interrupted. `say` again when it's done.

Choosing another model later downloads just that model, the next time you
use `say`.

To remove everything, `parsec plugin remove voice`, then delete
`~/.local/share/parsec/plugin-data/voice`.

## Configuration

Click the edit button next to Voice in Parsec's Settings › Plugins, or
create `~/.config/parsec/plugins/voice.toml` yourself. Your values survive
plugin updates, and Parsec restarts the plugin when you save.

```toml
model = "base"          # tiny 75 MB, base 145 MB, small 485 MB, medium 1.5 GB
language = ""           # "" detects it; or "it", "en", ...
max_seconds = 30        # stop listening after this long, at most 60
silence_seconds = 1.5   # stop after a pause this long, once you've spoken
device = "cpu"          # "cuda" for an NVIDIA GPU with CUDA 12 and cuDNN
compute_type = "int8"   # int8 is fastest on a CPU; "float16" on a GPU
```

`base` is good for English. For Italian and most other languages, `small`
is much better and still quick on a recent CPU; `medium` is better again
but takes several seconds per sentence. Setting `language` saves the
detection step and stops a short phrase being taken for another language.

## Privacy

After the one-time download nothing goes over the network: the model runs
with Hugging Face's offline mode on and is never looked up again. Audio is
held in memory while you speak and handed straight to the model; it is
never written to disk, and the microphone closes as soon as you pause or
leave the list. The downloads (the Python packages from PyPI, the model
from Hugging Face) send no telemetry.

## Tip: a keyboard shortcut

Parsec doesn't yet open with a query from the command line, so bind your
usual Parsec shortcut and type `say`. Three letters and Enter.

## How it works

`main.py` speaks Parsec's plugin protocol and needs only Python's standard
library. Listening and transcribing happen behind a slow browse: it reads
raw 16 kHz audio from `pw-record`, measures the loudness of each 30 ms
against the noise floor of the room (taken from the first 0.3 s, then the
quietest moment of the last few seconds), and stops on a pause. Meanwhile
`transcribe.py` runs in the venv and loads the model, so it is ready when
you stop talking; the audio goes to it on stdin and the text comes back as
JSON. The silence around the words is cut by Whisper's own voice detector,
which keeps it from making up words where there are none.

The setup runs as a separate, detached process (`main.py setup`), because
Parsec gives a plugin two minutes at most and stops it when you leave the
list. It writes its progress to `setup.json` in the data folder, which the
plugin shows, and logs to `setup.log` next to it.

## Development

```sh
python3 -m unittest      # offline: synthetic audio, a fake recorder and engine
```

## License

MIT
