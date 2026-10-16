#!/usr/bin/env python3
"""Parsec plugin: dictate with your voice, transcribed on this computer.

  say       Enter listens until you pause. Enter pastes what you said
            into the window you came from; Tab searches Parsec for it,
            or copies it.

Speech becomes text with faster-whisper, a Whisper model that runs on the
CPU. It lives in a venv of its own under
~/.local/share/parsec/plugin-data/voice, not in the plugin folder, which
an update replaces. The first use downloads it; after that nothing leaves
the computer.

Parsec sends a query on every keystroke and gives it three seconds, so
queries only look at a couple of files. The work is behind a slow browse:
record from the default microphone until a pause, then hand the audio to
transcribe.py in the venv, which loaded the model while you spoke.

A slow browse gets two minutes at most, and Parsec kills this process
when the user leaves the level, so the one-time setup runs as a detached
process of its own (`main.py setup`). It writes its progress to a status
file that the browse follows, holds a lock while it runs, and picks up
where it stopped if it is interrupted.
"""
import array
import collections
import ctypes
import fcntl
import importlib.util
import json
import math
import os
import select
import shutil
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TRANSCRIBE = os.path.join(HERE, "transcribe.py")
PY_VERSION = "%d.%d" % sys.version_info[:2]
ENGINE_PACKAGE = "faster-whisper>=1.0,<2"
ICON = "audio-input-microphone-symbolic"

# What the recorder delivers: raw signed 16-bit little-endian mono, 16 kHz,
# the rate Whisper works at.
RATE = 16000
# Loudness is measured over 30 ms frames.
FRAME = 480
# The first moments are taken as the room's noise floor; after that the
# floor is the quietest frame of the last few seconds, which follows a fan
# that starts and is found again in the gaps between words.
CALIBRATE = 0.3
FLOOR_WINDOW = 3.0
# Speech is louder than the floor by this factor (about 10 dB), and never
# quieter than MIN_LEVEL, which keeps a muted, all-zero input from
# counting every click as speech.
FACTOR = 3.0
MIN_LEVEL = 10.0
# Voiced frames in a row before it counts as speech, not a click.
SPEECH_MIN = 0.12
# Without speech, give up after this long.
NO_SPEECH = 8.0
# A recorder that sends nothing at all for this long has no microphone.
NO_AUDIO = 2.0
MAX_SECONDS = 60
# Seconds between progress lines while listening.
PROGRESS_EVERY = 0.15
# Parsec allows a slow browse 120 s; stop a little before.
BUDGET = 112
# How long a browse follows the setup before handing it to the background.
SETUP_BUDGET = 100
POLL = 0.25

# Approximate downloads, in MB, for the messages.
ENGINE_MB = 130
MODEL_MB = {"tiny": 75, "base": 145, "small": 485, "medium": 1530,
            "large-v3": 3100, "turbo": 1620}


class VoiceError(Exception):
    pass


class SetupError(Exception):
    pass


def data_dir():
    """Where the venv and models live. Parsec doesn't define a data folder
    for plugins, so this sits next to its own, outside the plugin folder."""
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, "parsec", "plugin-data", "voice")


def short(text, limit=80):
    line = " ".join(text.split())
    return line if len(line) <= limit else line[: limit - 1] + "…"


def number(value, default):
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(value, low, high):
    return max(low, min(high, value))


def read_json(path):
    try:
        with open(path) as f:
            value = json.load(f)
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path, value):
    """Replace the file in one step, so a reader never sees half of it."""
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(value, f)
    os.replace(tmp, path)


def parse_status(text):
    """The setup's status file, with defaults for anything missing or odd."""
    try:
        raw = json.loads(text or "")
    except ValueError:
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    state = raw.get("state")
    return {
        "state": state if state in ("running", "done", "error") else "idle",
        "stage": str(raw.get("stage") or ""),
        "message": str(raw.get("message") or ""),
        "model": str(raw.get("model") or ""),
        "started": number(raw.get("started"), 0.0),
        "updated": number(raw.get("updated"), 0.0),
    }


def mb(size):
    return f"{size / 1e6:.0f} MB"


def download_size(model, engine=True):
    """The first-run download in words: engine and model."""
    size = MODEL_MB.get(model)
    model_part = f"the {model} model ({size} MB)" if size else f"the {model} model"
    return f"the speech engine (about {ENGINE_MB} MB) and {model_part}" if engine else model_part


def venv_missing():
    """Debian and Ubuntu ship venv and ensurepip separately, as python3-venv."""
    return any(importlib.util.find_spec(m) is None for m in ("venv", "ensurepip"))


def recorder_argv():
    """Raw 16 kHz mono audio from the default microphone on stdout."""
    if shutil.which("pw-record"):
        return ["pw-record", "--rate", str(RATE), "--channels", "1", "--format", "s16",
                "--raw", "-P", '{ application.name = "Parsec Voice" }', "-"]
    if shutil.which("parecord"):
        return ["parecord", "--raw", "--format=s16le", f"--rate={RATE}", "--channels=1",
                "--latency-msec=100", "--client-name=Parsec Voice"]
    return None


def die_with_parent():
    """Run in the child before exec: SIGTERM it when this process dies, so
    the microphone closes when Parsec kills the plugin."""
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except OSError:
        pass


def stop(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(1)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    for pipe in (proc.stdin, proc.stdout, proc.stderr):
        if pipe:
            try:
                pipe.close()
            except OSError:
                pass


def tail(path, lines=3):
    try:
        with open(path, errors="replace") as f:
            text = f.read()[-4000:]
    except OSError:
        return ""
    return " ".join(l.strip() for l in text.strip().splitlines()[-lines:])


# ------------------------------------------------------------ listening

def loudness(samples, prev=0):
    """RMS of the difference between neighbouring samples: a crude
    high-pass. Microphones often open with a large DC offset that takes a
    second to fade, and plain RMS would hear that as speech."""
    total = 0
    for s in samples:
        d = s - prev
        total += d * d
        prev = s
    return math.sqrt(total / len(samples)) if samples else 0.0


class Gate:
    """Decides when to stop listening, from the loudness of 30 ms frames:
    after a pause once you've spoken, at the time limit, or when nobody
    speaks at all."""

    def __init__(self, silence=1.5, max_seconds=30, no_speech=NO_SPEECH):
        self.silence = silence
        self.max_seconds = max_seconds
        self.no_speech = no_speech
        self.frames = 0
        self.recent = collections.deque(maxlen=int(FLOOR_WINDOW * RATE / FRAME))
        self.floor = None
        self.level = 0.0
        self.speech = False
        self.voiced_run = 0
        self.quiet = 0
        self.prev = 0
        self.pending = b""

    @property
    def seconds(self):
        return self.frames * FRAME / RATE

    def threshold(self):
        return max(MIN_LEVEL, (self.floor or 0.0) * FACTOR)

    def feed(self, pcm):
        """Take raw audio; return why to stop ("silence", "max",
        "nothing"), or None to keep listening."""
        data = self.pending + pcm
        usable = len(data) - len(data) % (FRAME * 2)
        self.pending = data[usable:]
        for offset in range(0, usable, FRAME * 2):
            reason = self.frame(data[offset:offset + FRAME * 2])
            if reason:
                return reason
        return None

    def frame(self, chunk):
        samples = array.array("h")
        samples.frombytes(chunk)
        if sys.byteorder == "big":
            samples.byteswap()
        self.level = loudness(samples, self.prev)
        self.prev = samples[-1]
        self.frames += 1
        self.recent.append(self.level)
        if self.seconds < CALIBRATE:
            return None
        self.floor = min(self.recent)
        voiced = self.level > self.threshold()
        if self.speech:
            self.quiet = 0 if voiced else self.quiet + 1
        elif voiced:
            self.voiced_run += 1
            if self.voiced_run * FRAME / RATE >= SPEECH_MIN:
                self.speech = True
        else:
            self.voiced_run = 0
        if self.seconds >= self.max_seconds:
            return "max"
        if self.speech and self.quiet * FRAME / RATE >= self.silence:
            return "silence"
        if not self.speech and self.seconds >= self.no_speech:
            return "nothing"
        return None


def meter(level, floor, width=8):
    """A level meter in block characters, 30 dB above the noise floor."""
    ref = max(floor or 0.0, MIN_LEVEL / FACTOR, 1.0)
    db = 20 * math.log10(max(level, 1.0) / ref)
    filled = int(round(clamp(db / 30 * width, 0, width)))
    return "█" * filled + "░" * (width - filled)


# ------------------------------------------------------------ the engine

class Engine:
    """The plugin's own files: a venv with faster-whisper, the models, and
    the setup's status, lock and log."""

    def __init__(self, root):
        self.root = root
        self.venv = os.path.join(root, "venv")
        self.python = os.path.join(self.venv, "bin", "python")
        self.models = os.path.join(root, "models")
        self.cache = os.path.join(root, "cache")
        self.ready_path = os.path.join(root, "ready.json")
        self.status_path = os.path.join(root, "setup.json")
        self.lock_path = os.path.join(root, "setup.lock")
        self.log_path = os.path.join(root, "setup.log")
        self.transcribe_log = os.path.join(root, "transcribe.log")

    def marks(self):
        return read_json(self.ready_path)

    def engine_ready(self):
        marks = self.marks()
        return (os.path.exists(self.python) and marks.get("engine") is True
                and marks.get("python") == PY_VERSION)

    def ready(self, model):
        return self.engine_ready() and model in (self.marks().get("models") or [])

    def mark(self, engine=None, model=None):
        marks = self.marks()
        if engine is not None:
            marks["engine"] = engine
            marks["python"] = PY_VERSION
        if model is not None:
            marks["models"] = sorted(set(marks.get("models") or []) | {model})
        write_json(self.ready_path, marks)

    def status(self):
        try:
            with open(self.status_path) as f:
                return parse_status(f.read())
        except OSError:
            return parse_status("")

    def write_status(self, state, stage, message, model, started=None):
        write_json(self.status_path, {
            "state": state, "stage": stage, "message": message, "model": model,
            "pid": os.getpid(), "started": started or time.time(), "updated": time.time()})

    def setup_running(self):
        """True while a setup process holds the lock, wherever it came from."""
        try:
            with open(self.lock_path, "a") as f:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        except OSError:
            return False
        return False

    def env(self, offline):
        """Keep pip's and Hugging Face's caches in our folder, no telemetry,
        and once set up, no network at all."""
        env = dict(os.environ)
        for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
            env.pop(name, None)
        env.update({
            "PYTHONUNBUFFERED": "1",
            "PIP_CACHE_DIR": os.path.join(self.cache, "pip"),
            "PIP_USER": "0",
            "PIP_NO_INPUT": "1",
            "HF_HOME": os.path.join(self.cache, "hf"),
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        })
        if offline:
            env["HF_HUB_OFFLINE"] = "1"
        return env


def dir_size(path):
    total = 0
    for base, _, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(base, name)).st_size
            except OSError:
                pass
    return total


def pip_message(line):
    """pip's output in a few words, or None for lines not worth showing."""
    line = line.strip()
    if line.startswith("Collecting "):
        return f"Looking up {line.split()[1]}"
    if line.startswith("Downloading ") and ".metadata" not in line:
        parts = line.split()
        name = os.path.basename(parts[1]).split("-")[0]
        size = line[line.rfind("(") + 1:line.rfind(")")] if line.endswith(")") else ""
        return f"Downloading {name} ({size})" if size else f"Downloading {name}"
    if line.startswith("Installing collected packages"):
        return "Installing the speech engine"
    return None


def make_venv(engine):
    """A venv whose pip works. One left half made by an interrupted setup,
    or made with another Python version, is made again."""
    marks = engine.marks()
    stale = marks.get("python") not in (None, PY_VERSION)
    if os.path.exists(engine.python) and not stale:
        ok = subprocess.run([engine.python, "-m", "pip", "--version"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        if ok:
            return
    proc = subprocess.run([sys.executable, "-m", "venv", "--clear", engine.venv],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    sys.stderr.write(proc.stdout)
    if proc.returncode != 0:
        if "ensurepip" in proc.stdout:
            raise SetupError("ensurepip is missing: install python3-venv")
        raise SetupError(short(proc.stdout.strip() or "python3 -m venv failed", 300))


def pip_install(engine, report):
    argv = [engine.python, "-m", "pip", "install", "--disable-pip-version-check",
            "--progress-bar", "off", ENGINE_PACKAGE]
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=engine.env(offline=False))
    errors = []
    for line in proc.stdout:
        sys.stderr.write(line)
        if line.startswith("ERROR"):
            errors.append(line.strip())
        message = pip_message(line)
        if message:
            report("pip", message)
    if proc.wait() != 0:
        raise SetupError(" ".join(errors[-2:]) or f"pip exited with {proc.returncode}")


def fetch_model(engine, model, report):
    """Download through faster-whisper itself, so the files land where it
    looks for them. Hugging Face resumes a partial download."""
    argv = [engine.python, TRANSCRIBE, "--download", "--model", model,
            "--models-dir", engine.models]
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, env=engine.env(offline=False))
    expected = MODEL_MB.get(model)
    of = f" of about {expected} MB" if expected else ""
    folders = (engine.models, os.path.join(engine.cache, "hf"))
    before = sum(dir_size(f) for f in folders)  # other models, an earlier attempt
    while proc.poll() is None:
        got = max(0, sum(dir_size(f) for f in folders) - before)
        report("model", f"Downloading the {model} model: {mb(got)}{of}")
        time.sleep(1)
    reply = {}
    for line in proc.stdout.read().decode(errors="replace").splitlines():
        try:
            reply = json.loads(line)
        except ValueError:
            continue
    if proc.returncode != 0 or "path" not in reply:
        raise SetupError(str(reply.get("error") or f"Model download exited with {proc.returncode}"))


def run_setup(root, model):
    """The one-time setup, in its own process. It holds the lock while it
    runs; every step is skipped when already done, so running it again
    resumes it."""
    engine = Engine(root)
    os.makedirs(root, exist_ok=True)
    with open(engine.lock_path, "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0  # another setup is already on it
        return set_up_engine(engine, model)


def set_up_engine(engine, model):
    started = time.time()

    def report(stage, message):
        engine.write_status("running", stage, message, model, started)

    try:
        if not engine.engine_ready():
            report("venv", "Creating a Python environment")
            make_venv(engine)
            report("pip", "Installing faster-whisper")
            pip_install(engine, report)
            engine.mark(engine=True)
        if not engine.ready(model):
            report("model", f"Downloading the {model} model")
            fetch_model(engine, model, report)
            engine.mark(model=model)
        # The model is in models/; downloaded wheels and Hugging Face's
        # chunk cache would only take room.
        shutil.rmtree(engine.cache, ignore_errors=True)
        engine.write_status("done", "done", "Ready", model, started)
        return 0
    except Exception as e:  # anything goes to the status file, for the browse to show
        print(f"voice: setup failed: {e!r}", file=sys.stderr)
        message = str(e) if isinstance(e, SetupError) else f"{type(e).__name__}: {e}"
        engine.write_status("error", "error", message, model, started)
        return 1


# ------------------------------------------------------------ the plugin

def listen_action(label="Voice", waiting="Listening…"):
    return {"label": label, "browse": {"mode": "listen"}, "slow": True, "waiting": waiting}


def again_item():
    return {"id": "again", "title": "Listen again", "icon": ICON,
            "actions": [listen_action()]}


def result_items(text, language=None):
    """What you said: Enter pastes it, then Search and Copy."""
    actions = [
        {"label": "Paste", "paste": text},
        # Puts the text in Parsec's search box, as if typed.
        {"label": "Search", "query": text},
        {"label": "Copy", "copy": text},
    ]
    item = {"id": "text", "title": short(text), "icon": ICON, "actions": actions,
            "subtitle": "Enter pastes · Tab: search, copy" + (f" · {language}" if language else "")}
    if short(text) != text:
        item["text"] = text
    return [item, again_item()]


def problem(title, text, icon="dialog-warning-symbolic", actions=None):
    return {"id": "problem", "title": title, "text": text, "icon": icon,
            "actions": actions or []}


class Transcriber:
    """transcribe.py in the venv. Started when listening starts, so the
    model loads while you speak; the audio goes in when you stop."""

    def __init__(self, engine, model, language, device, compute_type):
        argv = [engine.python, TRANSCRIBE, "--model", model, "--models-dir", engine.models,
                "--language", language, "--device", device, "--compute-type", compute_type]
        self.log_path = engine.transcribe_log
        log = open(self.log_path, "wb")
        with log:
            self.proc = subprocess.Popen(
                argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                env=engine.env(offline=True), preexec_fn=die_with_parent)

    def cancel(self):
        stop(self.proc)

    def finish(self, pcm, deadline, on_partial):
        """Send the audio, follow the partial text, return (text, language)."""
        try:
            self.proc.stdin.write(pcm)
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        fd = self.proc.stdout.fileno()
        buf = b""
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                self.cancel()
                raise VoiceError("Transcription took too long. A smaller model is faster.")
            ready, _, _ = select.select([fd], [], [], min(left, 0.5))
            if not ready:
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(msg, dict):
                    continue
                if "error" in msg:
                    self.cancel()
                    raise VoiceError(str(msg["error"]))
                if "partial" in msg:
                    on_partial(str(msg["partial"]))
                if "text" in msg:
                    self.cancel()
                    return str(msg["text"]).strip(), msg.get("language")
        stop(self.proc)
        raise VoiceError(tail(self.log_path) or f"The transcriber exited with {self.proc.returncode}")


class Voice:
    def __init__(self, out=sys.stdout, root=None):
        self.out = out
        self.engine = Engine(root or data_dir())
        self.setup_proc = None
        self.init({})

    def init(self, config):
        config = config or {}
        self.model = str(config.get("model") or "base").strip()
        self.language = str(config.get("language") or "").strip().lower()
        self.max_seconds = clamp(number(config.get("max_seconds"), 30), 3, MAX_SECONDS)
        self.silence = clamp(number(config.get("silence_seconds"), 1.5), 0.3, 5)
        self.device = str(config.get("device") or "cpu").strip()
        self.compute_type = str(config.get("compute_type") or "int8").strip()

    # ------------------------------------------------------------ query

    def query(self, text):
        if not recorder_argv():
            subtitle = "Needs pw-record (PipeWire) or parecord (PulseAudio)"
        elif self.engine.ready(self.model):
            subtitle = "Speak, then pause · Enter pastes what you said"
        else:
            subtitle = f"The first time downloads {self.to_download()}"
        return [{"id": "listen", "title": "Start listening", "subtitle": subtitle, "icon": ICON,
                 "actions": [listen_action()]}]

    # ------------------------------------------------------------ browse

    def browse(self, data):
        if not isinstance(data, dict) or data.get("mode") != "listen":
            return []
        started = time.monotonic()
        if not self.engine.ready(self.model):
            return self.set_up(started)
        return self.listen(started)

    def set_up(self, started):
        if venv_missing():
            return [self.venv_item()]
        launches = 0
        while time.monotonic() - started < SETUP_BUDGET:
            if self.engine.ready(self.model):
                return [{"id": "ready", "title": "Voice typing is ready",
                         "subtitle": "Enter, then speak", "icon": ICON,
                         "actions": [listen_action("Start listening")]}]
            status = self.engine.status()
            if not self.setup_active():
                if launches and status["state"] == "error":
                    return [self.setup_failed(status["message"])]
                if launches >= 2:
                    return [self.setup_failed(status["message"] or "Setup stopped")]
                self.start_setup()
                launches += 1
                status = self.engine.status()
            self.send({"type": "progress", "items": [self.setup_item(status, started)]})
            time.sleep(POLL)
        status = self.engine.status()
        return [{"id": "setup", "title": "Still setting up voice typing",
                 "subtitle": status["message"], "icon": ICON,
                 "text": "It carries on in the background. Try again in a minute.",
                 "actions": [listen_action("Voice", "Checking…")]}]

    def setup_active(self):
        if self.setup_proc is not None and self.setup_proc.poll() is None:
            return True
        return self.engine.setup_running()

    def start_setup(self):
        """Detached, in a session of its own: it outlives this process, which
        Parsec kills when the user leaves the level."""
        os.makedirs(self.engine.root, exist_ok=True)
        self.engine.write_status("running", "start", "Starting", self.model)
        with open(self.engine.log_path, "wb") as log:
            self.setup_proc = subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), "setup", self.engine.root, self.model],
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, cwd=self.engine.root,
                start_new_session=True, close_fds=True)

    def to_download(self):
        return download_size(self.model, not self.engine.engine_ready())

    def setup_item(self, status, started):
        took = int(time.monotonic() - started)
        return {"id": "setup", "title": f"Setting up voice typing… {took // 60}:{took % 60:02d}",
                "subtitle": status["message"] or "Starting", "icon": ICON,
                "text": f"A one-time download: {self.to_download()}. After that it "
                        "works offline. It carries on if you close Parsec.",
                "actions": []}

    def setup_failed(self, message):
        if "ensurepip" in message:
            return self.venv_item()
        return problem("Voice typing couldn't be set up", f"{message}\n\nLog: {self.engine.log_path}",
                       actions=[listen_action("Try again", "Setting up…")])

    def venv_item(self):
        command = "sudo apt install python3-venv"
        return problem("Install python3-venv first",
                       f"Voice typing keeps its speech engine in a Python venv of its own. "
                       f"Run:\n{command}", actions=[{"label": "Copy", "copy": command}])

    def listen(self, started):
        argv = recorder_argv()
        if not argv:
            return [problem("No audio recorder", "Voice typing records with pw-record (PipeWire) "
                            "or parecord (PulseAudio); neither is installed.")]
        try:
            transcriber = Transcriber(self.engine, self.model, self.language,
                                      self.device, self.compute_type)
        except OSError as e:
            return [problem("Can't start the speech engine", str(e))]
        try:
            reason, pcm = self.record(argv)
        except VoiceError as e:
            transcriber.cancel()
            return [problem("Can't record", str(e)), again_item()]
        if reason == "no audio":
            transcriber.cancel()
            return [problem("No sound from the microphone",
                            "Check the input device in GNOME Settings › Sound."), again_item()]
        if reason == "nothing":
            transcriber.cancel()
            return [problem("Didn't hear anything", "Speak after pressing Enter, then pause.",
                            icon=ICON), again_item()]
        self.send({"type": "progress", "items": [self.transcribing_item("")]})
        last = [0.0]

        def on_partial(text):
            now = time.monotonic()
            if now - last[0] >= PROGRESS_EVERY:
                last[0] = now
                self.send({"type": "progress", "items": [self.transcribing_item(text)]})

        try:
            text, language = transcriber.finish(pcm, started + BUDGET, on_partial)
        except VoiceError as e:
            return [problem("Couldn't transcribe", str(e)), again_item()]
        if not text:
            return [problem("Didn't catch that", "Try again a little closer to the microphone.",
                            icon=ICON), again_item()]
        return result_items(text, None if self.language else language)

    def transcribing_item(self, text):
        item = {"id": "text", "title": "Transcribing…", "icon": ICON, "actions": []}
        if text:
            item["text"] = text
        return item

    def record(self, argv):
        """Listen until the gate says stop; return (reason, audio)."""
        try:
            proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, preexec_fn=die_with_parent)
        except OSError as e:
            raise VoiceError(f"Can't start {argv[0]}: {e.strerror}")
        gate = Gate(self.silence, self.max_seconds)
        pcm = bytearray()
        fd, err_fd = proc.stdout.fileno(), proc.stderr.fileno()
        err = b""
        begun = time.monotonic()
        last = 0.0
        reason = None
        try:
            while reason is None:
                ready, _, _ = select.select([fd, err_fd], [], [], 0.05)
                if err_fd in ready:
                    err = (err + os.read(err_fd, 4096))[-2000:]
                if fd in ready:
                    chunk = os.read(fd, 16384)
                    if not chunk:
                        break
                    pcm += chunk
                    reason = gate.feed(chunk)
                elif not pcm and time.monotonic() - begun > NO_AUDIO:
                    reason = "no audio"
                now = time.monotonic()
                if now - last >= PROGRESS_EVERY:
                    last = now
                    self.send({"type": "progress", "items": [self.listening_item(gate)]})
            if reason is None:
                proc.wait(1)
                err += proc.stderr.read()
        except subprocess.TimeoutExpired:
            pass
        finally:
            stop(proc)
        if reason is None:
            raise VoiceError(short(err.decode(errors="replace"), 300) or f"{argv[0]} stopped")
        return reason, bytes(pcm)

    def listening_item(self, gate):
        if gate.speech:
            subtitle = "Pause to finish"
        else:
            subtitle = "Speak now"
        return {"id": "listening", "icon": ICON, "actions": [],
                "title": f"Listening… {int(gate.seconds)} s  {meter(gate.level, gate.floor)}",
                "subtitle": f"{subtitle} · stops at {int(self.max_seconds)} s"}

    def send(self, msg):
        self.out.write(json.dumps(msg) + "\n")
        self.out.flush()


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "setup":
        sys.exit(run_setup(sys.argv[2], sys.argv[3]))
    plugin = Voice()
    for line in sys.stdin:
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(msg, dict):
            continue
        kind = msg.get("type")
        try:
            if kind == "init":
                plugin.init(msg.get("config"))
                reply = {"type": "ready"}
            elif kind == "query":
                reply = {"type": "results", "id": msg.get("id"),
                         "items": plugin.query(msg.get("text", ""))}
            elif kind == "browse":
                reply = {"type": "results", "items": plugin.browse(msg.get("data"))}
            else:
                reply = {"type": "ok"}
        except Exception as e:  # never leave Parsec without its one reply line
            print(f"voice: {kind} failed: {e!r}", file=sys.stderr)
            if kind == "init":
                reply = {"type": "ready"}
            elif kind == "query":
                reply = {"type": "results", "id": msg.get("id"), "items": []}
            elif kind == "browse":
                reply = {"type": "results", "items": []}
            else:
                reply = {"type": "ok"}
        plugin.send(reply)


if __name__ == "__main__":
    main()
