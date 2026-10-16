"""Offline tests: synthetic audio, a fake recorder and a fake engine. No
microphone, no network, no faster-whisper. Run with `python3 -m unittest`."""
import array
import fcntl
import io
import json
import math
import os
import random
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import main


def noise(seconds, amp=15, seed=1):
    rng = random.Random(seed)
    return array.array("h", [int(rng.gauss(0, amp)) for _ in range(int(seconds * main.RATE))]).tobytes()


def tone(seconds, amp=3000):
    """A voice-like buzz: 200 Hz with a few harmonics."""
    n = int(seconds * main.RATE)
    return array.array("h", [int(amp / 3 * sum(math.sin(2 * math.pi * f * i / main.RATE)
                                              for f in (200, 400, 800)))
                             for i in range(n)]).tobytes()


def run_gate(pcm, **kw):
    gate = main.Gate(**kw)
    step = 4800  # the recorder hands over 150 ms at a time
    for offset in range(0, len(pcm), step):
        reason = gate.feed(pcm[offset:offset + step])
        if reason:
            return reason, gate
    return None, gate


def script(path, body):
    with open(path, "w") as f:
        f.write(body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    return path


# Stands in for the venv's python running transcribe.py: reads the audio,
# answers like it. FAKE_FAIL makes it fail like a broken engine.
FAKE_PYTHON = r"""#!/usr/bin/env python3
import json, os, sys
audio = sys.stdin.buffer.read()
with open(os.environ["FAKE_LOG"], "w") as f:
    json.dump({"argv": sys.argv[1:], "bytes": len(audio)}, f)
if os.environ.get("FAKE_FAIL"):
    print(json.dumps({"error": "CUDA driver not found"}))
    sys.exit(1)
print("a library talking on stdout")
print(json.dumps({"loaded": True}))
print(json.dumps({"partial": "hello"}))
print(json.dumps({"text": " hello world ", "language": "en"}))
"""

# Stands in for pw-record: speech-like audio, then quiet, then waits.
FAKE_RECORDER = r"""#!/usr/bin/env python3
import sys, time
sys.stdout.buffer.write(open(sys.argv[1], "rb").read())
sys.stdout.flush()
time.sleep(30)
"""


class GateTest(unittest.TestCase):
    def test_stops_after_a_pause_once_speech_started(self):
        reason, gate = run_gate(noise(0.6) + tone(1.0) + noise(5), silence=1.5)
        self.assertEqual(reason, "silence")
        self.assertTrue(gate.speech)
        self.assertAlmostEqual(gate.seconds, 0.6 + 1.0 + 1.5, delta=0.2)

    def test_a_short_pause_inside_speech_doesnt_stop_it(self):
        pcm = noise(0.5) + tone(1) + noise(0.8) + tone(1) + noise(3)
        reason, gate = run_gate(pcm, silence=1.5)
        self.assertEqual(reason, "silence")
        self.assertGreater(gate.seconds, 0.5 + 1 + 0.8 + 1 + 1.3)

    def test_nobody_speaks(self):
        reason, gate = run_gate(noise(10))
        self.assertEqual(reason, "nothing")
        self.assertFalse(gate.speech)
        self.assertAlmostEqual(gate.seconds, main.NO_SPEECH, delta=0.1)

    def test_time_limit(self):
        reason, gate = run_gate(noise(0.5) + tone(6), max_seconds=4)
        self.assertEqual(reason, "max")
        self.assertAlmostEqual(gate.seconds, 4, delta=0.1)

    def test_digital_silence_and_a_click_are_not_speech(self):
        click = array.array("h", [20000, -20000] * 40).tobytes()
        reason, gate = run_gate(bytes(16000) + click + bytes(16000 * 18))
        self.assertEqual(reason, "nothing")

    def test_a_fading_dc_offset_is_not_speech(self):
        # What a real laptop microphone sends in its first second.
        hiss = array.array("h")
        hiss.frombytes(noise(10))
        pcm = array.array("h", [int(26000 * math.exp(-i / 2400)) + h for i, h in enumerate(hiss)])
        reason, gate = run_gate(pcm.tobytes())
        self.assertEqual(reason, "nothing")

    def test_a_fan_that_starts_doesnt_keep_it_listening(self):
        reason, gate = run_gate(noise(0.5, 3) + tone(1) + noise(20, 60), silence=1.5)
        self.assertEqual(reason, "silence")
        self.assertLess(gate.seconds, 0.5 + 1 + main.FLOOR_WINDOW + 1.5 + 0.2)

    def test_speaking_right_away_still_counts(self):
        # The calibration hears the first word; the floor drops in the pause.
        pcm = tone(0.5) + noise(0.4) + tone(1) + noise(3)
        reason, gate = run_gate(pcm, silence=1.5)
        self.assertEqual(reason, "silence")
        self.assertTrue(gate.speech)

    def test_meter(self):
        self.assertEqual(main.meter(0, 10), "░" * 8)
        self.assertEqual(main.meter(10 * 10 ** 1.5, 10), "█" * 8)
        self.assertEqual(len(main.meter(100, None)), 8)


class Parsing(unittest.TestCase):
    def test_status_defaults(self):
        for junk in ("", "not json", "[1, 2]", '{"state": "weird"}'):
            status = main.parse_status(junk)
            self.assertEqual(status["state"], "idle")
            self.assertEqual(status["message"], "")

    def test_status_fields(self):
        status = main.parse_status(json.dumps(
            {"state": "running", "stage": "pip", "message": "Downloading av (35.0 MB)",
             "model": "base", "updated": 12.5}))
        self.assertEqual((status["state"], status["stage"], status["model"], status["updated"]),
                         ("running", "pip", "base", 12.5))

    def test_pip_messages(self):
        self.assertEqual(main.pip_message("Collecting faster-whisper>=1.0"),
                         "Looking up faster-whisper>=1.0")
        self.assertEqual(main.pip_message(
            "Downloading ctranslate2-4.8.3-cp312-cp312-manylinux_2_28_x86_64.whl (40.3 MB)"),
            "Downloading ctranslate2 (40.3 MB)")
        self.assertIsNone(main.pip_message(
            "  Downloading av-19.0.1-cp312-abi3-manylinux_2_28_x86_64.whl.metadata (5.0 kB)"))
        self.assertEqual(main.pip_message("Installing collected packages: av, numpy"),
                         "Installing the speech engine")
        self.assertIsNone(main.pip_message("Requirement already satisfied: numpy"))


class EngineTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.engine = main.Engine(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def fake_venv(self):
        os.makedirs(os.path.dirname(self.engine.python))
        script(self.engine.python, "#!/bin/sh\n")

    def test_ready_needs_venv_engine_and_model(self):
        self.assertFalse(self.engine.ready("base"))
        self.fake_venv()
        self.engine.mark(engine=True)
        self.assertTrue(self.engine.engine_ready())
        self.assertFalse(self.engine.ready("base"))
        self.engine.mark(model="base")
        self.assertTrue(self.engine.ready("base"))
        self.assertFalse(self.engine.ready("small"))

    def test_another_python_version_means_set_up_again(self):
        self.fake_venv()
        main.write_json(self.engine.ready_path, {"engine": True, "python": "2.7", "models": ["base"]})
        self.assertFalse(self.engine.ready("base"))

    def test_lock_tells_a_running_setup(self):
        self.assertFalse(self.engine.setup_running())
        with open(self.engine.lock_path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertTrue(self.engine.setup_running())
        self.assertFalse(self.engine.setup_running())

    def test_setup_resumes_and_skips_whats_done(self):
        self.fake_venv()
        self.engine.mark(engine=True, model="tiny")
        with mock.patch.object(main, "make_venv") as venv, \
                mock.patch.object(main, "fetch_model") as fetch:
            self.assertEqual(main.run_setup(self.dir.name, "tiny"), 0)
        venv.assert_not_called()
        fetch.assert_not_called()
        self.assertEqual(self.engine.status()["state"], "done")

    def test_setup_fetches_only_a_new_model(self):
        self.fake_venv()
        self.engine.mark(engine=True, model="tiny")
        with mock.patch.object(main, "make_venv") as venv, \
                mock.patch.object(main, "fetch_model") as fetch:
            main.run_setup(self.dir.name, "small")
        venv.assert_not_called()
        self.assertEqual(fetch.call_args[0][1], "small")
        self.assertTrue(self.engine.ready("small"))

    def test_setup_failure_goes_to_the_status_file(self):
        with mock.patch.object(main, "make_venv", side_effect=main.SetupError("no network")):
            self.assertEqual(main.run_setup(self.dir.name, "base"), 1)
        status = self.engine.status()
        self.assertEqual((status["state"], status["message"]), ("error", "no network"))

    def test_a_second_setup_leaves_the_first_alone(self):
        with open(self.engine.lock_path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with mock.patch.object(main, "make_venv") as venv:
                self.assertEqual(main.run_setup(self.dir.name, "base"), 0)
        venv.assert_not_called()
        self.assertEqual(self.engine.status()["state"], "idle")


class Plugin(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.out = io.StringIO()
        self.plugin = main.Voice(out=self.out, root=os.path.join(self.dir.name, "data"))
        self.plugin.init({"model": "base", "max_seconds": 20})
        self.log = os.path.join(self.dir.name, "log.json")
        self.env = mock.patch.dict(os.environ, {"FAKE_LOG": self.log})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.dir.cleanup()

    def make_ready(self):
        engine = self.plugin.engine
        os.makedirs(os.path.dirname(engine.python))
        script(engine.python, FAKE_PYTHON)
        engine.mark(engine=True, model="base")

    def recorder(self, pcm):
        audio = os.path.join(self.dir.name, "audio.raw")
        with open(audio, "wb") as f:
            f.write(pcm)
        path = script(os.path.join(self.dir.name, "recorder"), FAKE_RECORDER)
        return mock.patch.object(main, "recorder_argv", return_value=[path, audio])

    def progress(self):
        return [json.loads(l) for l in self.out.getvalue().splitlines()]

    def test_config_is_clamped(self):
        self.plugin.init({"model": "small", "language": "IT", "max_seconds": 600,
                          "silence_seconds": "x"})
        p = self.plugin
        self.assertEqual((p.model, p.language, p.max_seconds, p.silence), ("small", "it", 60, 1.5))
        self.assertEqual((p.device, p.compute_type), ("cpu", "int8"))

    def test_query_is_one_slow_listen_item(self):
        [item] = self.plugin.query("")
        self.assertEqual(item["title"], "Start listening")
        self.assertIn("downloads", item["subtitle"])
        [action] = item["actions"]
        self.assertEqual(action, {"label": "Voice", "browse": {"mode": "listen"},
                                  "slow": True, "waiting": "Listening…"})
        self.assertFalse(os.path.exists(self.plugin.engine.root), "a query writes nothing")

    def test_query_when_ready(self):
        self.make_ready()
        [item] = self.plugin.query("")
        self.assertIn("Enter pastes", item["subtitle"])

    def test_actions_paste_then_search_then_copy(self):
        [item, again] = main.result_items("ciao mondo", "it")
        self.assertEqual(item["actions"], [{"label": "Paste", "paste": "ciao mondo"},
                                           {"label": "Search", "query": "ciao mondo"},
                                           {"label": "Copy", "copy": "ciao mondo"}])
        self.assertEqual(item["title"], "ciao mondo")
        self.assertNotIn("text", item)
        self.assertTrue(again["actions"][0]["slow"])

    def test_long_text_goes_in_full_under_the_title(self):
        text = "word " * 40
        [item, _] = main.result_items(text.strip())
        self.assertTrue(item["title"].endswith("…"))
        self.assertEqual(item["text"], text.strip())

    def test_listen_records_then_transcribes(self):
        self.make_ready()
        with self.recorder(noise(0.5) + tone(1) + noise(2)):
            [item, again] = self.plugin.browse({"mode": "listen"})
        self.assertEqual(item["title"], "hello world")
        self.assertEqual(item["actions"][0], {"label": "Paste", "paste": "hello world"})
        self.assertEqual(again["title"], "Listen again")
        titles = [m["items"][0]["title"] for m in self.progress()]
        self.assertTrue(titles[0].startswith("Listening… "))
        self.assertIn("Transcribing…", titles)
        with open(self.log) as f:
            call = json.load(f)
        self.assertEqual(call["argv"][0], main.TRANSCRIBE)
        self.assertEqual(call["argv"][call["argv"].index("--model") + 1], "base")
        self.assertGreater(call["bytes"], 32000 * 2)

    def test_transcription_failure_is_shown(self):
        self.make_ready()
        with self.recorder(noise(0.5) + tone(1) + noise(2)), \
                mock.patch.dict(os.environ, {"FAKE_FAIL": "1"}):
            [item, _] = self.plugin.browse({"mode": "listen"})
        self.assertEqual(item["title"], "Couldn't transcribe")
        self.assertIn("CUDA driver", item["text"])

    def test_silent_microphone(self):
        self.make_ready()
        with self.recorder(noise(9)):
            [item, again] = self.plugin.browse({"mode": "listen"})
        self.assertEqual(item["title"], "Didn't hear anything")
        self.assertFalse(os.path.exists(self.log), "nothing to transcribe")

    def test_no_audio_at_all(self):
        self.make_ready()
        with self.recorder(b""), mock.patch.object(main, "NO_AUDIO", 0.3):
            [item, _] = self.plugin.browse({"mode": "listen"})
        self.assertEqual(item["title"], "No sound from the microphone")

    def test_recorder_that_fails(self):
        self.make_ready()
        argv = [sys.executable, "-c", "import sys; sys.exit('no such node: mic')"]
        with mock.patch.object(main, "recorder_argv", return_value=argv):
            [item, _] = self.plugin.browse({"mode": "listen"})
        self.assertEqual(item["title"], "Can't record")
        self.assertIn("no such node", item["text"])

    def test_no_recorder(self):
        self.make_ready()
        with mock.patch.object(main, "recorder_argv", return_value=None):
            [item] = self.plugin.browse({"mode": "listen"})
        self.assertEqual(item["title"], "No audio recorder")

    def test_missing_venv_module(self):
        with mock.patch.object(main, "venv_missing", return_value=True):
            [item] = self.plugin.browse({"mode": "listen"})
        self.assertEqual(item["title"], "Install python3-venv first")
        self.assertEqual(item["actions"], [{"label": "Copy", "copy": "sudo apt install python3-venv"}])

    def test_setup_streams_progress_then_says_ready(self):
        calls = []

        def fake_setup():
            calls.append(1)
            self.make_ready()

        with mock.patch.object(self.plugin, "start_setup", side_effect=fake_setup), \
                mock.patch.object(main, "venv_missing", return_value=False):
            [item] = self.plugin.browse({"mode": "listen"})
        self.assertEqual(item["title"], "Voice typing is ready")
        self.assertEqual(item["actions"][0]["browse"], {"mode": "listen"})
        self.assertEqual(calls, [1])
        self.assertTrue(self.progress()[0]["items"][0]["title"].startswith("Setting up"))

    def test_failed_setup_offers_a_retry(self):
        engine = self.plugin.engine

        def fake_setup():
            os.makedirs(engine.root, exist_ok=True)
            engine.write_status("error", "error", "pip: no network", "base")

        with mock.patch.object(self.plugin, "start_setup", side_effect=fake_setup), \
                mock.patch.object(main, "venv_missing", return_value=False), \
                mock.patch.object(main, "POLL", 0.01):
            [item] = self.plugin.browse({"mode": "listen"})
        self.assertEqual(item["title"], "Voice typing couldn't be set up")
        self.assertIn("no network", item["text"])
        self.assertTrue(item["actions"][0]["slow"])

    def test_long_setup_carries_on_in_the_background(self):
        with mock.patch.object(self.plugin, "setup_active", return_value=True), \
                mock.patch.object(main, "venv_missing", return_value=False), \
                mock.patch.object(main, "SETUP_BUDGET", 0.3), \
                mock.patch.object(main, "POLL", 0.05):
            [item] = self.plugin.browse({"mode": "listen"})
        self.assertEqual(item["title"], "Still setting up voice typing")
        self.assertIn("background", item["text"])

    def test_junk_data_is_ignored(self):
        self.assertEqual(self.plugin.browse({"mode": "other"}), [])
        self.assertEqual(self.plugin.browse("nope"), [])


class Protocol(unittest.TestCase):
    def test_one_line_per_message(self):
        with tempfile.TemporaryDirectory() as home:
            lines = "\n".join(json.dumps(m) for m in [
                {"type": "init", "version": "1.8.0", "config": {"model": "tiny"}},
                {"type": "query", "id": 7, "text": "", "keyword": "say"},
                "not json",
                {"type": "activate", "item": "x", "data": None, "text": ""},
            ]) + "\n"
            env = dict(os.environ, XDG_DATA_HOME=home)
            proc = subprocess.run([sys.executable, main.__file__], input=lines, env=env,
                                  capture_output=True, text=True, timeout=10)
            replies = [json.loads(l) for l in proc.stdout.splitlines()]
            self.assertEqual(os.listdir(home), [])
        self.assertEqual([r["type"] for r in replies], ["ready", "results", "ok"])
        self.assertEqual(replies[1]["id"], 7)
        self.assertEqual(replies[1]["items"][0]["title"], "Start listening")


if __name__ == "__main__":
    unittest.main()
