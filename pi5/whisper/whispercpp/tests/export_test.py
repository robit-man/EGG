from __future__ import annotations

import shutil as s
import typing as t
from pathlib import Path

import pytest as p

import whispercpp as w
import wave

if t.TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
else:
    np = w.utils.LazyLoader("np", globals(), "numpy")

ROOT = Path(__file__).parent.parent
JFK_WAV = ROOT.joinpath("samples", "jfk.wav")
_EXPECTED = " And so my fellow Americans ask not what your country can do for you ask what you can do for your country."

def preprocess(file: Path, sample_rate: int = 16000) -> NDArray[np.float32]:

    sound = wave.open(str(file), "rb")
    nchannels = sound.getnchannels()
    N = sound.getnframes()
    dstr = sound.readframes(N*nchannels)
    data = np.fromstring(dstr, np.int16)
    data = data.reshape(-1, nchannels)

    return data.flatten().astype(np.float32) / 32768.0

def test_invalid_models():
    with p.raises(RuntimeError):
        w.Whisper.from_pretrained("whisper_v0.1")
    with p.raises(RuntimeError):
        w.Whisper.from_params(
            "whisper_v0.1", w.api.Params.from_enum(w.api.SAMPLING_GREEDY)
        )

def test_invalid_filepaths():
    with p.raises(RuntimeError):
        w.Whisper.from_pretrained("nonexistent.bin")
    with p.raises(RuntimeError):
        w.Whisper.from_params(
            "nonexistent.bin", w.api.Params.from_enum(w.api.SAMPLING_GREEDY)
        )

def test_forbid_init():
    with p.raises(RuntimeError):
        w.Whisper()

def test_from_pretrained_name():
    m = w.Whisper.from_pretrained("tiny.en")
    assert _EXPECTED == m.transcribe(preprocess(JFK_WAV))

@p.mark.parametrize(
    "models", [path.__fspath__() for path in Path(__file__).parent.joinpath("models").glob("*.bin")]
)
def test_from_pretrained_file(models: str):
    m = w.Whisper.from_pretrained(models)
    assert _EXPECTED == m.transcribe(preprocess(JFK_WAV))


def test_from_params_name():
    m = w.Whisper.from_params("tiny.en", w.api.Params.from_enum(w.api.SAMPLING_GREEDY))
    assert _EXPECTED == m.transcribe(preprocess(JFK_WAV))


@p.mark.parametrize(
    "models", [path.__fspath__() for path in Path(__file__).parent.joinpath("models").glob("*.bin")]
)
def test_from_params_file(models: str):
    m = w.Whisper.from_params(models, w.api.Params.from_enum(w.api.SAMPLING_GREEDY))
    assert _EXPECTED == m.transcribe(preprocess(JFK_WAV))

def test_load_wav_file():
    np.testing.assert_almost_equal(
        preprocess(JFK_WAV),
        w.api.load_wav_file_mono(str(JFK_WAV.resolve())),
    )

def transcribe_strict():
    m = w.Whisper.from_pretrained("tiny.en", no_state=True)
    with p.raises(AssertionError, match="* and context is not initialized *"):
        m.transcribe_from_file(str(JFK_WAV.resolve()))

def test_transcribe_from_wav():
    m = w.Whisper.from_pretrained("tiny.en")
    assert (
        m.transcribe_from_file(
            ROOT.joinpath("samples", "jfk.wav").resolve().__fspath__()
        )
        == _EXPECTED
    )

def test_callback():
    def handleNewSegment(context: w.api.Context, n_new: int, text: list[str]):
        segment = context.full_n_segments() - n_new
        while segment < context.full_n_segments():
            text.append(context.full_get_segment_text(segment))
            print(text)
            segment += 1

    m = w.Whisper.from_pretrained("tiny.en")

    text = []
    m.params.on_new_segment(handleNewSegment, text)

    correct = m.transcribe(preprocess(ROOT / "samples" / "jfk.wav"))
    assert "".join(text) == correct

def test_progress_callback():
    def handleProgress(context: w.api.Context, progress: int, progresses: list[int]):
        progresses.append(progress)

    m = w.Whisper.from_pretrained("tiny.en")

    progresses = []
    m.params.on_progress(handleProgress, progresses)

    m.transcribe(preprocess(ROOT / "samples" / "jfk.wav"))
    assert len(progresses) > 0
