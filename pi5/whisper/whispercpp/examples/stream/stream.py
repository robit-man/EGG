import sys
import typing as t
import time
import socket
import re
import math
import whispercpp as w

class StreamTranscriber:
    def __init__(
        self,
        model_name: str,
        target_host: str,
        target_port: int,
        asr_blip_host: str = "127.0.0.1",
        asr_blip_port: int = 6353,
        asr_blip_rate: int = 22050,
        asr_blip_channels: int = 1,
    ):
        self.transcriber = w.Whisper.from_pretrained(model_name)
        self.paused = False
        self.target_host = target_host
        self.target_port = target_port
        self.asr_blip_host = str(asr_blip_host or "127.0.0.1")
        self.asr_blip_port = int(asr_blip_port or 6353)
        self.asr_blip_rate = max(8000, int(asr_blip_rate or 22050))
        self.asr_blip_channels = max(1, min(2, int(asr_blip_channels or 1)))
        self._blip_last_at = 0.0

    def clean_text(self, text: str) -> str:
        """
        Remove text inside square brackets and ensure the text contains valid characters.
        Only return text if it has meaningful content (e.g., alphanumeric or linguistic characters).
        """
        # Remove content inside square brackets
        cleaned = re.sub(r"\[.*?\]", "", text).strip()
        
        # Check if the cleaned text contains meaningful characters (letters, digits, etc.)
        if not re.search(r"[a-zA-Z0-9]", cleaned):
            return ""  # Return empty string if no meaningful content remains
        
        return cleaned

    def send_to_server(self, text: str):
        """Send cleaned transcribed text to the server."""
        cleaned_text = self.clean_text(text)
        if cleaned_text:  # Only send if there's text left after cleaning
            self.play_asr_blip()
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                    client_socket.connect((self.target_host, self.target_port))
                    client_socket.sendall(cleaned_text.encode("utf-8"))
                    print(f"Sent to server: {cleaned_text}")
            except Exception as e:
                print(f"Error sending data to server: {e}")

    def play_asr_blip(self):
        # Throttle cue emission slightly to avoid machine-gun beeps on bursty segment callbacks.
        now = time.time()
        if (now - float(self._blip_last_at or 0.0)) < 0.15:
            return
        self._blip_last_at = now
        try:
            sample_rate = max(8000, int(self.asr_blip_rate or 22050))
            channels = max(1, min(2, int(self.asr_blip_channels or 1)))
            duration_seconds = 0.2
            requested_hz = 12000.0
            max_hz = (sample_rate * 0.5) - 120.0
            if max_hz < 200.0:
                max_hz = 200.0
            tone_hz = min(requested_hz, max_hz)
            amplitude = 0.18
            frame_count = max(1, int(sample_rate * duration_seconds))
            pcm = bytearray(frame_count * channels * 2)
            omega = (2.0 * math.pi * tone_hz) / float(sample_rate)
            idx = 0
            for i in range(frame_count):
                value = int(32767.0 * amplitude * math.sin(omega * i))
                if value > 32767:
                    value = 32767
                if value < -32768:
                    value = -32768
                lo = value & 0xFF
                hi = (value >> 8) & 0xFF
                for _ in range(channels):
                    pcm[idx] = lo
                    pcm[idx + 1] = hi
                    idx += 2
            with socket.create_connection((self.asr_blip_host, int(self.asr_blip_port)), timeout=0.25) as sock:
                sock.sendall(pcm)
        except Exception:
            pass

    def store_transcript_handler(self, ctx, n_new, data):
        segment = ctx.full_n_segments() - n_new
        cur_segment = ""
        while segment < ctx.full_n_segments():
            cur_segment = ctx.full_get_segment_text(segment)
            data.append(cur_segment)
            self.send_to_server(cur_segment)  # Send transcription downstream
            segment += 1
        

    def main(self, **kwargs: t.Any):
        transcription: t.Iterator[str] | None = None
        runtime_kwargs = dict(kwargs or {})
        for ignored_key in (
            "target_host",
            "target_port",
            "asr_blip_host",
            "asr_blip_port",
            "asr_blip_rate",
            "asr_blip_channels",
        ):
            runtime_kwargs.pop(ignored_key, None)
        try:
            transcription = self.transcriber.stream_transcribe(
                callback=self.store_transcript_handler,
                **runtime_kwargs,
            )
        finally:
            assert transcription is not None, "Something went wrong!"
            sys.stderr.writelines(
                ["\nTranscription (line by line):\n"] + [f"{it}\n" for it in transcription]
            )
            sys.stderr.flush()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", default="tiny.en", choices=list(w.utils.MODELS_URL)
    )
    parser.add_argument(
        "--device_id", type=int, help="Choose the audio device", default=0
    )
    parser.add_argument(
        "--length_ms",
        type=int,
        help="Length of the audio buffer in milliseconds",
        default=5000,
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        help="Sample rate of the audio device",
        default=w.api.SAMPLE_RATE,
    )
    parser.add_argument(
        "--n_threads",
        type=int,
        help="Number of threads to use for decoding",
        default=3,
    )
    parser.add_argument(
        "--step_ms",
        type=int,
        help="Step size of the audio buffer in milliseconds",
        default=0,
    )
    parser.add_argument(
        "--keep_ms",
        type=int,
        help="Length of the audio buffer to keep in milliseconds",
        default=200,
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        help="Maximum number of tokens to decode",
        default=32,
    )
    parser.add_argument("--audio_ctx", type=int, help="Audio context", default=512)
    parser.add_argument(
        "--list_audio_devices",
        action="store_true",
        default=False,
        help="Show available audio devices",
    )
    parser.add_argument(
        "--target_host",
        type=str,
        help="Target host to send transcriptions",
        default="127.0.0.1",
    )
    parser.add_argument(
        "--target_port",
        type=int,
        help="Target port to send transcriptions",
        default=6545,
    )
    parser.add_argument(
        "--asr_blip_host",
        type=str,
        help="Host for ASR cue tone raw PCM output.",
        default="127.0.0.1",
    )
    parser.add_argument(
        "--asr_blip_port",
        type=int,
        help="Port for ASR cue tone raw PCM output.",
        default=6353,
    )
    parser.add_argument(
        "--asr_blip_rate",
        type=int,
        help="Sample rate for ASR cue tone generation.",
        default=22050,
    )
    parser.add_argument(
        "--asr_blip_channels",
        type=int,
        help="Channel count for ASR cue tone generation.",
        default=1,
    )

    args = parser.parse_args()

    if args.list_audio_devices:
        w.utils.available_audio_devices()
        sys.exit(0)

    transcriber = StreamTranscriber(
        args.model_name,
        args.target_host,
        args.target_port,
        asr_blip_host=args.asr_blip_host,
        asr_blip_port=args.asr_blip_port,
        asr_blip_rate=args.asr_blip_rate,
        asr_blip_channels=args.asr_blip_channels,
    )
    transcriber.main(**vars(args))
