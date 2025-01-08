import sys
import typing as t
import time
import socket
import re
import whispercpp as w

class StreamTranscriber:
    def __init__(self, model_name: str, target_host: str, target_port: int):
        self.transcriber = w.Whisper.from_pretrained(model_name)
        self.paused = False
        self.target_host = target_host
        self.target_port = target_port

    def clean_text(self, text: str) -> str:
        """Remove text inside square brackets."""
        return re.sub(r"\[.*?\]", "", text).strip()

    def send_to_server(self, text: str):
        """Send cleaned transcribed text to the server."""
        cleaned_text = self.clean_text(text)
        if cleaned_text:  # Only send if there's text left after cleaning
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                    client_socket.connect((self.target_host, self.target_port))
                    client_socket.sendall(cleaned_text.encode("utf-8"))
                    print(f"Sent to server: {cleaned_text}")
            except Exception as e:
                print(f"Error sending data to server: {e}")

    def store_transcript_handler(self, ctx, n_new, data):
        segment = ctx.full_n_segments() - n_new
        cur_segment = ""
        while segment < ctx.full_n_segments():
            cur_segment = ctx.full_get_segment_text(segment)
            data.append(cur_segment)
            self.send_to_server(cur_segment)  # Send transcription downstream
            segment += 1
        if "hey toaster" in cur_segment.lower() and not self.paused:
            self.transcriber.pause_audio()
            self.paused = True
            print("Stopping")
            time.sleep(5)
            self.transcriber.resume_audio()
            self.paused = False
            print("Resumed")

    def main(self, **kwargs: t.Any):
        transcription: t.Iterator[str] | None = None
        try:
            transcription = self.transcriber.stream_transcribe(callback=self.store_transcript_handler, **kwargs)
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

    args = parser.parse_args()

    if args.list_audio_devices:
        w.utils.available_audio_devices()
        sys.exit(0)

    transcriber = StreamTranscriber(args.model_name, args.target_host, args.target_port)
    transcriber.main(**vars(args))
