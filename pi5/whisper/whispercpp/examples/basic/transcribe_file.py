import argparse
import whispercpp as w

def transcribe_file(file_path, model_name):
    transcriber = w.Whisper.from_pretrained(model_name)
    res = transcriber.transcribe_from_file(file_path)
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", default="../../samples/jfk.wav", help="Path to the audio file", nargs='?')
    parser.add_argument("--model_name", default="tiny.en", choices=list(w.utils.MODELS_URL), help="Name of the model to use for transcription")
    args = parser.parse_args()

    res = transcribe_file(args.file_path, args.model_name)
    print(res)
