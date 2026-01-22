"""
Copyright 2025 Intelligent Editing Team.
"""
import sys
import warnings


class FilteredStream:
    def __init__(self, stream, banned_texts):
        self.stream = stream
        self.banned_texts = banned_texts

    def write(self, data):
        # Only write data if none of the banned texts appear
        if not any(banned in data for banned in self.banned_texts):
            return self.stream.write(data)
        # Optionally, you can return the number of characters that would have been written.
        return len(data)

    def flush(self):
        return self.stream.flush()

    def __getattr__(self, attr):
        # Delegate any other attribute/method calls to the underlying stream.
        return getattr(self.stream, attr)


banned_messages = [
    "You are attempting to use Flash Attention 2.0",
    "This is not supported for all configurations of models and can yield errors."
]
sys.stdout = FilteredStream(sys.stdout, banned_messages)
sys.stderr = FilteredStream(sys.stderr, banned_messages)

warnings.filterwarnings("ignore", message=".*torch.cpu.amp.autocast.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*bytedmetrics is renamed to bytedance.metrics.*", category=UserWarning)