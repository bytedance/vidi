import numpy as np
from subprocess import run
from PIL import Image
from decord import VideoReader, cpu


def load_video(file, fps=1., time_range=None, num_threads=0):
    vr = VideoReader(str(file), ctx=cpu(0), num_threads=num_threads)
    if time_range is None:
        sample_fps = round(vr.get_avg_fps()/fps)
        frame_idx = [i for i in range(0, len(vr), sample_fps)]
    else:
        idx_s = time_range[0] * vr.get_avg_fps()
        idx_e = time_range[1] * vr.get_avg_fps()
        num_steps = (time_range[1] - time_range[0]) * fps
        frame_idx = np.linspace(round(idx_s), round(idx_e), round(num_steps), dtype=int)
    video = vr.get_batch(frame_idx).asnumpy()
    video = [Image.fromarray(x).convert('RGB') for x in video]
    
    return video


def load_audio(file, sample_rate=16000, time_range=None):
    # follow whisper's audio loading strategy
    # Ref: https://github.com/openai/whisper/blob/main/whisper/audio.py#L25-L62
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file
    ]
    if time_range is not None:
        cmd += [
            "-ss", f"{time_range[0]:.2f}",
            "-t", f"{time_range[1]-time_range[0]:.2f}"
        ]
    cmd += [
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-"
    ]
    out = run(cmd, capture_output=True, check=True).stdout
    out = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    return out


def process_audio(audio, audio_processor):
    audios = [
        audio[i:i + audio_processor.n_samples]
        for i in range(0, len(audio), audio_processor.n_samples)
    ]
    audios = audio_processor(
        audios, sampling_rate=audio_processor.sampling_rate,
        return_tensors='pt', return_token_timestamps=True
    )
    length = int(audios.num_frames.sum())

    return audios.input_features, length
