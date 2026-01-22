"""
Copyright 2025 Intelligent Editing Team.
"""

import argparse
import os
from vidi.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from vidi.model.builder import load_pretrained_model
from vidi.dataset.img_utils import process_images
from vidi.dataset.txt_utils import tokenizer_image_token, preprocess_chat
from vidi.dataset.vid_utils import load_video, load_audio, process_audio

import torch
import re
import subprocess
# ========================================================
# ask model
def ask(question, vid_path, model, tokenizer, image_processor, audio_processor):
    # Check if the video exists
    if os.path.exists(vid_path):
        video = load_video(vid_path)
        video = process_images(video, image_processor, model.config)
        video = video.unsqueeze(0).half().cuda()

        audio = load_audio(vid_path, audio_processor.sampling_rate)
        audio_tensor, audio_size = process_audio(audio, audio_processor)
        audio = audio_tensor.unsqueeze(0).half().cuda()
        length = get_length(vid_path)
    else:
        print("Video not found.")
        raise Exception

    # question_tmp = "Given the frames from a video, answer the time range in percentage that corresponds to query text split by comma. Video length is: {:.2f} and text query is: {}.".format(length, question[:-1] if question.endswith('.') else question)
    question_tmp = "During which time segments in the video can we see {}?".format(question[:-1] if question.endswith('.') else question)
    qs = DEFAULT_IMAGE_TOKEN + '\n' + question_tmp
    prompt = preprocess_chat([{"from": "human", "value": qs}], tokenizer)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=video,
            audios=audio,
            audio_sizes=[audio_size],
            do_sample=False,
            max_new_tokens=1024,
            use_cache=True,
            disable_compile=True,
            pad_token_id=tokenizer.pad_token_id
            )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    # print(outputs)
    pattern = re.compile(r'(\d\.\d+)-(\d\.\d+)')
    results = pattern.findall(outputs)
    output_list = []
    
    for result in results:
        t_0 = float(result[0])*length
        t_1 = float(result[1])*length
        output_list.append('{:02d}:{:02d}:{:02d}-{:02d}:{:02d}:{:02d}'.format(
            int(t_0/3600), (int(t_0)%3600)//60, int(t_0)%60, 
            int(t_1/3600), (int(t_1)%3600)//60, int(t_1)%60
            ))
    return ', '.join(output_list)

def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename], 
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return float(result.stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #============================================
    parser.add_argument("--video-path", type=str, default="video path")
    parser.add_argument("--query", type=str, default="text query")
    parser.add_argument("--model-path", type=str, default="model path")
    args = parser.parse_args()
    # ========================================================
    # load model
    model, tokenizer, image_processor, audio_processor = load_pretrained_model(args.model_path)
    model.config.mm_splits = 32
    print(ask(args.query, args.video_path, model, tokenizer, image_processor, audio_processor))

