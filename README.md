# [Vidi2: Large Multimodal Models for Video Understanding and Creation](https://arxiv.org/pdf/2511.19529)

Homepage: https://bytedance.github.io/vidi-website/

> We introduce Vidi, a family of Large Multimodal Models (LMMs) for a wide range of video understanding and editing (VUE) scenarios. The first release focuses on temporal retrieval (TR), i.e., identifying the time ranges in input videos corresponding to a given text query. The second release evolves toward a foundation model with state-of-the-art spatio-temporal grounding (STG) and temporal retrieval capability while maintaining basic open-ended video QA performance.

## Release
- [11/25/2025] 🔥 Vidi2 released at [Report](https://arxiv.org/pdf/2511.19529), [Github](https://github.com/bytedance/vidi), [Homepage](https://bytedance.github.io/vidi-website/), [Demo](https://vidi.byteintl.com/).
- [08/29/2025] 🔥 Vidi1.5-9B demo released at https://vidi.byteintl.com/ with new UI design.
- [06/06/2025] 🔥 Vidi-7B demo released at https://vidi.byteintl.com/. Follow the instructions in the [demo](#demo) section to run the demo.
- [04/21/2025] 🔥 The first release of Vidi consists of tech report and the VUE-TR evaluation benchmark. The 7B model demo and weights are coming soon. 

## Content
- [Demo](https://vidi.byteintl.com/)
- [Installation](#installation)
- [Evaluation (VUE-STG)](#evaluation-vue-stg)
- [Evaluation (VUE-TR-V2)](#evaluation-vue-tr-v2)
- [Model](#model-and-inference)
- [ ] Vidi1.5-9B Weight and inference code
- [ ] Vidi1.5-9B Sample finetune code


## Demo
1. Select a mode from ["Highlight", "VQA", "Retrieval", "Grounding"] on the segmented button. Please use English query for the best experience.

- "Highlight": No input query needed. Directly output a set of highlight clips with title.

- "Grounding": Input a text query indicating the object to be searched. The model will find the clips corresponding to text query with bounding boxes on the object.

- "Retrieval": Input a text query to be searched. The model will find the clips corresponding to text query.

- "VQA": Input a question/instruction about the video. The model will answer the question.


2. Click "Upload" button and select a video local file (mp4 format). Make sure the video is not corrupted, and the resolution is not too high. 480p is recommended for fast uploading and decoding.
2. After the video is uploaded, wait till the uploading is finished and the video is ready to play in the box.
3. Enter the text query if needed. Click the "Send" button.
4. Wait till the result clips show in the chat box. This could take several minutes for long video.

## Installation
Run the [install.sh](install.sh).



## Evaluation (VUE-STG)

We release the video ids, ground-truth annotation and evaluation results in csv files. Follow the instruction in [VUE_STG/README.md](VUE_STG/README.md) to conduct evaluation.
```
cd VUE_STG
python3 evaluate.py
```

To evaluate your own model:
1. First download the videos based on the ids in ["VUE_STG/vue-stg-benchmark/video.csv"](VUE_STG/vue-stg-benchmark/video.csv) from Youtube (e.g., [yt-dlp
](https://github.com/yt-dlp/yt-dlp)). 
2. Generate the result following the format in [VUE_STG/results/vidi2/tubes.csv](VUE_STG/results/vidi2/tubes.csv). Run evaluation script.


## Evaluation (VUE-TR-V2)
We release the ground-truth annotation and evaluation results in 5 json files. Run the script for a standalone evaluation:
```
cd VUE_TR_V2
python3 -u qa_eval.py --pred_path results_Vidi.json
```
The result figures will be saved in the output folder ('./results' by default)
.

<img src="VUE_TR_V2/results/IoU_radar_plot.png" width="300"/> <img src="VUE_TR_V2/results/overall_IoU_plot.png" width="377"/> 

For evaluation of new models, first download the videos based on the ids in [VUE_TR_V2/video_id.txt](VUE_TR_V2/video_id.txt) from Youtube (e.g., [yt-dlp
](https://github.com/yt-dlp/yt-dlp)). Then run inference and save the results in the following format:
```
[
    {
        "query_id": 0,
        "video_id": "6Qv-LrXJjSM",
        "duration": 3884.049,
        "query": "The slide showcases Taco Bell's purple ang pow for Chinese New Year, while a woman explains that purple symbolizes royalty in the Chinese tradition.",
        "answer": [
            [
                913.1399199,
                953.5340295
            ]
        ],
        "task": "temporal_retrieval"
    },
    ...
]
```

You may find the instruction and data for the previous version (VUE-TR) [here](VUE_TR/README.md).


## Model and Inference
We release the 7B model weight for reproduction of Vidi results in 2025/04/15 tech report. 

First download the checkpoint from [https://huggingface.co/bytedance-research/Vidi-7B](https://huggingface.co/bytedance-research/Vidi-7B).

Then run [install.sh](Vidi_7B/install.sh) in "./Vidi_7B":
```
cd Vidi_7B
bash install.sh
```

For a given video (e.g., [example_video](https://drive.google.com/file/d/1PZXUmTwUivFV_0nRhAnVR4LO9N9AAA1e/view?usp=sharing)) and text query (e.g., slicing onion), run the following command to get the results:

```
python3 -u inference.py --video-path [video path] --query [query] --model-path [model path]
``` 

## Citation
If you find Vidi useful for your research and applications, please cite using this BibTeX:
```
@article{Vidi2025vidi2,
    title={Vidi2: Large Multimodal Models for Video 
            Understanding and Creation},
    author={Vidi Team, Celong Liu, Chia-Wen Kuo, Chuang Huang, 
            Dawei Du, Fan Chen, Guang Chen, Haoji Zhang, 
            Haojun Zhao, Lingxi Zhang, Lu Guo, Lusha Li, 
            Longyin Wen, Qihang Fan, Qingyu Chen, Rachel Deng,
            Sijie Zhu, Stuart Siew, Tong Jin, Weiyan Tao,
            Wen Zhong, Xiaohui Shen, Xin Gu, Zhenfang Chen, Zuhua Lin},
    journal={arXiv preprint arXiv:2511.19529},
    year={2025}
}
@article{Vidi2025vidi,
    title={Vidi: Large Multimodal Models for Video 
            Understanding and Editing},
    author={Vidi Team, Celong Liu, Chia-Wen Kuo, Dawei Du, 
            Fan Chen, Guang Chen, Jiamin Yuan, Lingxi Zhang,
            Lu Guo, Lusha Li, Longyin Wen, Qingyu Chen, 
            Rachel Deng, Sijie Zhu, Stuart Siew, Tong Jin, 
            Wei Lu, Wen Zhong, Xiaohui Shen, Xin Gu, Xing Mei, 
            Xueqiong Qu, Zhenfang Chen},
    journal={arXiv preprint arXiv:2504.15681},
    year={2025}
}
```
