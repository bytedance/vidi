# Vidi1.5-9B Inference and Finetune

We release the 9B model weight for reproduction of Vidi1.5 results in 2025/07/15 tech report.

First download the checkpoint from [https://huggingface.co/bytedance-research/Vidi1.5-9B](https://huggingface.co/bytedance-research/Vidi1.5-9B).

Then run [install.sh](Vidi1.5_9B/install.sh) in "./Vidi1.5_9B":
```
cd Vidi1.5_9B
bash install.sh
```

## Inference
For a given video (e.g., [example_video](https://drive.google.com/file/d/1PZXUmTwUivFV_0nRhAnVR4LO9N9AAA1e/view?usp=sharing)) and text query (e.g., slicing onion), run the following command to get the results:

```
python3 -u vidi/eval/inference.py --video-path [video path] --query [query] --model-path [model path]
``` 

## Finetune
We have tested the code on a single node machine with 8 GPUs. 

1. Modify the example.json to your own data json. Change the dummy video path to your own path and include the video length/duration.
2. Modify all the paths in the command [finetune.sh](Vidi1.5_9B/scripts/finetune.sh) and run:
```
bash scripts/finetune.sh
```

