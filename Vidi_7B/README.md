# Vidi-7B Inference

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