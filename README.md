# Stream-T1: Test-Time Scaling for Streaming Video Generation
<a href="https://ttttttttyj.github.io/StreamT1/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=blue"></a>
<a href="https://arxiv.org/pdf/2505.02192"><img src="https://img.shields.io/badge/arXiv-2505.02192-b31b1b.svg"></a>
<a href="https://www.apache.org/licenses/LICENSE-2.0.txt"><img src="https://img.shields.io/badge/License-Apache-yellow"></a>

## TODO List

- [x] Release the paper and project page.
- [x] Release the inference code.
- [x] Release test cases with our pretrained model, prompts, and reference image.

## Requirements
The inference are conducted on 1 A800 GPU (80GB VRAM)
## Setup
```
git clone https://github.com/Ttttttttyj/Stream-T1.git
cd Stream-T1

cd metrics
https://github.com/KlingAIResearch/VideoAlign.git
```

## Environment
All the tests are conducted in Linux. To set up our environment in Linux, please run:
```
conda create -n StreamT1 python=3.10 -y
conda activate StreamT1

pip install -r requirements.txt
```

## Checkpoints
1.base model checkpoints
```
huggingface-cli download Efficient-Large-Model/LongLive --local-dir longlive_models
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
```
2.reward model checkpoints
```
huggingface-cli download MizzenAI/HPSv3 --local-dir metrics/models/hpsv3_model
huggingface-cli download KlingTeam/VideoReward --local-dir metrics/models/videoalign
```
## Inference
```
bash stream_scaling.sh
```
## Citation:
Don't forget to cite this source if it proves useful in your research!
```
@article{wang2025dualreal,
  title={Stream-T1: Test-Time Scaling for Streaming Video Generation},
  author={Wang, Wenchuan and Huang, Mengqi and Tu, Yijing and Mao, Zhendong},
  journal={arXiv preprint arXiv:2505.02192},
  year={2025}
}
```
## Acknowledgement:
LongLive: the codebase and algorithm we built upon. Thanks for their wonderful work.