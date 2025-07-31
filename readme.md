# System requirements
USE DOCKER!!!

# FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects
[[Paper]](https://arxiv.org/abs/2312.08344) [[Website]](https://nvlabs.github.io/FoundationPose/)

This is the official implementation of our paper to be appeared in CVPR 2024 (Highlight)

Contributors: Bowen Wen, Wei Yang, Jan Kautz, Stan Birchfield

We present FoundationPose, a unified foundation model for 6D object pose estimation and tracking, supporting both model-based and model-free setups. Our approach can be instantly applied at test-time to a novel object without fine-tuning, as long as its CAD model is given, or a small number of reference images are captured. We bridge the gap between these two setups with a neural implicit representation that allows for effective novel view synthesis, keeping the downstream pose estimation modules invariant under the same unified framework. Strong generalizability is achieved via large-scale synthetic training, aided by a large language model (LLM), a novel transformer-based architecture, and contrastive learning formulation. Extensive evaluation on multiple public datasets involving challenging scenarios and objects indicate our unified approach outperforms existing methods specialized for each task by a large margin. In addition, it even achieves comparable results to instance-level methods despite the reduced assumptions.


<img src="assets/intro.jpg" width="70%">

**🤖 For ROS version, please check [Isaac ROS Pose Estimation](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation), which enjoys TRT fast inference and C++ speed up.**

\
**🥇 No. 1 on the world-wide [BOP leaderboard](https://bop.felk.cvut.cz/leaderboards/pose-estimation-unseen-bop23/core-datasets/) (as of 2024/03) for model-based novel object pose estimation.**
<img src="assets/bop.jpg" width="80%">

## Demos

Robotic Applications:

https://github.com/NVlabs/FoundationPose/assets/23078192/aa341004-5a15-4293-b3da-000471fd74ed


AR Applications:

https://github.com/NVlabs/FoundationPose/assets/23078192/80e96855-a73c-4bee-bcef-7cba92df55ca


Results on YCB-Video dataset:

https://github.com/NVlabs/FoundationPose/assets/23078192/9b5bedde-755b-44ed-a973-45ec85a10bbe



# Bibtex
```bibtex
@InProceedings{foundationposewen2024,
author        = {Bowen Wen, Wei Yang, Jan Kautz, Stan Birchfield},
title         = {{FoundationPose}: Unified 6D Pose Estimation and Tracking of Novel Objects},
booktitle     = {CVPR},
year          = {2024},
}
```

If you find the model-free setup useful, please also consider cite:

```bibtex
@InProceedings{bundlesdfwen2023,
author        = {Bowen Wen and Jonathan Tremblay and Valts Blukis and Stephen Tyree and Thomas M\"{u}ller and Alex Evans and Dieter Fox and Jan Kautz and Stan Birchfield},
title         = {{BundleSDF}: {N}eural 6-{DoF} Tracking and {3D} Reconstruction of Unknown Objects},
booktitle     = {CVPR},
year          = {2023},
}
```

# Data prepare


1) Download all network weights from [here](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing) and put them under the folder `weights/`. For the refiner, you will need `2023-10-28-18-33-37`. For scorer, you will need `2024-01-11-20-02-45`.

1) [Download demo data](https://drive.google.com/drive/folders/1pRyFmxYXmAnpku7nGRioZaKrVJtIsroP?usp=sharing) and extract them under the folder `demo_data/`

1) [Optional] Download our large-scale training data: ["FoundationPose Dataset"](https://drive.google.com/drive/folders/1s4pB6p4ApfWMiMjmTXOFco8dHbNXikp-?usp=sharing)

1) [Optional] Download our preprocessed reference views [here](https://drive.google.com/drive/folders/1PXXCOJqHXwQTbwPwPbGDN9_vLVe0XpFS?usp=sharing) in order to run model-free few-shot version.

# Env setup option 1: docker (recommended)
  > **Note**
  > This is only for a Nvidia RTX 40xx user!
  ```
  docker pull shingarey/foundationpose_custom_cuda121:latest
  docker tag shingarey/foundationpose_custom_cuda121 foundationpose
  bash docker/run_container.sh
  ```


If it's the first time you launch the container, you need to build extensions. Run this command *inside* the Docker container.
```
bash build_all.sh
```

### FoundationPose TCP Server
If all requirements are prepared, you can run 'FoundationPose TCP server' by running shell-script directly!
```
./run_foundationpose.sh
```

Then, you can use FoundationPose in any python environment!
```python
from FoundationPoseTCP import FoundationPoseTCPServer

client = FoundationPoseTCPClient(server_ip='localhost', port=9999)
client.connect()

# Initialize FoundationPose module first!
color, depth, K, mask = <From your own awesome camera module>
mesh_file = '<MESH FILE PATH>'
data = {
    'action': 'initialize'
    'color': color, # (H,W,3) np.ndarray
    'depth': depth, # (H,W) np.ndarray
    'mask': mask,   # (H,W) np.ndarray
    'K': K,         # (3, 3) np.ndarray, camera intrinsic matrix
    'mesh_file': mesh_file, # absolute file path in string
}
response = client.send(data, verbose=True)
```
```python
while True:
    color, depth, K, mask = <From your own awesome camera module>
    data = {
        'action': 'inference'
        'color': color,
        'depth': depth,
        'mask': None,
        'K': K,
    }
    response = client.send(data, verbose=False) # dict_keys: T_pose, vis
    
    vis = response['vis']
    cv2.imshow('1', vis[...,::-1])
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
```


# Env setup option 2: conda (experimental)

<details><summary> DEPRECATED! </summary>
  - Setup conda environment

  ```bash
  # create conda environment
  conda create -n foundationpose python=3.9
  
  # activate conda environment
  conda activate foundationpose
  
  # Install Eigen3 3.4.0 under conda environment
  conda install conda-forge::eigen=3.4.0
  export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:/eigen/path/under/conda"
  
  # install dependencies
  python -m pip install -r requirements.txt
  
  # Install NVDiffRast
  python -m pip install --quiet --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git
  
  # Kaolin (Optional, needed if running model-free setup)
  python -m pip install --quiet --no-cache-dir kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.0_cu121.html
  
  # PyTorch3D
  python -m pip install --quiet --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu121_pyt210/download.html
  
  # Build extensions
  CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh
  ```
</details>




# Run model-based demo
The paths have been set in argparse by default. If you need to change the scene, you can pass the args accordingly. By running on the demo data, you should be able to see the robot manipulating the mustard bottle. Pose estimation is conducted on the first frame, then it automatically switches to tracking mode for the rest of the video. The resulting visualizations will be saved to the `debug_dir` specified in the argparse. (Note the first time running could be slower due to online compilation)
```
python run_demo.py
```


<img src="assets/demo.jpg" width="50%">


Feel free to try on other objects (**no need to retrain**) such as driller, by changing the paths in argparse.

<img src="assets/demo_driller.jpg" width="50%">


# Troubleshooting


- For more recent GPU such as 4090, refer to [this](https://github.com/NVlabs/FoundationPose/issues/27).

- For setting up on Windows, refer to [this](https://github.com/NVlabs/FoundationPose/issues/148).

- If you are getting unreasonable results, check [this](https://github.com/NVlabs/FoundationPose/issues/44#issuecomment-2048141043) and [this](https://github.com/030422Lee/FoundationPose_manual)


# Notes
Due to the legal restrictions of Stable-Diffusion that is trained on LAION dataset, we are not able to release the diffusion-based texture augmented data, nor the pretrained weights using it. We thus release the version without training on diffusion-augmented data. Slight performance degradation is expected.

# Acknowledgement

We would like to thank Jeff Smith for helping with the code release; NVIDIA Isaac Sim and Omniverse team for the support on synthetic data generation; Tianshi Cao for the valuable discussions. Finally, we are also grateful for the positive feebacks and constructive suggestions brought up by reviewers and AC at CVPR.

<img src="assets/cvpr_review.png" width="100%">


# License
The code and data are released under the NVIDIA Source Code License. Copyright © 2024, NVIDIA Corporation. All rights reserved.

# Contact
For questions, please contact [Bowen Wen](https://wenbowen123.github.io/).
