# Effective Heterogeneous Federated Learning via Efficient Hypernetwork-based Weight Generation

These codes demonstrate a proof-of-concept for a novel federated learning technique, HypeMeFed. The technique is based on the idea of using a hypernetwork to generate the weights of a neural network. The hypernetwork is trained in a centralized manner, while the neural network is trained in a federated manner. The hypernetwork is trained to generate weights that are effective for the federated learning task. The technique is evaluated on three datasets: UniMiB, STL-10, and SVHN.

You can find the code for the paper [here](https://github.com/eis-lab/HypeMeFed) or you can download the code from [here](https://www.dropbox.com/scl/fi/t5iwhm3foizk7v2dnw68u/HypeMeFed.zip?rlkey=zxtlfz3h0n5ozei40xrcs9sx4&dl=0).

If you have any questions, please feel free to contact us at [yujin_shin@yonsei.ac.kr](mailto:Yujin).

## Requirements if not using Docker
- PyTorch: 2.2.2+cu121
- Python: 3.8.10
- scikit-learn: 1.1.1
- torchvision: 0.17.2+cu121
- numpy: 1.22.0

## Installaion Instructions

### 1. Clone the repository
```bash
git clone https://github.com/eis-lab/HypeMeFed.git
cd HypeMeFed
```

### 2. Build and run the docker image.
```bash
docker build -t hypemefed .
docker run -it hypemefed
```

## Execution Instructions
Once the Docker container is running, you can execute the project using the following command format:

```bash
python3 main.py --dataset=[unimib | stl10 | svhn] --iid=[float] --gpu=[int] --use_hn=[str:True|False]
```

### Command Line Arguments
- `--dataset`: The dataset to use. Options are `unimib`, `stl10`, and `svhn`.
- `--iid`: The percentage of IID data to use. Must be a float between 0 and 1.
- `--gpu`: The GPU to use. Must be an integer.
- `--use_hn`: Whether to use the hypernetwork. Options are `True` and `False`. Default is `True`. Due to the information disparity, when `--use_hn=False`, the model will not use the hypernetwork and will not converge, especially on small datasets such as `unimib`.

### Example
```bash
python3 main.py --dataset=unimib --iid=0.1 --gpu=0
```

### Notes
- Ensure that the dataset is placed in the correct directory (i.e., datasets/), or that the volume is mounted correctly.
- The project will utilize the GPU specified via the --gpu argument, so make sure your environment supports GPU execution (NVIDIA drivers and CUDA should be properly configured).
- The project will download `SVHN`, `STL10` automatically. You can download `UniMiB` from [here](https://www.dropbox.com/scl/fi/0as6cugy53govyzwx99bp/Unimib_SHAR.zip?rlkey=8rvqhyrai5x28wdhyn8ivp9sd&dl=0) and unzip it in the `datasets/UniMiB_SHAR` directory.

### Results
The results of the experiments will print to the console. The results will include the accuracy of the model on the test set over the federated learning rounds.

## Additional Information
### Expected Execution Time for Each Dataset
We provide an estimate of the expected execution time for each dataset. The times are approximate and may vary depending on the hardware and environment. The times are based on the following environment: `Intel i9-10900K` + `NVIDIA RTX 3090` + `64GB RAM`.
- `unimib`: 1~2 hours
- `stl10`: 4~5 hours
- `svhn`: 10~12 hours
