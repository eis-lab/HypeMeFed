# Effective Heterogeneous Federated Learning via Efficient Hypernetwork-based Weight Generation

These codes demonstrate a proof-of-concept for a novel federated learning technique, HypeMeFed. The technique is based on the idea of using a hypernetwork to generate the weights of a neural network. The hypernetwork is trained in a centralized manner, while the neural network is trained in a federated manner. The hypernetwork is trained to generate weights that are effective for the federated learning task. The technique is evaluated on a synthetic dataset and a real-world dataset.

## Requirements
- PyTorch: 2.2.2+cu121
- Python: 3.8.10
- sklearn: 1.1.1
- torchvision: 0.17.2+cu121
- numpy: 1.22.0
