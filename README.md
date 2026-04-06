# Quantum-Enhanced-Deep-Learning-Cryptanalysis-on-Round-Reduced-Speck32-64

This repository provides the official implementation for the paper: [Quantum-Enhanced Deep Learning Cryptanalysis on Round-Reduced Speck32/64].

**##1. Requirements**
   - Python: 3.12
   - Quantum Framework: PennyLane==0.40.0
   - Deep Learning: PyTorch

**##2. Model Training**: train folder
   - quanvh.py: Training QuavnH.
   - classical_conv.py: Training Conv.
   - classical_resnet.py: Training ResNet.
  
**##3. Evaluation**: key_rank & bayesian folder
   - quanvh_keyrank.py: Rank all last round subkeys by using a trained QuanvH.
   - conv_keyrank.py: Rank all last round subkeys by using a trained Conv.
   - resnet_keyrank.py: Rank all last round subkeys by using a trained ResNet.
