# Quantum-Enhanced-Deep-Learning-Cryptanalysis-on-Round-Reduced-Speck32-64

This repository provides the official implementation for the paper: [Quantum-Enhanced Deep Learning Cryptanalysis on Round-Reduced Speck32/64].

## **1. Requirements**
   - Python: 3.12
   - Quantum Framework: PennyLane==0.40.0
   - Deep Learning: PyTorch

## **2. Model Training**: train folder
   - quanvh.py: Training QuavnH.
   - classical_conv.py: Training Conv.
   - classical_resnet.py: Training ResNet.

## **3. Wrong-key response**: wkr folder
   - [model_name]_wkr.py: Get data of mu and sigma for wrong-key response.
  
## **4. Evaluation**: key_rank & bayesian folder
   - [model_name]_keyrank.py: Rank all last round subkeys by using a trained model.
   - [model_name]_bayesian_singleround.py: Recover the last round key by using one neural distinguisher without any round extension.
   - [model_name]_bayesian: Recover 10-round subkey by using two neural distinguishers based on the Gohr's Bayesian key search technique.
