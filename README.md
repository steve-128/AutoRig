# Netflix AutoRig – From 2D Drawing to Animated Character in Unity

AutoRig is a Unity + Python pipeline that turns a **hand-drawn 2D character** into a **rigged, animated character** directly inside Unity.  

You give it:
- A drawing (front view, 2 arms, 2 legs)
- A Hugging Face token  
It gives you:
- A rigged character prefab
- Ready-to-play animations you can preview in the Unity editor

DeepWiki with in-depth explanations: https://deepwiki.com/steve-128/AutoRig
Deck: https://docs.google.com/presentation/d/1oOz153M77NEF_UTOOMR24zbMWx1N-OxOtayr23uYdKg/edit?slide=id.g355984a9e45_0_1781#slide=id.g355984a9e45_0_1781 

---

## ✨ Features

- **Custom drawing input** – Bring your own 2D character art.
- **Automatic rigging** – Python pipeline detects keypoints and generates skeleton data & weights.
- **Unity animation playback** – View and play available animations on the generated character in the Scene.
- **Multi-character support** – Run the pipeline again to add additional characters to the same scene.

> Current pipeline is designed for **humanoid-like characters** with **two arms and two legs**.

---
# Python Environment Setup

This project uses a dedicated Conda environment called `netflix_f25` to manage Python dependencies.

## 1. Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/)
- Python compatible with PyTorch (e.g. 3.10)

## 2. Create and Activate the Conda Environment

Open a terminal (or Anaconda Prompt on Windows) and run:

```bash
# Create environment (you can change the Python version if needed)
conda create -n netflix_f25 python=3.10 -y

# Activate the environment
conda activate netflix_f25
```

## 3. Install dependency
```bash
pip install torch torchvision opencv-python pyyaml numpy
pip install ultralytics
pip install huggingface_hub
pip install imageio
pip install scipy
pip install shapely
pip install scikit-image
pip install pygltflib
```

## 4. Download Pretrained Pose Estimation Weights
1. Open this Google Drive folder in your browser: https://drive.google.com/drive/folders/1fGLeCgTbaO50wylfV_j1OFTEx8DDpaqh
2. Download the file: pose_resnet_50_256x192.pth.tar
3. Create a folder called pretrained_weights in Assets/Scripts
4. Save the file in the Assets/Scripts/pretrained_weights folder 

# Unity Setup Guide

This guide explains how to properly open the project in Unity and run the character generation pipeline.

---

## Requirements

- Unity Hub installed
- Unity Editor **6000.2.5f1**
- Hugging Face account and access token
- This repository cloned locally: https://github.com/steve-128/AutoRig/

---

## Unity Project Setup

1. Open **Unity Hub**  
2. Click **Add project from disk**  
3. Select the **`AutoRig`** folder (root of this repository)
4. Set the **Editor Version** to: 6000.2.5f1
5. **Launch** the project

---

## Running the Netflix Pipeline

6. At the **top Unity menu bar**, open: GenAI@berkeley → Netflix Pipeline
7. When the pipeline window opens:
   - Set **Pipeline Asset** to: GenAI_PipelineAsset
   
8. Fill in the pipeline input fields:

- **Image Path**  
  Select the input image to generate the character from

- **Job Name**  
  Enter a folder name for this run  
  - All input + output files will be saved to: Assets/GenAI/<JobName>/

- **HF_TOKEN**  
  Paste your **Hugging Face Access Token**

---

## Run the Pipeline

9. Click **Run Pipeline**  
   - Wait for the process to finish  
   - When complete, the generated **character will appear in the Unity scene**

---

## Play Animations

10. Select the generated **character** in the **Hierarchy**
- View available **Animation modes** and set **current_mode** to desired animation
- Click **Play** to preview animations

---

## Importing Additional Characters

To insert a **new character**:

- Run the **same pipeline again** with a new image and/or job name
- A **new character will be added to the scene**
- ⚠️ The previous character will **NOT be removed automatically**

You must manually delete or disable old characters if needed.

---

