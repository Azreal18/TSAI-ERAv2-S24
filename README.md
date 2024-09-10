# TSAI-ERAv2-S24 - Stable diffusion textual inversion demo

## Overview
This project is designed to generate images using stable diffusion techniques. It leverages various machine learning libraries such as PyTorch, Transformers, and Diffusers.

## Objective

1. select 5 different styles from "community-created SD concepts library" and show output for the same prompt using these 5 different styles.
2. implement a variant of additional guidance loss and generate images using the same prompts used above to show differences. An example of such loss is `blue_loss` - when applied the generated images will be saturated with blue colour.
3. Convert this to HuggingFace Spaces app.


## Project Structure

- **app.py**: Main application script.
- **notebooks/**: Directory containing Jupyter notebooks for experiments and analysis.
- **requirements.txt**: List of dependencies required for the project.
- **Stable_difusion.py**: Script containing the core logic for image generation using stable diffusion.
- **utils/**: Utility functions and scripts.


## Dependencies
The project requires the following Python packages, as listed in [`requirements.txt`]:

- scipy
- numpy
- pillow
- torch
- transformers==4.25.1
- diffusers
- huggingface_hub
- accelerate
- open_clip_torch

To install the dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

The application is Deployed on huggingface and can be accesed here :- [Demo](https://huggingface.co/spaces/Azreal18/Stable_Diffusion-Textual_Inversion)
![image](https://github.com/Azreal18/TSAI-ERAv2-S24/blob/main/assets/Gradio_app.png)


Also shared This on my LinkedIN : [Linkedin](https://www.linkedin.com/posts/viraj-bhanushali-3339201ab_machinelearning-deeplearning-stablediffusion-activity-7237370703992242177-gTL2?utm_source=share&utm_medium=member_desktop)