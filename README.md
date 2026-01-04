# Z-Videogen - Wan 2.2 5B – Local WebUI (Text to Video & Image to Video)

A lightweight local WebUI to run **Wan 2.2 5B** entirely on your own machine or inside a local network.
The interface supports both **Text to Video** and **Image to Video** generation from a single page, with a simple Flask backend and no external services required.

The project is designed for experimentation, local workflows, and small servers, with a focus on clarity, stability, and controllable motion.

---

## Features

* Text to Video and Image to Video in one interface
* Local execution on a single GPU
* Adjustable resolution, frame count, steps, guidance, seed, and motion
* Scheduler selection
* Optional CPU offload for low VRAM scenarios
* Concatenate mode to generate multiple clips in sequence and export a single video
* Automatic gallery preview with generation metadata
* No external tools required for final video export

---

## Resolution and frame constraints

Wan 2.2 requires specific resolutions for stable video generation.

Recommended sizes:

* 1280 × 704 for horizontal video
* 704 × 1280 for vertical video

Frame count is based on 24 frames per second.
For example:

* 72 frames ≈ 3 seconds
* 121 frames ≈ 5 seconds
* 168 frames ≈ 7 seconds

The application automatically normalizes frame counts when required by the model.

---

## Requirements

* Linux system
* Python 3.10 or newer
* A working CUDA setup with PyTorch
* Diffusers with support for Wan 2.2
* Transformers, Accelerate, Pillow, Flask

A GPU with at least 24 GB of VRAM is recommended for 720p video generation.

---

## Installation

Create and activate a virtual environment:

```bash
python -m venv env
source env/bin/activate
pip install -U pip wheel
```

Verify your existing PyTorch and CUDA setup:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Install required libraries:

```bash
pip install transformers accelerate pillow flask
```

If your installed Diffusers version does not support Wan 2.2, install a recent version from GitHub:

```bash
pip install "git+https://github.com/huggingface/diffusers"
```

---

## Model download

Download the model once using the Hugging Face CLI:

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-5B-Diffusers --local-dir /opt/Wan2.2-5B-Diffusers
```

You can change the model directory path in the application configuration if needed.

---

## Running the application

Start the server with:

```bash
python app_wan22.py
```

The application supports HTTPS using locally generated certificates.

* If valid certificate files are found at startup, the server runs in HTTPS mode
* If no certificates are present, the server automatically falls back to HTTP

This makes it suitable for both secure local access and quick test environments.

---

## Parameter overview

* **Width and Height**
  Control video resolution. Higher values increase detail but require more VRAM and time.

* **Frames**
  Define video length at 24 frames per second.

* **Steps**
  Control refinement quality. Higher values improve clarity but increase generation time.

* **Guidance**
  Controls how strictly the model follows the prompt. Moderate values usually give the best motion.

* **Seed**
  Use a fixed value for reproducible results or random for exploration.

* **Strength**
  Mainly used in Image to Video. Lower values preserve the input image, higher values allow stronger changes.

* **Motion bucket**
  Controls overall motion intensity. Lower values are calmer, higher values add dynamic movement.

* **Scheduler**
  Affects stability and quality. Some schedulers are faster, others more precise.

---

## Concatenate mode

Concatenate mode allows generating longer sequences by chaining multiple clips.

* The first clip can start from text or an image
* Each subsequent clip uses the last frame of the previous clip as its starting image
* The final output is exported as a single video file

This provides visual continuity without requiring external video tools.

---

## LoRA support

LoRA styles can be added by placing them in the designated LoRA folder.
Lower scale values apply subtle style changes, while higher values push the style more aggressively.

For video generation, lower LoRA strength is generally recommended.

---

## Notes for 24 GB GPUs

* Start with 1280 × 704 resolution
* Use around 121 frames for testing
* Enable CPU offload if you encounter memory issues
* Reduce resolution or frame count before reducing steps

---

## License

* Model license: Apache 2.0, according to the model card
* This WebUI: MIT License

When you reuse or fork this repository you must:

Keep the credit “By ZetaLvX” in both code banner and any UI.
Preserve the LICENSE and NOTICE files, and this section.
Comply with the upstream licenses listed above.

