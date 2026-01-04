#【Z】【v】【i】【d】【e】【o】【G】【e】【n】
# Z-VideoGen - By ZetaLvX
# ⚠️  DO NOT REMOVE "By ZetaLvX" ⚠️
# Keep the banner when reusing Z-VideoGen.


import gc, time, random, datetime, inspect
from ftfy import fix_text
from pathlib import Path
from flask import Flask, request, render_template, jsonify, send_from_directory
import torch

from diffusers import WanPipeline, WanImageToVideoPipeline, UniPCMultistepScheduler, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from io import BytesIO
from PIL import Image
import subprocess, shlex, math

# ------------------ Config ------------------
WAN22_DIR = "/path/to/Wan2.2-TI2V-5B-Diffusers"   # Your local locale
MODEL_ID_FALLBACK = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"     # repo HF (only fallback)
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

SAVE_DIR = Path("static/videos")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# LoRA directory
LORA_DIR =  Path("/loras") #Loras path
LORA_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

pipe_t2v = None
pipe_i2v = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _from_pretrained(model_id, cls, local_only=True):
    try:
        return cls.from_pretrained(
            model_id,
            torch_dtype=DTYPE,
            local_files_only=local_only,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        if local_only:
            print(f"[Wan2.2] Local load failed for {cls.__name__}, fallback to remote:", e)
            return _from_pretrained(model_id, cls, local_only=False)
        raise


def load_pipe(mode: str, enable_offload: bool = False):
    """
    Get la correcly pipeline:
      - T2V  -> WanPipeline
      - I2V  -> WanImageToVideoPipeline
    """
    global pipe_t2v, pipe_i2v
    model_id = WAN22_DIR if Path(WAN22_DIR).exists() else MODEL_ID_FALLBACK

    if mode == "i2v":
        if pipe_i2v is None:
            print(f"[Wan2.2] Loading I2V from: {model_id}")
            pipe_i2v = _from_pretrained(model_id, WanImageToVideoPipeline, local_only=True)
            pipe_i2v.scheduler = UniPCMultistepScheduler.from_config(pipe_i2v.scheduler.config)
            pipe_i2v.to(DEVICE)
            if enable_offload:
                try:
                    pipe_i2v.enable_model_cpu_offload()
                    print("[Wan2.2] I2V CPU offload enabled.")
                except Exception as e:
                    print("[Wan2.2] I2V CPU offload not available:", e)
            else:
                try:
                    pipe_i2v.enable_attention_slicing()
                except Exception:
                    pass
            try:
                print("[Wan2.2] I2V __call__ signature:", inspect.signature(pipe_i2v.__call__))
            except Exception:
                pass
        return pipe_i2v

    # default: T2V
    if pipe_t2v is None:
        print(f"[Wan2.2] Loading T2V from: {model_id}")
        pipe_t2v = _from_pretrained(model_id, WanPipeline, local_only=True)
        pipe_t2v.scheduler = UniPCMultistepScheduler.from_config(pipe_t2v.scheduler.config)
        pipe_t2v.to(DEVICE)
        if enable_offload:
            try:
                pipe_t2v.enable_model_cpu_offload()
                print("[Wan2.2] T2V CPU offload enabled.")
            except Exception as e:
                print("[Wan2.2] T2V CPU offload not available:", e)
        else:
            try:
                pipe_t2v.enable_attention_slicing()
            except Exception:
                pass
        try:
            print("[Wan2.2] T2V __call__ signature:", inspect.signature(pipe_t2v.__call__))
        except Exception:
            pass
    return pipe_t2v


def parse_size(s: str):
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except Exception:
        return 1280, 704


def _norm_frames(n: int) -> int:
    # helper: (n-1) multiplo di 4, minimo 5 frame
    if n < 5:
        return 5
    k = round((n - 1) / 4)
    return 1 + 4 * max(1, k)


def _apply_lora_if_any(pipe, lora_name: str, lora_scale: float):
    """
    Apply (Or remove) a LoRA .safetensors from the folder LORA_DIR.
    It Ignore is lora_name is 'none' or empy.
    """
    # remove previous adapters
    try:
        pipe.unload_lora_weights()
    except Exception:
        pass

    if not lora_name or lora_name.lower() in ("none", "default"):
        return pipe

    f = (LORA_DIR / lora_name)
    if not f.exists():
        raise FileNotFoundError(f"LoRA not found: {f}")

    try:
        if f.is_file():
            # Load specific file
            pipe.load_lora_weights(str(LORA_DIR), weight_name=f.name, adapter_name="use")
        else:
            # or a lora dedicated folder
            pipe.load_lora_weights(str(f), adapter_name="use")

        # Scale (is supported)
        try:
            pipe.set_adapters(["use"], adapter_weights=[float(lora_scale)])
        except Exception:
            pass
        print(f"[LoRA] Applied '{lora_name}' scale={lora_scale}")
    except Exception as e:
        raise RuntimeError(f"Failed to load LoRA '{lora_name}': {e}")
    return pipe


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/lora-list")
def lora_list():
    # list of .safetensors in the folder LORA_DIR (only first level)
    items = ["none"]
    try:
        for p in sorted(LORA_DIR.iterdir()):
            if p.is_file() and p.suffix.lower() == ".safetensors":
                items.append(p.name)
            elif p.is_dir() and any(x.suffix.lower() == ".safetensors" for x in p.iterdir()):
                items.append(p.name)  # to select the folder
    except Exception:
        pass
    return jsonify({"loras": items})



def _set_scheduler(pipe, sched_name: str):
    """
    set the scheduler on the pipeline.
    Support: 'UniPC' (default), 'DPM++2M Karras'.
    automatic Fallback to UniPC in case of error.
    """
    name = (sched_name or "UniPC").strip().lower()
    try:
        if name in ("dpm++2m karras", "dpmpp2m karras", "dpmpp-2m karras", "dpm++ 2m karras"):
            sch = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            try:
                sch.use_karras_sigmas = True
            except Exception:
                pass
            try:
                sch.algorithm_type = "dpmsolver++"
            except Exception:
                pass
            pipe.scheduler = sch
        else:
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    except Exception as e:
        print("[Scheduler] Fallback to UniPC due to:", e)
        try:
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        except Exception:
            pass
    return pipe

@app.route("/generate", methods=["POST"])
def generate():
    try:
        mode = (request.form.get("mode") or "t2v").strip()  # "t2v" o "i2v"
        prompt = (request.form.get("prompt") or "").strip()
        if not prompt:
            return jsonify({"error": "Prompt required"}), 400

        negative = (request.form.get("negative_prompt") or "").strip() or None
        width_str = (request.form.get("width") or "").strip()
        height_str = (request.form.get("height") or "").strip()
        if width_str.isdigit() and height_str.isdigit():
            width = int(width_str)
            height = int(height_str)
        else:
            width, height = parse_size(request.form.get("size", "1280x704"))
        frames = int(request.form.get("frames", "121"))
        steps = int(request.form.get("steps", "50"))
        guidance = float(request.form.get("cfg", "5.0"))
        seed = int(request.form.get("seed", "-1"))
        enable_offload = request.form.get("offload", "0") == "1"

        # Strength / Denoise
        try:
            strength = float(request.form.get("strength", "0.65"))
        except Exception:
            strength = 0.65

        # nuovi campi
        motion_bucket_id = request.form.get("motion_bucket_id", "").strip()
        lora_name = (request.form.get("lora") or "none").strip()
        try:
            lora_scale = float(request.form.get("lora_scale", "1.0"))
        except Exception:
            lora_scale = 1.0

        # text cleaning
        prompt = fix_text(prompt)
        negative = fix_text(negative) if negative else None

        # vincolo Wan: (frames - 1) % 4 == 0
        frames = _norm_frames(frames)

        # generator
        gen = torch.Generator(device=DEVICE)
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        gen.manual_seed(seed)

        # (re)load pipeline appropriata
        pipe = load_pipe(mode, enable_offload=enable_offload)

        # Scheduler per richiesta
        sched = (request.form.get("sched") or "UniPC").strip()
        pipe = _set_scheduler(pipe, sched)

        # Apply LoRA (if selected)
        try:
            pipe = _apply_lora_if_any(pipe, lora_name, lora_scale)
        except Exception as e:
            return jsonify({"error": f"{type(e).__name__}: {e}"}), 400

        # prepare kwargs in base alla firma della pipeline
        call_params = set(inspect.signature(pipe.__call__).parameters.keys())

        t0 = time.time()
        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative,
            height=height,
            width=width,
            num_frames=frames,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=gen,
        )

        # Strength/Denoise only if supported by pipeline
        if "strength" in call_params:
            kwargs["strength"] = strength
        elif "denoising_strength" in call_params:
            kwargs["denoising_strength"] = strength

        # motion_bucket_id only if supported
        if motion_bucket_id and "motion_bucket_id" in call_params:
            try:
                kwargs["motion_bucket_id"] = int(motion_bucket_id)
            except Exception:
                pass

        # I2V image is requested
        if mode == "i2v":
            image_file = request.files.get("image")
            if not (image_file and image_file.filename):
                return jsonify({"error": "Image required for Image→Video"}), 400
            data = image_file.read()
            init_img = Image.open(BytesIO(data)).convert("RGB")
            kwargs["image"] = init_img  # WanImageToVideoPipeline accetta 'image'

        # generate
        result = pipe(**kwargs)
        out_frames = result.frames[0]  # lista di PIL images

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"wan22_{mode}_seed{seed}_{ts}.mp4"
        out_path = SAVE_DIR / out_name
        export_to_video(out_frames, str(out_path), fps=24)

        # cleaning: scarica lora dopo la generazione per evitare side-effect
        try:
            if lora_name and lora_name.lower() not in ("none", "default"):
                pipe.unload_lora_weights()
        except Exception:
            pass

        elapsed = time.time() - t0
        extra = []
        if "motion_bucket_id" in call_params and motion_bucket_id:
            extra.append(f"mb={motion_bucket_id}")
        if lora_name and lora_name.lower() not in ("none", "default"):
            extra.append(f"lora={lora_name}x{lora_scale}")
        extras = (" | " + " ".join(extra)) if extra else ""

        meta = (
            f"{mode.upper()} | {width}x{height} | {frames}f@24fps | "
            f"steps={steps} cfg={guidance} seed={seed}{extras} | {elapsed:.1f}s"
        )
        return jsonify({"video": f"static/videos/{out_name}", "meta": meta})

    except Exception as e:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.route("/static/videos/<path:filename>")
def serve_video(filename):
    return send_from_directory(SAVE_DIR.as_posix(), filename)


# ------------------ Concatenation multi-clip (robusta: PIL/NumPy) ------------------
# ------------------ Concatenation multi-clip without ffmpeg (unisce via export_to_video) ------------------
from PIL import Image
import numpy as np

def _to_pil_image(x):
    """Converte x (PIL o np.ndarray) in PIL.Image RGB."""
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    arr = np.asarray(x)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    return img.convert("RGB")

def _to_pil_frames(result):
    """
    Restituisce SEMPRE una lista di PIL.Image dai vari formati dell'output:
    - result.frames[0]  -> lista PIL o np.ndarray (T,H,W,C)
    - result.videos[0]  -> np.ndarray (T,H,W,C)
    """
    frames = None
    if hasattr(result, "frames"):
        frames = result.frames[0]
        if isinstance(frames, list) and (len(frames) == 0 or isinstance(frames[0], Image.Image)):
            return [_to_pil_image(fr) for fr in frames]
        if isinstance(frames, np.ndarray):
            return [_to_pil_image(frames[i]) for i in range(frames.shape[0])]
        try:
            return [_to_pil_image(fr) for fr in frames]
        except Exception:
            pass
    if hasattr(result, "videos"):
        v = result.videos[0]  # (T,H,W,C)
        return [_to_pil_image(v[i]) for i in range(v.shape[0])]
    raise RuntimeError("Pipeline output missing 'frames' and 'videos'")

@app.route("/generate_concat", methods=["POST"])
def generate_concat():
    """
    Generate N clip concatenate in a **unic MP4** usando export_to_video
    (no ffmpeg per concat). from 2° segment it uses the last frame from previous segment (I2V).
    """
    try:
        form = request.form
        prompt_count = int(form.get("prompt_count", "0"))
        if prompt_count < 2:
            return jsonify({"error": "prompt_count deve essere >= 2"}), 400

        prompts = form.getlist("prompts")
        if len(prompts) != prompt_count:
            prompts = [(form.get(f"prompt_{i}", "") or "").strip() for i in range(prompt_count)]
        if any(not p.strip() for p in prompts):
            return jsonify({"error": "Tutti i prompt devono essere non vuoti"}), 400

        mode_first = (form.get("mode_first", "t2v") or "t2v").lower()
        negative   = (form.get("negative_prompt") or "").strip() or None

        width  = int(form.get("width", "1280"))
        height = int(form.get("height", "704"))
        frames = _norm_frames(int(form.get("frames", "121")))
        steps  = int(form.get("steps", "40"))
        cfg    = float(form.get("cfg", "6.5"))
        seed   = int(form.get("seed", "-1"))
        strength = float(form.get("strength", "0.6"))
        motion_bucket_id = int(form.get("motion_bucket_id", "80"))
        lora_name  = (form.get("lora") or "none").strip()
        lora_scale = float(form.get("lora_scale", "1.0"))
        sched_name = (form.get("sched") or "UniPC").strip()
        enable_offload = form.get("offload", "0") == "1"

        # generator comune
        gen = torch.Generator(device=DEVICE)
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        gen.manual_seed(seed)

        # pipeline e scheduler
        p_t2v = _set_scheduler(load_pipe("t2v", enable_offload), sched_name)
        p_i2v = _set_scheduler(load_pipe("i2v", enable_offload), sched_name)

        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        clip_paths = []          # salviamo le clip singole (utile per debug)
        all_frames = []          # qui accumuliamo TUTTI i frame per il video unico
        last_frame_pil = None

        # ---- 1ª clip ----
        if mode_first == "i2v":
            if "image" not in request.files or not request.files["image"].filename:
                return jsonify({"error": "Image required for first I2V clip"}), 400
            init_img = Image.open(request.files["image"].stream).convert("RGB")
            pipe = _apply_lora_if_any(p_i2v, lora_name, lora_scale)
            kwargs = dict(
                prompt=prompts[0], negative_prompt=negative, image=init_img,
                height=height, width=width, num_frames=frames, num_inference_steps=steps,
                guidance_scale=cfg, generator=gen
            )
            call_params = set(inspect.signature(pipe.__call__).parameters.keys())
            if "strength" in call_params: kwargs["strength"] = strength
            elif "denoising_strength" in call_params: kwargs["denoising_strength"] = strength
            if "motion_bucket_id" in call_params: kwargs["motion_bucket_id"] = motion_bucket_id
            result = pipe(**kwargs)
        else:
            pipe = _apply_lora_if_any(p_t2v, lora_name, lora_scale)
            kwargs = dict(
                prompt=prompts[0], negative_prompt=negative,
                height=height, width=width, num_frames=frames, num_inference_steps=steps,
                guidance_scale=cfg, generator=gen
            )
            call_params = set(inspect.signature(pipe.__call__).parameters.keys())
            if "motion_bucket_id" in call_params: kwargs["motion_bucket_id"] = motion_bucket_id
            result = pipe(**kwargs)

        frames_list = _to_pil_frames(result)
        last_frame_pil = _to_pil_image(frames_list[-1])
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out1 = SAVE_DIR / f"wan22_concat_1of{prompt_count}_{ts}.mp4"
        export_to_video(frames_list, str(out1), fps=24)   # clip singola (debug)
        clip_paths.append(str(out1))
        all_frames.extend(frames_list)                    # <-- accoda per video unico

        # ---- clip 2..N ----
        for idx in range(1, prompt_count):
            init_img = last_frame_pil.copy()
            pipe = _apply_lora_if_any(p_i2v, lora_name, lora_scale)

            kwargs = dict(
                prompt=prompts[idx], negative_prompt=negative, image=init_img,
                height=height, width=width, num_frames=frames, num_inference_steps=steps,
                guidance_scale=cfg, generator=gen
            )
            call_params = set(inspect.signature(pipe.__call__).parameters.keys())
            if "strength" in call_params: kwargs["strength"] = strength
            elif "denoising_strength" in call_params: kwargs["denoising_strength"] = strength
            if "motion_bucket_id" in call_params: kwargs["motion_bucket_id"] = motion_bucket_id

            result = pipe(**kwargs)
            frames_list = _to_pil_frames(result)
            last_frame_pil = _to_pil_image(frames_list[-1])

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_i = SAVE_DIR / f"wan22_concat_{idx+1}of{prompt_count}_{ts}.mp4"
            export_to_video(frames_list, str(out_i), fps=24)  # clip singola (debug)
            clip_paths.append(str(out_i))
            all_frames.extend(frames_list)                    # <-- accoda

        # ---- export finale unico (senza ffmpeg) ----
        final_path = SAVE_DIR / f"wan22_concat_final_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        export_to_video(all_frames, str(final_path), fps=24)

        # scarico eventuale LoRA
        try:
            if lora_name and lora_name.lower() not in ("none", "default"):
                for p in (p_t2v, p_i2v):
                    try: p.unload_lora_weights()
                    except Exception: pass
        except Exception:
            pass

        return jsonify({
            "ok": True,
            "final_video": f"/static/videos/{final_path.name}",
            "clips": [f"/static/videos/{Path(p).name}" for p in clip_paths],
            "meta": {
                "width": width, "height": height, "frames": frames, "steps": steps,
                "cfg": cfg, "seed": seed, "strength": strength,
                "motion_bucket_id": motion_bucket_id, "lora": lora_name,
                "lora_scale": lora_scale, "scheduler": sched_name, "offload": int(enable_offload)
            }
        })
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


if __name__ == "__main__":
    ssl_ctx = None
    cert = Path("cert.pem")
    key = Path("key.pem")
    if cert.exists() and key.exists():
        ssl_ctx = (str(cert), str(key))
    app.run(host="0.0.0.0", port=8443, debug=False, use_reloader=False, ssl_context=ssl_ctx)




