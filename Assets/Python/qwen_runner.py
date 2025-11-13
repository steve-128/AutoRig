#!/usr/bin/env python3
# qwen_runner.py
# Image -> VLM text -> Qwen image edit -> FLUX refine
# intermediates go to Working folder, final output to Output

import os, sys, json, argparse, base64, mimetypes, time, traceback
from pathlib import Path

def ensure_min_resolution(image_path: Path, min_size: int = 512) -> Path:
    """
    Check if image meets minimum resolution requirements.
    If not, upscale it and save to Working folder.
    Returns path to the (possibly upscaled) image.
    """
    img = Image.open(image_path)
    width, height = img.size
    
    # if image is already big enough, just return original
    if width >= min_size and height >= min_size:
        return image_path
    
    # scale factor
    scale = max(min_size / width, min_size / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    print(f"image too small ({width}x{height}), upscaling to {new_width}x{new_height}")
    
    upscaled = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # save to working folder
    working_path = image_path.parent.parent / "Working" / "upscaled_input.png"
    upscaled.save(working_path)
    
    return working_path

HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_KEY = os.getenv("GROQ_API_KEY")
FAL_KEY  = os.getenv("FAL_KEY")


from huggingface_hub import InferenceClient
from PIL import Image
import io


def ensure_dirs(root: Path):
    (root / "Input").mkdir(parents=True, exist_ok=True)
    (root / "Working").mkdir(parents=True, exist_ok=True)
    (root / "Output").mkdir(parents=True, exist_ok=True)

def write_status(root: Path, stage: str, progress: float, msg: str = ""):
    data = {"stage": stage, "progress": float(progress), "msg": msg}
    (root / "Working" / "status.json").write_text(json.dumps(data))

def write_log(root: Path, text: str):
    with open(root / "Working" / "runner.log", "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")

def die(root: Path, message: str, code: int = 1):
    write_status(root, "error", 0.0, message)
    write_log(root, "[ERROR] " + message)
    print("[ERROR]", message, file=sys.stderr)
    sys.exit(code)

def load_image_bytes(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def to_data_url(image_path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(image_path))
    if not mime or not mime.startswith("image/"):
        raise ValueError(f"Could not determine image MIME for {image_path}")
    b64 = base64.b64encode(load_image_bytes(image_path)).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def save_pil_to(path: Path, pil_image: Image.Image):
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(path)

def bytes_to_pil(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGBA")


# VLM prompt for creature identification
VLM_PROMPT = (
    "Identify the creature in the drawing and describe its prominent features in a single, "
    "detailed but concise sentence with less than 20 words. "
    "If the drawing is stick-like, avoid identifying the creature or describing its features using any form of the word stick. "
    "Output the response in this exact format: (creature name) | (prominent features). "
    "Start the feature sentence directly with the first descriptive feature, omitting any introductory phrases."
)

# Qwen/FLUX prompt template
QWEN_PROMPT = (
   "Based on the drawing of a {creature} with {features}, "
   "isolate it and remove all background clutter while preserving ALL visual elements of the creature including fire, flames, wings, tails, and any other distinctive features. "
   "Extrapolate the full body and limbs to achieve a precise T-pose: "
   "arms extended horizontally perpendicular to the body, legs straight and together. "
   "Apply intricate, tactile texture and color across its entire body surface, maintaining all original features."
   "Integrate textures with dynamic lighting and volumetric shading. "
   "The final image must be a front-facing, full-body, hyper-detailed, photorealistic illustration entirely within frame. "
   "Output the final image with a pure white, transparent PNG background."
)

FLUX_PROMPT = QWEN_PROMPT


def step_vlm_caption(root: Path, image_path: Path) -> str:
    # run Llama VLM via HuggingFace + Groq
    write_status(root, "vlm", 0.1, "captioning")
    write_log(root, "VLM: starting")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        die(root, "Missing HF_TOKEN environment variable")

    client = InferenceClient(provider="groq", api_key=hf_token)

    data_url = to_data_url(image_path)
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": VLM_PROMPT},
            {"type": "image_url", "image_url": {"url": data_url}}
        ],
    }]

    model_name = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        # extract text content from response
        text = None
        if hasattr(completion, "choices"):
            text = completion.choices[0].message.content
        else:
            text = str(completion)
    except Exception as e:
        die(root, f"VLM request failed: {e}")

    (root / "Working" / "vlm_caption.json").write_text(json.dumps({"caption": text}))
    write_log(root, f"VLM caption: {text}")
    write_status(root, "vlm", 0.35, "caption ok")
    return text


def step_parse_caption(root: Path, caption: str):
    # parse "Creature | features" format
    write_status(root, "parse", 0.4, "parsing caption")
    parts = [p.strip() for p in caption.split("|", 1)]
    if len(parts) != 2:
        # try alternate delimiters
        if ":" in caption:
            parts = [p.strip() for p in caption.split(":", 1)]
        elif "-" in caption:
            parts = [p.strip() for p in caption.split("-", 1)]
        else:
            parts = [caption.strip(), ""]
    creature, features = parts[0], parts[1]
    (root / "Working" / "parsed.json").write_text(
        json.dumps({"creature": creature, "features": features})
    )
    write_log(root, f"Parsed -> creature='{creature}' features='{features}'")
    return creature, features


def step_qwen_edit(root: Path, image_path: Path, creature: str, features: str) -> Image.Image:
    # Qwen image edit pass
    write_status(root, "qwen", 0.55, "qwen image edit")
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        die(root, "Missing HF_TOKEN environment variable")

    client = InferenceClient(provider="fal-ai", api_key=hf_token)
    prompt = QWEN_PROMPT.format(creature=creature, features=features)

    try:
        with open(image_path, "rb") as f:
            inp = f.read()
        
        out = client.image_to_image(
            inp,
            prompt=prompt,
            negative_prompt=(
                "text, numbers, watermark, logo, signature, low quality, "
                "extra elements, noisy, distorted, poorly rendered"
            ),
            model="Qwen/Qwen-Image-Edit",
            seed=12345,                   # deterministic output
            guidance_scale=9,             # stronger adherence to prompt
            strength=0.92,                # how much it overwrites input pose
        )


        # handle different return types from provider
        if isinstance(out, Image.Image):
            img = out
        elif isinstance(out, (bytes, bytearray)):
            img = bytes_to_pil(out)
        else:
            img = out.image if hasattr(out, "image") else None
            if img is None:
                raise RuntimeError("Unknown return type from Qwen image_to_image")
    except Exception as e:
        die(root, f"Qwen step failed: {e}")

    save_pil_to(root / "Working" / "qwen.png", img)
    write_log(root, "Qwen output saved to Working/qwen.png")
    write_status(root, "qwen", 0.75, "qwen ok")
    return img


def step_flux_refine(root: Path, qwen_img: Image.Image, creature: str, features: str) -> Image.Image:
    # FLUX refinement pass
    write_status(root, "flux", 0.8, "flux refine")
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        die(root, "Missing HF_TOKEN environment variable")

    client = InferenceClient(provider="fal-ai", api_key=hf_token)
    prompt = FLUX_PROMPT.format(creature=creature, features=features)

    # convert PIL to bytes for API input
    buf = io.BytesIO()
    qwen_img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    try:
        out = client.image_to_image(
            png_bytes,
            prompt=prompt,
            negative_prompt=(
                "text, numbers, watermark, logo, signature, low quality, "
                "extra elements, noisy, distorted, poorly rendered"
            ),
            model="black-forest-labs/FLUX.1-Kontext-dev",
            seed=12345,                   # match Qwen seed
            guidance_scale=10,            # follow T-pose instruction aggressively
            strength=0.98,                
            num_inference_steps=40        
        )
        if isinstance(out, Image.Image):
            img = out
        elif isinstance(out, (bytes, bytearray)):
            img = bytes_to_pil(out)
        else:
            img = out.image if hasattr(out, "image") else None
            if img is None:
                raise RuntimeError("Unknown return type from FLUX image_to_image")
    except Exception as e:
        die(root, f"FLUX step failed: {e}")

    save_pil_to(root / "Working" / "flux.png", img)
    write_log(root, "FLUX output saved to Working/flux.png")
    write_status(root, "flux", 0.95, "flux ok")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Job root folder (…/Assets/GenAI/<job>)")
    ap.add_argument("--image", required=True, help="Path to input drawing (…/Input/drawing.png)")
    ap.add_argument("--job", required=True, help="Short job name (used for filenames)")
    args = ap.parse_args()

    root = Path(args.root)
    image_path = Path(args.image)
    job = args.job

    try:
        ensure_dirs(root)
        write_log(root, "Runner started")
        write_status(root, "start", 0.05, "preparing")

        if not image_path.exists():
            die(root, f"Input image not found: {image_path}")
        
        image_path = ensure_min_resolution(image_path, min_size=512)

        # pipeline execution
        caption = step_vlm_caption(root, image_path)
        creature, features = step_parse_caption(root, caption)
        qwen_img = step_qwen_edit(root, image_path, creature, features)
        flux_img = step_flux_refine(root, qwen_img, creature, features)

        # save final outputs
        out_png = root / "Output" / f"{job}_refined.png"
        save_pil_to(out_png, flux_img)

        # TODO: add 3D reconstructor, output glb
        # (root / "Output" / f"{job}_model.glb").write_bytes(...)

        # Unity import metadata
        meta = {"scale": 1.0, "pivot": [0, 0, 0], "upAxis": "Y", "rig": "none"}
        (root / "Output" / "meta.json").write_text(json.dumps(meta))

        write_status(root, "done", 1.0, "ok")
        write_log(root, "Runner finished successfully")
        print("OK")
        sys.exit(0)
        
    except Exception as e:
        tb = traceback.format_exc()
        die(root, f"Unhandled exception: {e}\n{tb}", code=2)


if __name__ == "__main__":
    main()