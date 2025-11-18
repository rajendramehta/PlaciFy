import os
import time
from io import BytesIO
from typing import List, Optional
from pathlib import Path
import uuid

from dotenv import load_dotenv
from google import genai
from google.genai import types as gt
from google.genai.errors import ClientError
from PIL import Image

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse


# Config & client

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env")

client = genai.Client(api_key=API_KEY)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


# System instruction

IMAGE_SYSTEM_INSTRUCTION = """
You are a STRICT, RULE-BASED  and an EXPERT virtual interior lighting try-on engine.

Overall behavior:
- You are a STRICT, RULE-BASED virtual try-on engine for lights,lamps/ any type of lights.
- You prioritize physical realism, product category correctness, and scene consistency over creativity.
- You must not ignore or reinterpret these rules.

Inputs:
- Image 1: a REAL photograph of a room. This must remain the base scene.
- Images 2..N: product photos of lights / lamps / chandeliers. These are the objects to insert into the room.

Your job:
1. KEEP the original room structure, camera angle, framing, and furniture layout exactly as in Image 1.
2. INSERT the light products from the product photos into the room in realistic, physically possible positions.
3. You MUST NOT change the camera or composition of Image 1:
   - Do NOT crop, zoom in, zoom out, pan, rotate, or change the field of view.
   - The output image must show the room from the same viewpoint and framing as Image 1.
4. Only adjust LOCAL lighting where necessary so the new fixtures look naturally integrated
   (shadows, highlights, reflections near the lights), but do NOT globally relight or restyle the whole room.
5. DO NOT remove, redesign, or hide existing objects in the room (stairs, plants, windows, walls, floors, ceilings, furniture, etc.).


Before deciding final placement, internally consider:
- Where are existing focal points (sofa, bed, TV, table, stairs, plants, etc.)?
- Which walls or ceiling areas are free and suitable for mounting?
- How many lights will feel balanced without overcrowding the scene?

Non-negotiable camera & framing rules:
- You MUST KEEP the original room framing exactly as in Image 1.
- Do NOT crop, zoom in, zoom out, pan, rotate, or change the camera viewpoint.
- Do NOT change the aspect ratio of the room image.
- The output image must show the room from the SAME viewpoint and framing as Image 1, only with new lights added.

Category & mounting rules:
- You MUST preserve the lamp CATEGORY and MOUNT TYPE implied by each product image.
- FLOOR / STANDING LAMP:
  - Must remain a floor/standing lamp.
  - Must stand on the FLOOR, not on the ceiling or walls, and not floating.
- CEILING LIGHT / CHANDELIER:
  - Must remain a ceiling-mounted light attached to the CEILING surface.
- WALL LIGHT / SCONCE:
  - Must remain a wall-mounted light attached to a VERTICAL WALL.
- TABLE / DESK LAMP:
  - Must remain a table/desk lamp sitting on a HORIZONTAL SURFACE (table, desk, console, shelf).
- You are NOT allowed to convert one type into another (e.g. floor lamp → ceiling lamp).

Scene Preservation rules:
- Keep the room layout, structure, furniture, walls, floors, ceilings, windows, and stairs exactly as in Image 1.
- Do NOT remove, redesign, hide, or replace existing objects.
- Do NOT hallucinate new furniture, walls or architectural features.
- Do NOT invent extra lights or products that were not provided.

Placement rules:
- Place lights only in physically possible locations (no intersection through walls, furniture, or windows).
- Respect structural logic: ceiling lights on ceiling, wall lights on walls, floor lamps on floor, etc.
- If multiple lights are provided, arrange them in a balanced way without overcrowding the scene.

Lighting rules:
- Match lighting, shadows, and reflections locally around the new fixtures so they look integrated.
- Do NOT globally restyle or relight the entire room; preserve the original mood and color palette as much as possible.
""".strip()


# -----------------------------
# 2. Core generation functions
# -----------------------------
def generate_room_with_light(
    room_path: Path,
    light_paths: List[Path],
    out_path: Path,
    placement_prompt: Optional[str] = None,
):
    """
    Use gemini-2.5-flash-image to generate a composite image:
    user's room + chosen light(s).
    Returns: (composite_img, gemini_image) where gemini_image is used for Veo.
    """
    if not room_path.exists():
        raise FileNotFoundError(f"Room image not found: {room_path}")

    if not light_paths:
        raise ValueError("Need at least one light image path")

    room_img = Image.open(room_path).convert("RGB")
    light_images = [Image.open(p).convert("RGB") for p in light_paths]

    base_prompt = """
    Install these light products into this room in the most realistic way.

    Constraints:
    - Keep the exact same framing and camera view as the room photo.
    - Do NOT crop or zoom; show the room exactly as in the original image.
    - Use the product image(s) only as the design of the fixtures.
    - Place each light according to its real-world mounting type
      (floor lamp on the floor, ceiling lamp on the ceiling, wall lamp on the wall, etc.).
    """.strip()

    if placement_prompt:
        base_prompt += f"\n\nUser placement instruction:\n{placement_prompt}"

    config = gt.GenerateContentConfig(
        system_instruction=IMAGE_SYSTEM_INSTRUCTION,
        response_modalities=["IMAGE"],
        image_config=gt.ImageConfig(image_size="1K"),
        temperature=0.15,
        candidate_count=1,
    )

    contents = [base_prompt, room_img, *light_images]

    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=contents,
        config=config,
    )

    composite_img = None
    gemini_image = None

    # New google-genai responses usually have "candidates", but also expose .parts
    parts = getattr(response, "parts", None)
    if parts is None and getattr(response, "candidates", None):
        parts = response.candidates[0].content.parts

    if not parts:
        raise RuntimeError("No image parts returned from gemini-2.5-flash-image")

    for part in parts:
        if getattr(part, "inline_data", None) is not None:
            raw_bytes = part.inline_data.data
            composite_img = Image.open(BytesIO(raw_bytes)).convert("RGB")
            gemini_image = part.as_image()
            break

    if composite_img is None or gemini_image is None:
        raise RuntimeError("No image returned from gemini-2.5-flash-image")

    composite_img.save(out_path)
    print(f"[IMAGE] Saved composite frame to: {out_path}")
    return composite_img, gemini_image


def generate_360_video_from_image(
    gemini_image,
    out_path: Path,
):
    """
    Use Veo 3.1 via Gemini API to generate a short 360-style room video
    starting from the composite image as the first frame.
    """
    video_prompt = """
    A smooth, slow 8-second 360-degree camera move inside this exact room interior.

    The video should:
    - Start from the same perspective as the input frame.
    - Slowly orbit in a subtle way, revealing more of the room specifically the lights installed[composited].
    - Keep the installed light fixtures clearly visible and fixed in space across the move.
    - Do NOT change the design,category or mounting type of the lights.
    - Do NOT change the furniture, walls, or layout.
    - Maintain consistent lighting,shadows and reflections based on the input frame.
    - Use a stable, cinematic camera path (no wild shaking, no cuts).
    - 16:9 aspect ratio, 720p or 1080p if possible.
    """.strip()

    operation = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        prompt=video_prompt,
        image=gemini_image,
    )

    while not operation.done:
        print("[VIDEO] Waiting for video generation to complete...")
        time.sleep(10)
        operation = client.operations.get(operation)

    video = operation.response.generated_videos[0]

    # Download the MP4
    client.files.download(file=video.video)
    video.video.save(str(out_path))
    print(f"[VIDEO] Saved Veo video to: {out_path}")



# 3. FastAPI app

app = FastAPI(
    title="Room Lighting Video Virtual Try-On",
    description="Upload a room photo + lights, get back a 360° Veo video.",
    version="1.0.0",
)

# CORS so your HTML can call this from localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/v1/lighting-tryon")
async def lighting_tryon(
    room: UploadFile = File(..., description="Room photograph (JPG/PNG)"),
    lights: List[UploadFile] = File(..., description="One or more light product photos (JPG/PNG)"),
    placement_prompt: Optional[str] = Form(None),
):
    """
    1) Save uploads
    2) Gemini 2.5 → composite image
    3) Veo 3.1 → video
    4) Return URLs
    """
    if room.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="Room must be JPEG or PNG")

    if not lights:
        raise HTTPException(status_code=400, detail="At least one light image is required")

    for lf in lights:
        if lf.content_type not in ("image/jpeg", "image/png"):
            raise HTTPException(status_code=400, detail="All light files must be JPEG or PNG")

    job_id = uuid.uuid4().hex

    room_path = OUTPUT_DIR / f"{job_id}_room.png"
    light_paths: List[Path] = []
    composite_path = OUTPUT_DIR / f"{job_id}_room_with_light.png"
    video_path = OUTPUT_DIR / f"{job_id}_room_with_light_360.mp4"

    # Save room
    try:
        room_bytes = await room.read()
        with open(room_path, "wb") as f:
            f.write(room_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save room image: {e}")

    # Save lights
    try:
        for idx, lf in enumerate(lights):
            light_bytes = await lf.read()
            lp = OUTPUT_DIR / f"{job_id}_light_{idx + 1}.png"
            with open(lp, "wb") as f:
                f.write(light_bytes)
            light_paths.append(lp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save light image: {e}")

    # Step 1: composite
    try:
        _, gemini_image = generate_room_with_light(
            room_path=room_path,
            light_paths=light_paths,
            out_path=composite_path,
            placement_prompt=placement_prompt,
        )
    except ClientError as ce:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini API error while generating composite image: {ce}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error while generating composite image: {e}",
        )

    # Step 2: video
    try:
        generate_360_video_from_image(
            gemini_image=gemini_image,
            out_path=video_path,
        )
    except ClientError as ce:
        return JSONResponse(
            status_code=502,
            content={
                "job_id": job_id,
                "status": "partial_success",
                "message": f"Composite generated but Veo video failed: {ce}",
                "composite_image_url": f"/results/{job_id}/image",
                "video_url": None,
            },
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "job_id": job_id,
                "status": "partial_success",
                "message": f"Composite generated but video error: {e}",
                "composite_image_url": f"/results/{job_id}/image",
                "video_url": None,
            },
        )

    return {
        "job_id": job_id,
        "status": "success",
        "composite_image_url": f"/results/{job_id}/image",
        "video_url": f"/results/{job_id}/video",
    }


@app.get("/results/{job_id}/image")
def get_composite_image(job_id: str):
    path = OUTPUT_DIR / f"{job_id}_room_with_light.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Composite image not found")
    return FileResponse(path, media_type="image/png", filename="room_with_light.png")


@app.get("/results/{job_id}/video")
def get_video(job_id: str):
    path = OUTPUT_DIR / f"{job_id}_room_with_light_360.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path, media_type="video/mp4", filename="room_with_light_360.mp4")
