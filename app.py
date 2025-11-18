import os
import time
from io import BytesIO
from typing import List
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types as gt
from google.genai.errors import ClientError
from PIL import Image
import streamlit as st


# Config & client

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env")

client = genai.Client(api_key=API_KEY)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
COMPOSITE_IMAGE_PATH = OUTPUT_DIR / "room_with_light.png"
VIDEO_OUTPUT_PATH = OUTPUT_DIR / "room_with_light_360.mp4"



# SYSTEM INSTRUCTIONS

IMAGE_SYSTEM_INSTRUCTION = """
You are a STRICT, RULE-BASED virtual interior lighting try-on engine.

Inputs:
- Image 1: a REAL photograph of a room. This must remain the base scene.
- Images 2..N: product photos of lights / lamps / chandeliers. These are the objects to insert into the room.

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

Scene preservation rules:
- Keep the room layout, structure, furniture, walls, floors, ceilings, windows, and stairs exactly as in Image 1.
- Do NOT remove, redesign, hide, or replace existing objects.
- Do NOT hallucinate new furniture, walls, or architectural features.
- Do NOT invent extra lights or products that were not provided.

Placement rules:
- Place lights only in physically possible locations (no intersection through walls, furniture, or windows).
- Respect structural logic: ceiling lights on ceiling, wall lights on walls, floor lamps on floor, etc.
- If multiple lights are provided, arrange them in a balanced way without overcrowding the scene.

Lighting rules:
- Match lighting, shadows, and reflections locally around the new fixtures so they look integrated.
- Do NOT globally restyle or relight the entire room; preserve the original mood and color palette as much as possible.

If the user request conflicts with physical reality, follow these rules first and choose the most realistic interpretation.
""".strip()



# FUNCTIONS 

def generate_room_with_light(
    room_path: Path,
    light_paths: list[Path],
    out_path: Path,
) -> Image.Image:
    """
    Step 1: Use Gemini 2.5 Flash Image ("Nano Banana") to generate
    a composite image: user's room + chosen light(s).
    """

    if not room_path.exists():
        raise FileNotFoundError(f"Room image not found: {room_path}")

    if not light_paths:
        raise ValueError("Need at least one light image path")

    room_img = Image.open(room_path).convert("RGB")
    light_images = [Image.open(p).convert("RGB") for p in light_paths]

    # USER PROMPT – lightweight, all heavy rules live in system_instruction
    user_prompt = """
    Install these light products into this room in the most realistic way.

    Follow these constraints:
    - Keep the exact same framing and camera view as the room photo.
    - Do NOT crop or zoom; show the room exactly as in the original image.
    - Use the product image(s) only as the design of the fixtures.
    - Place each light according to its real-world mounting type
      (floor lamp on the floor, ceiling lamp on the ceiling, wall lamp on the wall, etc.).
    """.strip()

    # GenerateContentConfig with system_instruction
    config = gt.GenerateContentConfig(
        system_instruction=IMAGE_SYSTEM_INSTRUCTION,
        response_modalities=["IMAGE"],
        image_config=gt.ImageConfig(image_size="1K"),
        temperature=0.15,
        candidate_count=1,
    )

    contents = [user_prompt, room_img, *light_images]

    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=contents,
        config=config,
    )

    composite_img = None
    gemini_image = None

    for part in response.parts:
        if getattr(part, "inline_data", None) is not None:
            gemini_image =  part.as_image()
            raw_bytes = part.inline_data.data
            composite_img = Image.open(BytesIO(raw_bytes)).convert("RGB")
            break

    if composite_img is None or gemini_image is None:
        raise RuntimeError("No image returned from gemini-2.5-flash-image")

    composite_img.save(out_path)
    print(f"[IMAGE] Saved composite VTO frame to: {out_path}")
    return composite_img, gemini_image


def generate_360_video_from_image(
    gemini_image,
    out_path: Path,
):
    """
    Step 2: Use Veo 3.1 via Gemini API to generate a short '360-ish' room video
    starting from the composite image as the first frame.
    """

    # Veo 3.1 text+image → video
    video_prompt = """
    A smooth, slow 8-second 360-degree camera move inside this exact room interior.

    The video should:
    - Start from the same perspective as the input frame.
    - Slowly move / orbit in a subtle way, revealing more of the room lights, furniture and decore.
    - Keep the installed light fixtures clearly visible and fixed in spaces across the move.
    - Do NOT change the design, category, or mounting type of the lights.
    - Do NOT change the furniture, walls, or layout of the room.
    - Maintain consistent lighting, shadows, and reflections based on the input frame.
    - Use a stable, cinematic camera path (no wild shaking, no cuts).
    - 16:9 aspect ratio, 720p or 1080p if possible.
    """.strip()

    # create a google.genai.types.Image from file
    
    operation = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        prompt=video_prompt,
        image=gemini_image,  
    )

    # Long-running op 
    while not operation.done:
        print("[VIDEO] Waiting for video generation to complete...")
        time.sleep(10)
        operation = client.operations.get(operation)

    video = operation.response.generated_videos[0]
    client.files.download(file=video.video)
    video.video.save(str(out_path))
    print(f"[VIDEO] Saved Veo video to: {out_path}")



# STREAMLIT UI 

def main():
    st.set_page_config(
        page_title="Room Lighting Virtual Try-On",
        layout="wide",
    )

    st.title("Room Lighting Video Try-On ")
    st.write(
        "Upload a room photo and one or more light product photos. "
        "Wait for a video try-on tobe generated "
        
    )

    # Step 1 – Upload room photo
    st.subheader("Step 1 Upload Room Photo")
    room_file = st.file_uploader(
        "Room photograph (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        key="room_file",
    )

    room_preview = None
    if room_file is not None:
        try:
            room_preview = Image.open(room_file).convert("RGB")
            st.image(room_preview, caption="Room Image", use_column_width=True)
        except Exception as e:
            st.error(f"Failed to load room image: {e}")
            room_preview = None

    # Step 2 – Upload light product images
    st.subheader("Step 2 Upload Light Product Image(s)")
    light_files = st.file_uploader(
        "Light / lamp / chandelier product photos (one or more)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="light_files",
    )

    light_previews: List[Image.Image] = []
    if light_files:
        cols = st.columns(3)
        idx = 0
        for f in light_files:
            try:
                img = Image.open(f).convert("RGB")
                light_previews.append(img)
                with cols[idx % 3]:
                    st.image(img, caption=f"Light {idx + 1}", use_column_width=True)
                idx += 1
            except Exception as e:
                st.warning(f"Failed to load one product image: {e}")

    st.subheader("Step 3 Generate Composite + Video")

    disabled = not (room_file is not None and light_files and len(light_files) > 0)

    if disabled:
        st.info("Upload a room image and at least one light image to enable generation.")

    if st.button("🚀 Generate", disabled=disabled):
        # 1) Save room to disk
        try:
            room_file.seek(0)
            room_bytes = room_file.read()
            ui_room_path = OUTPUT_DIR / "ui_room_upload.png"
            with open(ui_room_path, "wb") as f:
                f.write(room_bytes)
        except Exception as e:
            st.error(f"Failed to save room image to disk: {e}")
            return

        # 2) Save lights to disk
        ui_light_paths: list[Path] = []
        for idx, lf in enumerate(light_files):
            try:
                lf.seek(0)
                light_bytes = lf.read()
                lp = OUTPUT_DIR / f"ui_light_{idx + 1}.png"
                with open(lp, "wb") as f:
                    f.write(light_bytes)
                ui_light_paths.append(lp)
            except Exception as e:
                st.error(f"Failed to save light image {idx + 1} to disk: {e}")
                return

        # Step 1: composite image (exact same function)
        try:
            with st.spinner("Generating composite image..."):
                composite_img, gemini_image = generate_room_with_light(
                    room_path=ui_room_path,
                    light_paths=ui_light_paths,
                    out_path=COMPOSITE_IMAGE_PATH,
                )
        except ClientError as ce:
            st.error(f"Gemini API error while generating composite image: {ce}")
            return
        except Exception as e:
            st.error(f"Unexpected error while generating composite image: {e}")
            return

        st.success("Composite room and lights image generated.")
        st.image(composite_img, caption="Room with Installed Lights", use_container_width=True)

        # Download composite
        img_buf = BytesIO()
        composite_img.save(img_buf, format="PNG")
        img_buf.seek(0)
        st.download_button(
            label="Download composite as PNG",
            data=img_buf,
            file_name="room_with_light.png",
            mime="image/png",
        )

        # Step 2: 360° video 
        video_success = False
        try:
            with st.spinner("Generating video ..."):
                generate_360_video_from_image(
                gemini_image=gemini_image,
                out_path=VIDEO_OUTPUT_PATH,
                )

                
                video_success = True
        except ClientError as ce:
            st.error(f"Veo API error while generating video: {ce}")
        except Exception as e:
            st.error(f"Unexpected error while generating video: {e}")
            video_success = False

        if video_success and VIDEO_OUTPUT_PATH.exists():
            with open(VIDEO_OUTPUT_PATH, "rb") as vf:
                video_bytes = vf.read()

            st.success("360° video generated.")
            st.video(video_bytes)
            st.download_button(
                label="Download 360° video (MP4)",
                data=video_bytes,
                file_name="room_with_light_360.mp4",
                mime="video/mp4",
            )
        else:
            st.info("Video file was not generated or could not be read. Check terminal logs for details.")


if __name__ == "__main__":
    main()
