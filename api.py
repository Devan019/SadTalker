import os
import uuid
import shutil
import requests
from time import strftime
from typing import Any, List, Dict
from fastapi import FastAPI
from pydantic import BaseModel, Field
import cloudinary.uploader
from dotenv import load_dotenv
load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)
# from helpers.add_subtitle import burn_subtitles
from src.utils.init_path import init_path
from src.test_audio2coeff import Audio2Coeff  
from src.utils.preprocess import CropAndExtract
from src.utils.preprocess import CropAndExtract
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.facerender.animate import AnimateFromCoeff
from helper.add_subtitle import burn_subtitles
app = FastAPI()

# Image name -> file path map
ImageMapper = {
    "Mark Zuckerberg": "./images/mark_zuckerberg.png",
    "Elon Musk": "./images/elon_musk.jpg",
}

class VideoRequest(BaseModel):
    source_image: str  # Can be a mapped name, local file path, or URL
    driven_audio: str  # Local path or URL
    size: int = 256
    pose_style: int = 0
    batch_size: int = 2
    expression_scale: float = 1.0
    use_cpu: bool = False
    captions: Any = Field(default_factory=list)  # Avoid shared list

def download_if_url(path_or_url: str, dest_path: str) -> str:
    """Download from URL if needed, otherwise copy local file."""
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        r = requests.get(path_or_url, stream=True)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return dest_path
    else:
        shutil.copy(path_or_url, dest_path)
        return dest_path

@app.post("/generate-video")
async def generate_video(req: VideoRequest):
    progress = {
        "download_files": False,
        "extract_coefficients": False,
        "audio_to_coefficients": False,
        "coefficients_to_video": False,
        "upload_video": False
    }

    # Resolve source image: map name -> file path if needed
    source_input = ImageMapper.get(req.source_image, req.source_image)

    device = "cuda" if not req.use_cpu else "cpu"
    timestamp = strftime("%Y_%m_%d_%H.%M.%S")
    result_dir = f"results/{timestamp}"
    os.makedirs(result_dir, exist_ok=True)

    # Step 1: Download or copy source image & audio
    source_image_dest = os.path.join(result_dir, os.path.basename(source_input))
    source_image_path = download_if_url(source_input, source_image_dest)

    audio_dest = os.path.join(result_dir, os.path.basename(req.driven_audio))
    audio_path = download_if_url(req.driven_audio, audio_dest)
    progress["download_files"] = True

    # Step 2: Init models
    current_root_path = os.getcwd()
    sadtalker_paths = init_path(
        './checkpoints',
        os.path.join(current_root_path, 'src/config'),
        req.size,
        old_version=False,
        preprocess='crop'
    )
    preprocess_model = CropAndExtract(sadtalker_paths, device)
    audio_to_coeff = Audio2Coeff(sadtalker_paths, device)
    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)

    # Step 3: Extract coefficients
    first_frame_dir = os.path.join(result_dir, "first_frame")
    os.makedirs(first_frame_dir, exist_ok=True)
    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
        source_image_path, first_frame_dir, 'crop',
        source_image_flag=True, pic_size=req.size
    )
    if first_coeff_path is None:
        return {"error": "Failed to extract coefficients from source image."}
    progress["extract_coefficients"] = True

    # Step 4: Audio to coefficients
    batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path=None, still=False)
    coeff_path = audio_to_coeff.generate(batch, result_dir, req.pose_style, ref_pose_coeff_path=None)
    progress["audio_to_coefficients"] = True

    # Step 5: Coefficients to video
    data = get_facerender_data(
        coeff_path, crop_pic_path, first_coeff_path, audio_path,
        batch_size=req.batch_size,
        input_yaw_list=None,
        input_pitch_list=None,
        input_roll_list=None,
        expression_scale=req.expression_scale,
        still_mode=False,
        preprocess='crop',
        size=req.size
    )
    result_video_path = animate_from_coeff.generate(
        data, result_dir, source_image_path, crop_info,
        enhancer=None,
        background_enhancer=None,
        preprocess='crop',
        img_size=req.size
    )
    progress["coefficients_to_video"] = True

    # Step 6: Burn subtitles into a final file
    final_video_path = os.path.join(result_dir, "final_with_subs.mp4")
    burn_subtitles(result_video_path, req.captions, final_video_path)

    # Step 7: Upload to Cloudinary
    unique_id = str(uuid.uuid4())
    upload_result = cloudinary.uploader.upload(
        final_video_path,
        resource_type="video",
        folder="zennvid",
        public_id=unique_id
    )
    progress["upload_video"] = True

    # Step 8: Clean up
    for file_path in [source_image_path, audio_path, result_video_path, final_video_path]:
        if os.path.exists(file_path):
            os.remove(file_path)

    return {
        "video": upload_result.get("secure_url"),
        "progress": progress
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9000)
