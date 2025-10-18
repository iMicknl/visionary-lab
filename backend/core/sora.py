import requests
import os
import logging
import json
import io
from typing import List
import os
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Sora:
    def __init__(self, resource_name, deployment_name, api_key, api_version="preview"):
        self.resource_name = resource_name
        self.deployment_name = deployment_name
        self.api_key = api_key
        self.api_version = api_version
        self.base_url = f"https://{self.resource_name}.openai.azure.com/openai/v1/video"

        self.client = OpenAI(
            api_key=api_key,
            base_url=f"https://{self.resource_name}.openai.azure.com/openai/v1/",
        )

        self.headers = {"api-key": self.api_key, "Content-Type": "application/json"}
        logger.info(
            f"Initialized Sora client with resource: {resource_name}, deployment: {deployment_name}"
        )

    def create_video_generation_job(
        self, prompt, n_seconds, height, width, n_variants=1
    ):
        video = self.client.videos.create(
            model=self.deployment_name,
            size="1280x720",
            seconds="12",
            prompt=prompt,
        )

        logger.info(f"Creating video generation job with prompt: {prompt[:50]}...")

        print(video.to_json())

        response = video.to_dict()

        response["prompt"] = prompt
        response["n_variants"] = n_variants
        response["n_seconds"] = n_seconds
        response["height"] = height
        response["width"] = width

        return response

    def create_video_generation_job_with_images(
        self, prompt, images, image_filenames, n_seconds, height, width, n_variants=1
    ):
        """Create video generation job with image inpainting using multipart upload.
        First attempts without explicit crop bounds to let API defaults apply.
        If the API rejects the request, retries once including full-image crop bounds.
        """
        url = f"{self.base_url}/generations/jobs?api-version={self.api_version}"

        def build_files():
            return [
                ("files", (filename, io.BytesIO(image_content), "image/jpeg"))
                for image_content, filename in zip(images, image_filenames)
            ]

        # Remove Content-Type from headers for multipart request
        multipart_headers = {
            k: v for k, v in self.headers.items() if k.lower() != "content-type"
        }

        # Common form fields
        base_data = {
            "prompt": prompt,
            "height": str(height),
            "width": str(width),
            "n_seconds": str(n_seconds),
            "n_variants": str(n_variants),
            "model": self.deployment_name,
        }

        # Attempt 1: Without crop_bounds to rely on API defaults
        data_no_crop = {
            **base_data,
            "inpaint_items": json.dumps(
                [
                    {
                        "frame_index": 0,
                        "type": "image",
                        "file_name": filename,
                    }
                    for filename in image_filenames
                ]
            ),
        }

        logger.info(
            f"Creating video job (no crop bounds) with {len(images)} images and prompt: {prompt[:50]}..."
        )
        response = requests.post(
            url, headers=multipart_headers, data=data_no_crop, files=build_files()
        )

        if response.ok:
            return response.json()

        # Log and retry once with full-image crop bounds if the first attempt failed
        logger.warning(
            f"SORA API rejected request without crop bounds: {response.status_code} {response.text}. Retrying with crop bounds."
        )

        data_with_crop = {
            **base_data,
            "inpaint_items": json.dumps(
                [
                    {
                        "frame_index": 0,
                        "type": "image",
                        "file_name": filename,
                        "crop_bounds": {
                            "left_fraction": 0.0,
                            "top_fraction": 0.0,
                            "right_fraction": 1.0,
                            "bottom_fraction": 1.0,
                        },
                    }
                    for filename in image_filenames
                ]
            ),
        }

        response2 = requests.post(
            url, headers=multipart_headers, data=data_with_crop, files=build_files()
        )

        if not response2.ok:
            logger.error(
                f"SORA API error (with crop bounds): {response2.status_code} {response2.text}"
            )
            response2.raise_for_status()

        return response2.json()

    def get_video_generation_job(self, job_id):
        # url = f"{self.base_url}/generations/jobs/{job_id}?api-version={self.api_version}"
        logger.info(f"Getting video generation job: {job_id}")
        response = self.client.videos.retrieve(job_id)

        print(response.to_json())

        response = response.to_dict()

        response["prompt"] = ""
        response["n_variants"] = 0
        response["n_seconds"] = 0
        response["height"] = 0
        response["width"] = 0

        return response

    def delete_video_generation_job(self, job_id):
        url = (
            f"{self.base_url}/generations/jobs/{job_id}?api-version={self.api_version}"
        )
        logger.info(f"Deleting video generation job: {job_id}")
        response = requests.delete(url, headers=self.headers)
        response.raise_for_status()
        return response.status_code

    def list_video_generation_jobs(
        self, before=None, after=None, limit=10, statuses=None
    ):
        url = f"{self.base_url}/generations/jobs?api-version={self.api_version}"
        params = {"limit": limit}
        if before:
            params["before"] = before
        if after:
            params["after"] = after
        if statuses:
            params["statuses"] = ",".join(statuses)
        logger.info(f"Listing video generation jobs with params: {params}")
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()

    def get_video_generation_video_content(
        self, generation_id, file_name, target_folder="videos"
    ):
        """
        Download the video content for a given generation as an MP4 file to the local folder.

        Args:
            generation_id (str): The generation ID.
            file_name (str): The filename to save the video as (include .mp4 extension).
            target_folder (str): The folder to save the video to (default: 'videos').

        Returns:
            str: The path to the downloaded file.
        """
        url = f"{self.base_url}/generations/{generation_id}/content/video?api-version={self.api_version}"

        # Create directory if it doesn't exist
        os.makedirs(target_folder, exist_ok=True)

        file_path = os.path.join(target_folder, file_name)

        logger.info(
            f"Downloading video content for generation {generation_id} to {file_path}"
        )

        # Use the same headers as in the notebook - important!
        response = requests.get(url, headers=self.headers, stream=True)
        response.raise_for_status()

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)

        logger.info(f"Successfully downloaded video to {file_path}")
        return file_path
