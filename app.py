import io
import os

import modal

stub = modal.Stub()


@stub.function(
    image=modal.Image.debian_slim().pip_install("torch", "diffusers[torch]", "transformers", "ftfy"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="any",
)
async def run_stable_diffusion(prompt: str):
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        use_auth_token=os.environ["HUGGINGFACE_TOKEN"],
    ).to("cuda")

    image = pipe(prompt, num_inference_steps=10).images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    return img_bytes


@stub.local_entrypoint()
def main():
    img_bytes = run_stable_diffusion.call("Kim Jung Un supreme leader in Roman emperor clothes and crown")
    with open("./tmp/output.png", "wb") as f:
        f.write(img_bytes)
