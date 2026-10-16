
#!/usr/bin/env python3
"""Convert BC evaluation videos into GIF previews for GitHub README."""

import argparse
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image


def convert_to_gif(source, destination, width=320, stride=3, fps=8):
    reader = imageio.get_reader(str(source))
    frames = []

    try:
        for index, frame in enumerate(reader):
            if index % stride != 0:
                continue

            image = Image.fromarray(frame).convert("RGB")
            height = round(image.height * width / image.width)
            image = image.resize((width, height), Image.Resampling.LANCZOS)
            frames.append(image)

    finally:
        reader.close()

    if not frames:
        raise RuntimeError(f"No frames found in {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)

    frames[0].save(
        destination,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=True,
    )

    print(f"Created: {destination}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--videos-dir",
        default="results/bc_eval_multiseed_media/videos",
    )
    parser.add_argument("--output-dir", default="assets")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--stride", type=int, default=3)
    args = parser.parse_args()

    video_dir = Path(args.videos_dir)
    output_dir = Path(args.output_dir)

    for label, count in [("success", 4), ("failure", 2)]:
        videos = sorted(video_dir.glob(f"{label}_*.mp4"))

        if len(videos) < count:
            print(
                f"Warning: Found {len(videos)} {label} videos; "
                f"requested {count}."
            )

        for index, video in enumerate(videos[:count], start=1):
            destination = output_dir / f"{label}_{index}.gif"

            convert_to_gif(
                video,
                destination,
                width=args.width,
                stride=args.stride,
            )


if __name__ == "__main__":
    main()
