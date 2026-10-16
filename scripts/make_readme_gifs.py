#!/usr/bin/env python3
"""Create lightweight inline README GIF previews from evaluation MP4 clips."""
import argparse
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image


def create_gif(src, dst, width=420, fps=8, frame_stride=4, max_frames=90):
    reader = imageio.get_reader(str(src))
    frames = []
    try:
        for index, frame in enumerate(reader):
            if index % frame_stride:
                continue
            img = Image.fromarray(frame).convert('RGB')
            h = max(1, round(img.height * width / img.width))
            frames.append(img.resize((width, h)))
            if len(frames) >= max_frames:
                break
    finally:
        reader.close()
    if not frames:
        raise ValueError(f'No frames in {src}')
    dst.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(dst, save_all=True, append_images=frames[1:], duration=1000 // fps, loop=0, optimize=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos-dir', default='results/bc_eval_multiseed_media/videos')
    parser.add_argument('--width', type=int, default=420)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    video_dir = root / args.videos_dir
    assets = root / 'assets'
    text = (root / 'README.md').read_text()
    snippets = []
    for label in ['success', 'failure']:
        files = sorted(video_dir.glob(f'{label}_*.mp4'))[:3]
        snippets.append(f'**{label.title()} examples**')
        for idx, src in enumerate(files, 1):
            rel = f'assets/{label}_{idx}.gif'
            create_gif(src, root / rel, width=args.width)
            snippets.append(f'![{label} rollout {idx}]({rel})')
        if not files:
            snippets.append(f'No {label} MP4 files found. [View uploaded videos on W&B](https://wandb.ai/muhamuttaqien/robosuite-project/runs/atyi069p).')
        snippets.append('')
    start, end = '<!-- PREVIEW_START -->', '<!-- PREVIEW_END -->'
    if start not in text or end not in text:
        raise RuntimeError('Missing README preview markers')
    a, remainder = text.split(start, 1)
    _, b = remainder.split(end, 1)
    (root / 'README.md').write_text(a + start + '\n' + '\n'.join(snippets) + end + b)
    print('Updated README preview gallery. Review GIF sizes before committing.')


if __name__ == '__main__':
    main()
