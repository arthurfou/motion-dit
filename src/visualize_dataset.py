"""Visualize HumanML3D motions as 3D skeletons (no SMPL-H body model needed).

The `smpl_rifke` features already contain the 24 joint positions, so we decode them
with `smplrifkefeats_to_smpldata` and draw bones with trimesh + pyrender.

Examples:
    python src/visualize_dataset.py --list 10              # print some ids + captions
    python src/visualize_dataset.py --search "jump"         # first motion matching "jump"
    python src/visualize_dataset.py --id 000021 --out skel.mp4
    python src/visualize_dataset.py --index 8 --interactive # 3D window (needs a display)
"""

import argparse
import os
import random
import time
from pathlib import Path

# Under WSL the GPU-backed EGL driver (d3d12) crashes; Mesa's software renderer is reliable.
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("GALLIUM_DRIVER", "llvmpipe")

import numpy as np
import torch
import trimesh

from utils.rifke_utils import smplrifkefeats_to_smpldata

FPS = 20
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
LEFT = {1, 4, 7, 10, 13, 16, 18, 20, 22}
RIGHT = {2, 5, 8, 11, 14, 17, 19, 21, 23}
COLORS = {"left": [47, 111, 214, 255], "right": [224, 102, 47, 255], "spine": [54, 65, 77, 255]}

# ------------------------------------------------------------------------------------- #


def load_ids(data_dir: Path):
    return sorted(p.stem for p in (data_dir / "smpl_rifke").glob("*.npy"))


def load_captions(data_dir: Path, sample_id: str):
    path = data_dir / "caption_raw" / f"{sample_id}.txt"
    return [line.split("#")[0].strip() for line in path.open() if line.strip()]


def load_joints(data_dir: Path, sample_id: str) -> np.ndarray:
    feats = torch.from_numpy(np.load(data_dir / "smpl_rifke" / f"{sample_id}.npy")).float()
    joints, _ = smplrifkefeats_to_smpldata(feats)  # (F, 24, 3), Z up
    joints = joints.numpy()
    joints[:, :, :2] -= joints[0, 0, :2]  # start the root at the origin
    return joints


# ------------------------------------------------------------------------------------- #


def skeleton_mesh(frame: np.ndarray) -> trimesh.Trimesh:
    parts = []
    for j in range(1, 24):
        a, b = frame[PARENTS[j]], frame[j]
        if np.linalg.norm(b - a) < 1e-4:
            continue
        side = "left" if j in LEFT else "right" if j in RIGHT else "spine"
        bone = trimesh.creation.cylinder(radius=0.022, segment=[a, b], sections=12)
        bone.visual.vertex_colors = COLORS[side]
        parts.append(bone)
    for j in range(24):
        ball = trimesh.creation.icosphere(subdivisions=1, radius=0.09 if j == 15 else 0.03)
        ball.apply_translation(frame[j])
        ball.visual.vertex_colors = COLORS["spine"]
        parts.append(ball)
    return trimesh.util.concatenate(parts)


def ground_mesh(joints: np.ndarray) -> trimesh.Trimesh:
    lo = joints[:, 0, :2].min(0) - 2.0
    hi = joints[:, 0, :2].max(0) + 2.0
    tiles = []
    for i, x in enumerate(np.arange(np.floor(lo[0]), hi[0], 0.5)):
        for k, y in enumerate(np.arange(np.floor(lo[1]), hi[1], 0.5)):
            tile = trimesh.creation.box(extents=[0.5, 0.5, 0.01])
            tile.apply_translation([x + 0.25, y + 0.25, -0.005])
            tile.visual.vertex_colors = [228, 232, 237, 255] if (i + k) % 2 else [204, 211, 219, 255]
            tiles.append(tile)
    return trimesh.util.concatenate(tiles)


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Camera pose for pyrender (camera looks along its -Z axis, Y up)."""
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = x, y, z, eye
    return pose


CAM_OFFSET = np.array([2.6, -3.4, 1.4])

# ------------------------------------------------------------------------------------- #


def render_mp4(joints: np.ndarray, out: str, width: int = 960, height: int = 720):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import imageio
    import pyrender

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0], ambient_light=[0.45, 0.45, 0.45])
    scene.add(pyrender.Mesh.from_trimesh(ground_mesh(joints), smooth=False))
    cam_node = scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 4.5))
    scene.add(pyrender.DirectionalLight(intensity=3.0), pose=look_at(np.array([1.0, -2, 4]), np.zeros(3)))
    renderer = pyrender.OffscreenRenderer(width, height)

    target = np.array([*joints[0, 0, :2], 0.9])
    frames, node = [], None
    for f in range(len(joints)):
        root = np.array([*joints[f, 0, :2], 0.9])
        target = target + (root - target) * 0.15  # smooth follow
        scene.set_pose(cam_node, look_at(target + CAM_OFFSET, target))
        if node is not None:
            scene.remove_node(node)
        node = scene.add(pyrender.Mesh.from_trimesh(skeleton_mesh(joints[f])))
        color, _ = renderer.render(scene)
        frames.append(color)
    renderer.delete()
    imageio.mimsave(out, frames, fps=FPS)
    print(f"Saved {out} ({len(frames)} frames, {len(frames) / FPS:.1f} s)")


def show_interactive(joints: np.ndarray):
    import pyrender

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0], ambient_light=[0.45, 0.45, 0.45])
    scene.add(pyrender.Mesh.from_trimesh(ground_mesh(joints), smooth=False))
    start = np.array([*joints[0, 0, :2], 0.9])
    cam_pose = look_at(start + CAM_OFFSET, start)
    scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 4.5), pose=cam_pose)
    meshes = [pyrender.Mesh.from_trimesh(skeleton_mesh(fr)) for fr in joints]
    node = scene.add(meshes[0])

    viewer = pyrender.Viewer(scene, run_in_thread=True, use_raymond_lighting=True,
                             viewport_size=(1000, 750), window_title="HumanML3D skeleton")
    f = 0
    while viewer.is_active:  # loop the animation until the window is closed
        time.sleep(1 / FPS)
        f = (f + 1) % len(meshes)
        viewer.render_lock.acquire()
        scene.remove_node(node)
        node = scene.add(meshes[f])
        viewer.render_lock.release()


# ------------------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="./humanml3d-data")
    pick = parser.add_mutually_exclusive_group()
    pick.add_argument("--id", help="sample id, e.g. 000021")
    pick.add_argument("--index", type=int, help="index in the sorted list of ids")
    pick.add_argument("--search", help="first motion whose captions contain this text")
    pick.add_argument("--random", action="store_true", help="random motion")
    parser.add_argument("--list", type=int, metavar="N", help="print N ids with their first caption and exit")
    parser.add_argument("--out", default="skeleton.mp4", help="output video path")
    parser.add_argument("--interactive", action="store_true", help="open a 3D window instead of saving a video")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    ids = load_ids(data_dir)

    if args.list:
        for i, sid in enumerate(ids[: args.list]):
            n = np.load(data_dir / "smpl_rifke" / f"{sid}.npy", mmap_mode="r").shape[0]
            print(f"[{i:4d}] {sid}  {n / FPS:4.1f}s  {load_captions(data_dir, sid)[0]}")
        return

    if args.id:
        sid = args.id
    elif args.index is not None:
        sid = ids[args.index]
    elif args.search:
        term = args.search.lower()
        sid = next((s for s in ids if any(term in c.lower() for c in load_captions(data_dir, s))), None)
        if sid is None:
            raise SystemExit(f'No caption contains "{args.search}"')
    elif args.random:
        sid = random.choice(ids)
    else:
        sid = ids[0]

    joints = load_joints(data_dir, sid)
    print(f"Motion {sid}: {len(joints)} frames ({len(joints) / FPS:.1f} s)")
    for c in load_captions(data_dir, sid):
        print(f"  - {c}")

    if args.interactive:
        show_interactive(joints)
    else:
        render_mp4(joints, args.out)


if __name__ == "__main__":
    main()
