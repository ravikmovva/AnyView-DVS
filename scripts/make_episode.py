#!/usr/bin/env python3
"""
make_episode.py -- write your own footage + camera poses into AnyView-DVS's unified
scene layout (frames variant), so scripts/infer.py can run on it.

Dependency-light on purpose (numpy + imageio, no torch): data prep runs outside the
inference venv.

Layout written (matches anyview/unified.py):

  <out>/metadata.json
  <out>/rgb/<cam>/0000000000.jpg          one image per frame (input cam; target optional)
  <out>/lowdim/<cam>/0000000000.npz       camera, timestep, intrinsics (3,3), extrinsics (4,4)

Conventions (pinned from the repo):
  * extrinsics are stored as cam2world, OpenCV axes: columns of R = right | DOWN | forward,
    last column = camera position. infer.py inverts to world2cam itself.
  * infer.py re-anchors on the target camera's first frame, so the absolute world frame is
    irrelevant -- only the relative pose between input and target matters.
  * K is pinhole, zero skew, at the frames' NATIVE resolution (infer.py rescales it).
  * cam1 = input (needs RGB), cam0 = target (RGB optional; omit it for no-GT extreme DVS).
    The roles are also written to metadata specific.roles.
  * scale_factor goes to metadata specific.scale_factor, because custom data has no dataset
    name to look up in cameras.DATASET_SCALE_FACTORS. Repo values: driving 1/16,
    kubric 1/32, everything else 1/8. infer.py --scale-factor still overrides it.
  * clip length used by infer.py must be 1 + 4k and <= 41; it takes the longest such prefix
    by default, so the episode itself may be longer.

Library use (write_scene() takes any number of cameras; make_episode() is the 2-camera case):
    from make_episode import make_episode, orbit_target
    c2w_out = orbit_target(c2w_in, yaw_deg=90)
    make_episode("ep", frames, K, c2w_in, K, c2w_out)

CLI:
    # input frames dir + poses (.npz with intrinsics (3,3)|(T,3,3) and cam2world (T,4,4)),
    # target = a designed 0->90 deg orbit around the input camera's look-at point
    python make_episode.py --frames my_clip/ --poses my_poses.npz --orbit 90 --out ep/

    # target poses supplied explicitly (and optionally target GT frames)
    python make_episode.py --frames in/ --poses in.npz --target-poses tgt.npz \
        [--target-frames tgt/] --out ep/
"""

import argparse
import glob
import json
import os

import numpy as np

FRAME_NAME = "{:010d}"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


# --------------------------------------------------------------------------- loading
def load_frames(src, max_frames=None):
    """Frames from a directory of images (sorted by name) or a video file.
    Returns (T, H, W, 3) uint8."""
    import imageio.v2 as imageio

    if os.path.isdir(src):
        paths = sorted(p for p in glob.glob(os.path.join(src, "*"))
                       if p.lower().endswith(IMAGE_EXTENSIONS))
        if not paths:
            raise FileNotFoundError(f"no images in {src}")
        if max_frames:
            paths = paths[:max_frames]
        frames = [np.asarray(imageio.imread(p))[..., :3] for p in paths]
    else:
        reader = imageio.get_reader(src)
        frames = []
        for i, f in enumerate(reader):
            if max_frames and i >= max_frames:
                break
            frames.append(np.asarray(f)[..., :3])
        reader.close()
    video = np.stack(frames).astype(np.uint8)
    return video


def load_poses(path):
    """Poses from an .npz (keys: intrinsics or K; cam2world, or world2cam / extrinsics with
    an optional 'transform' string). Returns (K (T,3,3) or (3,3), cam2world (T,4,4))."""
    d = np.load(path, allow_pickle=False)
    K = d["intrinsics"] if "intrinsics" in d else d["K"]
    if "cam2world" in d:
        c2w = d["cam2world"]
    elif "world2cam" in d:
        c2w = np.linalg.inv(d["world2cam"].astype(np.float64))
    else:
        c2w = d["extrinsics"]
        if "transform" in d and str(d["transform"]) == "world2cam":
            c2w = np.linalg.inv(c2w.astype(np.float64))
    return np.asarray(K, np.float64), np.asarray(c2w, np.float64)


# ------------------------------------------------------------------------ geometry
def _as_seq(K, T):
    K = np.asarray(K, np.float64)
    if K.ndim == 2:
        K = np.broadcast_to(K, (T, 3, 3)).copy()
    if K.shape != (T, 3, 3):
        raise ValueError(f"intrinsics shape {K.shape}, expected (3,3) or ({T},3,3)")
    return K


def _check_c2w(c2w, name):
    c2w = np.array(c2w, np.float64)  # copy: callers' arrays are never modified
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"{name}: expected (T,4,4) cam2world, got {c2w.shape}")
    R = c2w[:, :3, :3]
    err = np.abs(np.einsum("nji,njk->nik", R, R) - np.eye(3)).max()
    if err > 1e-3:
        raise ValueError(f"{name}: rotation block not orthonormal (max err {err:.2e}); "
                         "is this world2cam or a non-rigid matrix?")
    if np.linalg.det(R).min() < 0:
        raise ValueError(f"{name}: det(R) < 0 -- a handedness flip (OpenGL vs OpenCV axes?)")
    if np.abs(c2w[:, 3] - [0, 0, 0, 1]).max() > 1e-6:
        raise ValueError(f"{name}: last row is not [0,0,0,1]")
    return c2w


def _rot(axis, deg):
    axis = np.asarray(axis, np.float64)
    axis = axis / np.linalg.norm(axis)
    a = np.radians(deg)
    x, y, z = axis
    Kx = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * Kx @ Kx


def orbit_target(c2w_in, yaw_deg=90.0, radius=None, sweep=True, up=None):
    """Design a target trajectory by orbiting the INPUT camera about a vertical axis through
    its look-at point.

    c2w_in : (T,4,4) input cam2world (OpenCV axes).
    yaw_deg: final orbit angle. sweep=True ramps 0 -> yaw_deg over the clip (a moving
             camera); sweep=False holds yaw_deg for every frame (a fixed offset view).
    radius : look-at distance along the input camera's forward axis at frame 0. Default:
             2x the input trajectory's extent, or 1.0 for a static camera -- set it from
             your scene's depth when you know it, since it decides the baseline.
    up     : world up vector. Default: the input camera's -y (OpenCV y points DOWN).

    The target keeps looking at the same point, so it is a valid extreme-DVS probe: at
    90 deg it sees the subject side-on from a place no camera ever stood."""
    c2w_in = _check_c2w(c2w_in, "c2w_in")
    T = len(c2w_in)
    pos = c2w_in[:, :3, 3]
    if radius is None:
        extent = np.linalg.norm(pos.max(0) - pos.min(0))
        radius = 2.0 * extent if extent > 1e-6 else 1.0
    up = -c2w_in[0, :3, 1] if up is None else np.asarray(up, np.float64)
    angles = np.linspace(0.0, yaw_deg, T) if sweep else np.full(T, float(yaw_deg))

    out = np.empty_like(c2w_in)
    for t in range(T):
        fwd = c2w_in[t, :3, 2]
        pivot = pos[t] + radius * fwd
        Rz = _rot(up, angles[t])
        out[t] = np.eye(4)
        out[t, :3, :3] = Rz @ c2w_in[t, :3, :3]
        out[t, :3, 3] = pivot + Rz @ (pos[t] - pivot)
    return out


# --------------------------------------------------------------------------- writer
def write_scene(out, cameras, roles=None, scale_factor=1.0 / 8.0, framerate=None,
                name="custom", recenter=True, jpeg_quality=95, extra=None):
    """Write one unified scene (frames variant) holding any number of cameras.

    cameras : {name: dict(K=(3,3)|(T,3,3), c2w=(T,4,4),
                          frames=(T,H,W,3) uint8 | None,     # pixels to encode as JPEG
                          frame_paths=[T paths] | None,      # or existing .jpg files, copied
                                                             # byte-exact (no re-compression)
                          hw=(H,W) | None)}                  # needed when there are no pixels
    roles   : {"input": cam, "target": cam} defaults for infer.py (--input-cam/--target-cam
              still pick any pair). All cameras must have the same T; row t of every camera
              is "time step t" (infer.py indexes positionally).
    recenter: subtract the input camera's frame-0 position (in float64) from every
              translation before the float32 save. Pure translation, so relative geometry
              is untouched; it keeps sub-millimetre precision when world coordinates are
              large (nuScenes global frames are kilometres from the origin).
    """
    import shutil
    import imageio.v2 as imageio

    if not cameras:
        raise ValueError("no cameras")
    prepped = {}
    T = None
    for cam, c in cameras.items():
        c2w = _check_c2w(c["c2w"], f"{cam}.c2w")
        T = len(c2w) if T is None else T
        if len(c2w) != T:
            raise ValueError(f"{cam}: {len(c2w)} poses, other cameras have {T}")
        K = _as_seq(c["K"], T)
        if np.abs(K[:, 0, 1]).max() > 1e-9 or np.abs(K[:, 2] - [0, 0, 1]).max() > 1e-9:
            raise ValueError(f"{cam}: expected pinhole zero-skew K with last row [0,0,1]")
        frames, paths, hw = c.get("frames"), c.get("frame_paths"), c.get("hw")
        if frames is not None:
            frames = np.asarray(frames)
            if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
                raise ValueError(f"{cam}: frames must be (T,H,W,3) uint8, got "
                                 f"{frames.shape} {frames.dtype}")
            if len(frames) != T:
                raise ValueError(f"{cam}: {len(frames)} frames, need {T}")
            hw = frames.shape[1:3]
        elif paths is not None:
            if len(paths) != T:
                raise ValueError(f"{cam}: {len(paths)} frame paths, need {T}")
            if not all(str(p).lower().endswith((".jpg", ".jpeg")) for p in paths):
                raise ValueError(f"{cam}: frame_paths must be .jpg (scene uses one extension)")
            if hw is None:
                hw = np.asarray(imageio.imread(paths[0])).shape[:2]
        if hw is None:
            raise ValueError(f"{cam}: no frames, so pass hw=(H,W) for its intrinsics")
        prepped[cam] = (K, c2w, frames, paths, tuple(int(v) for v in hw))

    roles = dict(roles or {})
    origin = None
    if recenter:
        anchor = roles.get("input", next(iter(prepped)))
        origin = prepped[anchor][1][0, :3, 3].copy()
        for cam in prepped:
            prepped[cam][1][:, :3, 3] -= origin
    usable = min(T, 41)
    usable -= (usable - 1) % 4
    if usable != T:
        print(f"note: infer.py will use the first {usable} of {T} frames (1+4k, <= 41)")

    os.makedirs(out, exist_ok=True)
    for cam, (K, c2w, frames, paths, hw) in prepped.items():
        ld = os.path.join(out, "lowdim", cam)
        os.makedirs(ld, exist_ok=True)
        if frames is not None or paths is not None:
            os.makedirs(os.path.join(out, "rgb", cam), exist_ok=True)
        for t in range(T):
            np.savez(os.path.join(ld, FRAME_NAME.format(t) + ".npz"),
                     camera=np.array(cam), timestep=np.int64(t),
                     intrinsics=K[t].astype(np.float32),
                     extrinsics=c2w[t].astype(np.float32))
            dst = os.path.join(out, "rgb", cam, FRAME_NAME.format(t) + ".jpg")
            if frames is not None:
                imageio.imwrite(dst, frames[t], quality=jpeg_quality)
            elif paths is not None:
                shutil.copyfile(paths[t], dst)

    sizes = {cam: list(p[4]) for cam, p in prepped.items()}
    uniq = {tuple(v) for v in sizes.values()}
    specific = {"scale_factor": float(scale_factor), "resolution_per_camera": sizes}
    if roles:
        specific["roles"] = roles
    if origin is not None:
        specific["world_offset"] = origin.tolist()  # subtracted from every camera position
    if extra:
        specific.update(extra)
    meta = {
        "info": {"name": name, "storage": "frames"},
        "cameras": list(prepped),
        "resolution": list(next(iter(uniq))) if len(uniq) == 1 else None,
        "num_frames": int(T),
        "framerate": framerate,
        "rgb": {"extension": "jpg"},
        "extrinsics": {"transform": "cam2world"},
        "specific": specific,
    }
    with open(os.path.join(out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return out


def make_episode(out, input_frames, K_in, c2w_in, K_out, c2w_out, target_frames=None,
                 scale_factor=1.0 / 8.0, input_cam="cam1", target_cam="cam0",
                 target_hw=None, framerate=None, name="custom", jpeg_quality=95):
    """Two-camera convenience wrapper around write_scene().

    input_frames : (T,H,W,3) uint8 -- the conditioning video.
    K_in, K_out  : (3,3) or (T,3,3) pinhole intrinsics at each camera's native resolution.
    c2w_in/out   : (T,4,4) cam2world, OpenCV axes (extra poses beyond T are dropped).
    target_frames: optional (T,H,W,3) ground truth for the target (for scoring only).
    target_hw    : (H,W) the target intrinsics refer to when there are no target frames;
                   defaults to the input size.
    """
    input_frames = np.asarray(input_frames)
    T = len(input_frames)
    if len(c2w_in) < T or len(c2w_out) < T:
        raise ValueError(f"need {T} poses per camera, got {len(c2w_in)} / {len(c2w_out)}")
    K_in = K_in[:T] if np.ndim(K_in) == 3 else K_in
    K_out = K_out[:T] if np.ndim(K_out) == 3 else K_out
    if target_frames is not None:
        target_frames = np.asarray(target_frames)[:T]
    hw_out = target_hw if target_hw is not None else input_frames.shape[1:3]
    cameras = {
        target_cam: dict(K=K_out, c2w=np.asarray(c2w_out)[:T], frames=target_frames, hw=hw_out),
        input_cam: dict(K=K_in, c2w=np.asarray(c2w_in)[:T], frames=input_frames),
    }
    return write_scene(out, cameras, roles={"input": input_cam, "target": target_cam},
                       scale_factor=scale_factor, framerate=framerate, name=name,
                       jpeg_quality=jpeg_quality)


# ------------------------------------------------------------------------------ CLI
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", required=True, help="input frames dir or video file")
    p.add_argument("--poses", required=True, help="input .npz: intrinsics + cam2world")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--orbit", type=float, metavar="DEG",
                   help="design the target as an orbit to DEG degrees")
    g.add_argument("--target-poses", help="target .npz: intrinsics + cam2world")
    p.add_argument("--fixed", action="store_true",
                   help="with --orbit: hold the final angle instead of sweeping 0->DEG")
    p.add_argument("--radius", type=float, default=None, help="with --orbit: look-at distance")
    p.add_argument("--target-frames", help="optional target GT frames dir or video")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--scale-factor", type=float, default=1.0 / 8.0,
                   help="driving 1/16 (0.0625), kubric 1/32, else 1/8 (default)")
    p.add_argument("--framerate", type=float, default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    frames = load_frames(a.frames, a.max_frames)
    K_in, c2w_in = load_poses(a.poses)
    T = len(frames)
    c2w_in = c2w_in[:T]
    if a.orbit is not None:
        K_out = K_in
        c2w_out = orbit_target(c2w_in, a.orbit, radius=a.radius, sweep=not a.fixed)
    else:
        K_out, c2w_out = load_poses(a.target_poses)
    tgt = load_frames(a.target_frames, T) if a.target_frames else None
    make_episode(a.out, frames, K_in, c2w_in, K_out, c2w_out, target_frames=tgt,
                 scale_factor=a.scale_factor, framerate=a.framerate)
    print(f"wrote {a.out}: {T} frames, {frames.shape[2]}x{frames.shape[1]}")


if __name__ == "__main__":
    main()
