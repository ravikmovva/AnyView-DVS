#!/usr/bin/env python3
"""
nuscenes_to_episode.py -- turn a nuScenes clip into one AnyView-DVS unified scene holding
a set of TIME-SYNCED surround cameras, so infer.py can run any camera -> any other camera in
the episode with real ground-truth pixels for the target, and train_dvs.py can sample any
ordered pair from it.

Why nuScenes: a calibrated surround rig (AnyView's driving domain) from a dataset AnyView
was NOT trained on (its driving sources are Argoverse2, DDAD, Lyft, PD4D, Waymo).

Reads the raw JSON tables directly (numpy + imageio only; no nuscenes-devkit needed):
    <dataroot>/<version>/{scene,sample,sample_data,ego_pose,calibrated_sensor,sensor}.json
    <dataroot>/samples/CAM_*/...jpg   (keyframes, 2 Hz)
    <dataroot>/sweeps/CAM_*/...jpg    (the in-between frames; cameras run at ~12 Hz)
The two big tables (sample_data, ego_pose) are streamed and filtered to the scene and
cameras requested, so full trainval converts in a few hundred MB of RAM.

Geometry, per camera, at THAT camera's own capture time:
    cam2world = T_global<-ego(ego_pose of this sample_data) @ T_ego<-cam(calibrated_sensor)
nuScenes camera frames are already OpenCV (x right, y down, z forward). nuScenes publishes
only a 3x3 K per camera (no distortion coefficients) and its devkit projects with K alone, so
the images are treated as pinhole. Rotations are quaternions [w, x, y, z].

Time sync -- measured on all 85 scenes of trainval part 1:
  * Each camera is triggered by the 20 Hz lidar and fires on 3 of every 5 rotations, so its
    frame gaps run 100, 100, 50 ms (12 Hz on average, not an even 83 ms).
  * CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT and CAM_BACK_RIGHT skip the SAME rotations:
    every pair among them stays within 8-16 ms for whole scenes. CAM_BACK and CAM_BACK_LEFT
    skip different ones, so against CAM_FRONT about a third of their frames are 50+ ms off
    and no scene gives 41 consecutive synced frames. Hence the default --cameras is the
    synced four (pairs at ~55, ~110 and ~165 deg -- the widest is FRONT_LEFT -> BACK_RIGHT).
  * Rows are the input camera's frames; every other camera is matched one-to-one (no image
    used twice), and the clip is the first window of --num-frames rows at or after
    --start-sec in which every camera is within --max-dt-ms and no frame was dropped
    (no gap over --max-gap-ms). Each pose is exact for its own frame; only things that
    move during the residual offset are off.

Usage:
    python nuscenes_to_episode.py --dataroot ~/data/nuscenes --version v1.0-trainval --list
    python nuscenes_to_episode.py --dataroot ~/data/nuscenes --version v1.0-trainval \
        --scene scene-0061 --start-sec 2 --out episodes/n0061 --overlay
    python scripts/infer.py --episode episodes/n0061 --input-cam CAM_FRONT \
        --target-cam CAM_BACK_RIGHT ...

--overlay writes <out>/_check/*.jpg: lidar points (from LIDAR_TOP, through the same saved
episode matrices) drawn on every camera's frame 0. If the points land on the scene's
edges, the poses are right; an axis or composition error throws them visibly off.

nuScenes is CC BY-NC-SA 4.0 (non-commercial); download it yourself from nuscenes.org.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_episode import write_scene  # noqa: E402

CAMERAS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
           "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_BACK")
SYNCED = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT")  # see docstring
DRIVING_SCALE = 1.0 / 16.0   # cameras.DATASET_SCALE_FACTORS value for every driving source


# ------------------------------------------------------------------------- tables
def iter_json_array(path, chunk=1 << 24):
    """Yield the objects of a top-level JSON array one at a time, in bounded memory.
    trainval's sample_data.json is 1.35 GB and ego_pose.json 0.65 GB; json.load() on them
    needs many times that in Python objects."""
    dec = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        buf = f.read(chunk)
        pos = buf.index("[") + 1
        while True:
            while True:
                while pos < len(buf) and buf[pos] in " \t\r\n,":
                    pos += 1
                if pos < len(buf):
                    break
                more = f.read(chunk)
                if not more:
                    return
                buf, pos = more, 0
            if buf[pos] == "]":
                return
            try:
                obj, end = dec.raw_decode(buf, pos)
            except json.JSONDecodeError:
                more = f.read(chunk)
                if not more:
                    raise
                buf, pos = buf[pos:] + more, 0
                continue
            yield obj
            pos = end
            if pos > chunk:
                buf, pos = buf[pos:], 0


SD_KEYS = ("token", "sample_token", "ego_pose_token", "calibrated_sensor_token", "timestamp",
           "is_key_frame", "height", "width", "filename", "next")


class NuScenes:
    def __init__(self, dataroot, version="v1.0-mini", channels=None, scenes=None, need_pose=True):
        """channels / scenes: keep only these (None = all). need_pose=False skips ego_pose.
        Only the small tables are loaded whole; sample_data and ego_pose are streamed and
        filtered, so a single-scene conversion on full trainval stays in a few hundred MB."""
        self.root = os.path.expanduser(dataroot)
        tdir = os.path.join(self.root, version)
        if not os.path.isdir(tdir):
            raise FileNotFoundError(f"no table directory {tdir} (wrong --dataroot/--version?)")

        def load(name):
            with open(os.path.join(tdir, name + ".json")) as f:
                return json.load(f)

        self.scenes = load("scene")
        samples = load("sample")
        self.calib = {r["token"]: r for r in load("calibrated_sensor")}
        sensor = {r["token"]: r for r in load("sensor")}
        self.sample_scene = {s["token"]: s["scene_token"] for s in samples}
        self.sample_ts = {s["token"]: s["timestamp"] for s in samples}
        self._chan = {t: sensor[c["sensor_token"]]["channel"] for t, c in self.calib.items()}

        keep_cal = {t for t, ch in self._chan.items() if channels is None or ch in channels}
        scene_toks = None
        if scenes is not None:
            scene_toks = {s["token"] for s in self.scenes if s["name"] in scenes}
            missing = set(scenes) - {s["name"] for s in self.scenes}
            if missing:
                raise KeyError(f"no scene(s) {sorted(missing)}; try --list")

        self.sample_data = {}
        for r in iter_json_array(os.path.join(tdir, "sample_data.json")):
            if r["calibrated_sensor_token"] not in keep_cal:
                continue
            if scene_toks is not None and self.sample_scene.get(r["sample_token"]) not in scene_toks:
                continue
            self.sample_data[r["token"]] = {k: r[k] for k in SD_KEYS}

        self.ego_pose = {}
        if need_pose:
            need = {r["ego_pose_token"] for r in self.sample_data.values()}
            for r in iter_json_array(os.path.join(tdir, "ego_pose.json")):
                if r["token"] in need:
                    self.ego_pose[r["token"]] = {"rotation": r["rotation"],
                                                 "translation": r["translation"]}

        # keyframe sample_data per (sample, channel): the entry points into each chain
        self.key = {}
        for sd in self.sample_data.values():
            if sd["is_key_frame"]:
                self.key[(sd["sample_token"], self.channel(sd))] = sd

    def channel(self, sd):
        return self._chan[sd["calibrated_sensor_token"]]

    def scene(self, name):
        for s in self.scenes:
            if s["name"] == name:
                return s
        raise KeyError(f"no scene {name!r}; try --list")

    def chain(self, scene, channel):
        """Every frame (keyframes + sweeps) of one channel within a scene, in time order."""
        first = self.key.get((scene["first_sample_token"], channel))
        if first is None:
            raise KeyError(f"{scene['name']}: no {channel} keyframe at the first sample")
        t_end = self.sample_ts[scene["last_sample_token"]]
        out, sd = [], first
        while sd is not None:
            if self.sample_scene.get(sd["sample_token"]) != scene["token"] or sd["timestamp"] > t_end:
                break
            out.append(sd)
            sd = self.sample_data.get(sd["next"]) if sd["next"] else None
        return out

    # ---- geometry
    @staticmethod
    def quat_to_R(q):
        w, x, y, z = np.asarray(q, np.float64) / np.linalg.norm(q)
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

    @classmethod
    def rigid(cls, rec):
        T = np.eye(4)
        T[:3, :3] = cls.quat_to_R(rec["rotation"])
        T[:3, 3] = rec["translation"]
        return T

    def sensor2global(self, sd):
        """T_global<-sensor at this sample_data's own timestamp."""
        T_ego_sensor = self.rigid(self.calib[sd["calibrated_sensor_token"]])
        T_global_ego = self.rigid(self.ego_pose[sd["ego_pose_token"]])
        return T_global_ego @ T_ego_sensor

    def intrinsics(self, sd):
        K = np.asarray(self.calib[sd["calibrated_sensor_token"]]["camera_intrinsic"], np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"{self.channel(sd)} has no 3x3 camera_intrinsic")
        return K


# ---------------------------------------------------------------------------- clip
def match_one_to_one(t_ref, ts):
    """Match each reference time to a distinct frame of another camera, closest pairs first.
    Returns (index into ts or -1, offset in ms or inf) per reference row."""
    cand = []
    for r, t in enumerate(t_ref):
        i = int(np.searchsorted(ts, t))
        for j in range(max(i - 2, 0), min(i + 2, len(ts))):
            cand.append((abs(int(ts[j]) - int(t)), r, j))
    cand.sort()
    row = np.full(len(t_ref), -1, np.int64)
    used = set()
    for _, r, j in cand:
        if row[r] < 0 and j not in used:
            row[r] = j
            used.add(j)
    dt = np.where(row >= 0, (ts[np.clip(row, 0, None)] - t_ref) / 1e3, np.inf)
    return row, dt


def build_clip(nus, scene_name, start_sec=0.0, num_frames=41, input_cam="CAM_FRONT",
               cameras=SYNCED, max_dt_ms=45.0, max_gap_ms=110.0):
    scene = nus.scene(scene_name)
    if input_cam not in cameras:
        cameras = (input_cam,) + tuple(cameras)
    chains = {c: nus.chain(scene, c) for c in cameras}
    base = chains[input_cam]
    t_all = np.array([sd["timestamp"] for sd in base], np.int64)
    worst = np.zeros(len(t_all))
    rows, dts = {}, {}
    for c in cameras:
        if c == input_cam:
            continue
        ts = np.array([sd["timestamp"] for sd in chains[c]], np.int64)
        rows[c], dts[c] = match_one_to_one(t_all, ts)
        worst = np.maximum(worst, np.abs(dts[c]))
    ok = worst <= max_dt_ms
    gap_ok = np.r_[True, np.diff(t_all) / 1e3 <= max_gap_ms]   # gap into row r

    i_start = int(np.searchsorted(t_all, t_all[0] + start_sec * 1e6))
    i0 = None
    for i in range(i_start, len(t_all) - num_frames + 1):
        if ok[i:i + num_frames].all() and gap_ok[i + 1:i + num_frames].all():
            i0 = i
            break
    if i0 is None:
        best = cur = 0
        for v in ok:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        bad = [c for c in rows if (np.abs(dts[c][i_start:]) > max_dt_ms).mean() > 0.2]
        raise ValueError(
            f"{scene_name}: no {num_frames}-frame window from {start_sec}s with every camera "
            f"within {max_dt_ms:.0f} ms of {input_cam} (longest synced run: {best} rows). "
            + (f"Cameras out of sync with {input_cam}: {bad} -- drop them from --cameras. "
               if bad else "Try an earlier --start-sec. ")
            + f"Synced set: {', '.join(SYNCED)}.")

    win = range(i0, i0 + num_frames)
    picked = {input_cam: [base[i] for i in win]}
    report = {}
    for c in rows:
        idx = rows[c][i0:i0 + num_frames]
        assert (np.diff(idx) > 0).all(), f"{c}: matched frames out of order"
        picked[c] = [chains[c][j] for j in idx]
        d = np.abs(dts[c][i0:i0 + num_frames])
        report[c] = (float(d.mean()), float(d.max()))
    t_win = t_all[i0:i0 + num_frames]
    gaps = np.diff(t_win) / 1e3
    fps = 1e6 * (num_frames - 1) / float(t_win[-1] - t_win[0])
    info = {"start_sec": float((t_win[0] - t_all[0]) / 1e6),
            "gap_ms_min_median_max": [float(gaps.min()), float(np.median(gaps)), float(gaps.max())]}
    return scene, picked, report, fps, info


def relative_view(c2w_a, c2w_b):
    """Angle between optical axes (deg) and camera-centre baseline (m) at one instant."""
    cos = np.clip(np.dot(c2w_a[:3, 2], c2w_b[:3, 2]), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos))), float(np.linalg.norm(c2w_a[:3, 3] - c2w_b[:3, 3]))


def write_clip(nus, scene, picked, out, input_cam, fps, report, info, target_cam=None):
    cams = {}
    for c, sds in picked.items():
        cams[c] = dict(K=np.stack([nus.intrinsics(sd) for sd in sds]),
                       c2w=np.stack([nus.sensor2global(sd) for sd in sds]),
                       frame_paths=[os.path.join(nus.root, sd["filename"]) for sd in sds],
                       hw=(picked[c][0]["height"], picked[c][0]["width"]))
    order = [input_cam] + [c for c in CAMERAS if c in cams and c != input_cam]
    cams = {c: cams[c] for c in order}
    geom = {c: relative_view(cams[input_cam]["c2w"][0], cams[c]["c2w"][0]) for c in order[1:]}
    if target_cam is None:   # default target = the widest view from the input
        target_cam = max(geom, key=lambda c: geom[c][0])
    extra = {"nuscenes": {
        "scene": scene["name"], "description": scene.get("description", ""),
        "timestamps_us": {c: [sd["timestamp"] for sd in sds] for c, sds in picked.items()},
        "sync_offset_ms_mean_max": report,
        "frame_gap_ms_min_median_max": info["gap_ms_min_median_max"],
        "clip_start_sec": info["start_sec"],
        "angle_deg_baseline_m_at_frame0": geom}}
    write_scene(out, cams, roles={"input": input_cam, "target": target_cam},
                scale_factor=DRIVING_SCALE, framerate=round(fps, 2),
                name="nuscenes", extra=extra)
    return geom


# ------------------------------------------------------------------------- overlay
def lidar_points_global(nus, scene, t_us):
    """LIDAR_TOP keyframe nearest t_us, as (N,3) global points."""
    best = None
    for (sample_tok, ch), sd in nus.key.items():
        if ch == "LIDAR_TOP" and nus.sample_scene.get(sample_tok) == scene["token"]:
            if best is None or abs(sd["timestamp"] - t_us) < abs(best["timestamp"] - t_us):
                best = sd
    if best is None:
        raise KeyError("no LIDAR_TOP keyframe in this scene")
    pts = np.fromfile(os.path.join(nus.root, best["filename"]), dtype=np.float32).reshape(-1, 5)
    P = np.c_[pts[:, :3].astype(np.float64), np.ones(len(pts))]
    return (nus.sensor2global(best) @ P.T).T[:, :3], best["timestamp"]


def project(points_world, c2w, K, hw, near=1.0):
    """World points -> pixel coords + depth, for points in front and inside the image."""
    w2c = np.linalg.inv(np.asarray(c2w, np.float64))
    pc = (w2c[:3, :3] @ points_world.T).T + w2c[:3, 3]
    keep = pc[:, 2] > near
    pc = pc[keep]
    uv = (np.asarray(K, np.float64) @ pc.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    H, W = hw
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    return uv[inside], pc[inside, 2]


def overlay(nus, scene, episode):
    """Draw LIDAR_TOP points, projected through the SAVED episode matrices, on frame 0 of
    every camera. Uses only what infer.py will read, so it validates the episode itself."""
    import imageio.v2 as imageio

    with open(os.path.join(episode, "metadata.json")) as f:
        meta = json.load(f)
    spec = meta["specific"]
    offset = np.asarray(spec.get("world_offset", [0.0, 0.0, 0.0]))
    t_us = spec["nuscenes"]["timestamps_us"][spec["roles"]["input"]][0]
    pts, t_lidar = lidar_points_global(nus, scene, t_us)
    pts = pts - offset
    os.makedirs(os.path.join(episode, "_check"), exist_ok=True)
    counts = {}
    for cam in meta["cameras"]:
        d = np.load(os.path.join(episode, "lowdim", cam, "0000000000.npz"))
        img = np.asarray(imageio.imread(os.path.join(episode, "rgb", cam, "0000000000.jpg")))[..., :3].copy()
        uv, z = project(pts, d["extrinsics"], d["intrinsics"], img.shape[:2])
        # near = bright, far = dark; a 3x3 dot, white-ringed so it reads on any background
        v = np.clip(1.0 - (z - 2.0) / 40.0, 0.15, 1.0)
        col = np.stack([255 * v, 64 * np.ones_like(v), 255 * (1 - v)], 1).astype(np.uint8)
        for (u, w), c in zip(uv.astype(int), col):
            img[max(w - 1, 0):w + 2, max(u - 1, 0):u + 2] = c
        imageio.imwrite(os.path.join(episode, "_check", f"{cam}.jpg"), img, quality=90)
        counts[cam] = len(uv)
    dt = (t_lidar - t_us) / 1e3
    return counts, dt


# ----------------------------------------------------------------------------- CLI
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--list", action="store_true", help="list scenes with images on disk, and exit")
    p.add_argument("--all", action="store_true", help="with --list: include scenes without images")
    p.add_argument("--scene", help="scene name, e.g. scene-0061")
    p.add_argument("--start-sec", type=float, default=0.0)
    p.add_argument("--num-frames", type=int, default=41, help="1+4k, <= 41 for infer.py")
    p.add_argument("--input-cam", default="CAM_FRONT", choices=CAMERAS)
    p.add_argument("--cameras", default=",".join(SYNCED),
                   help="comma list, or 'all' (default: the four that share a trigger pattern)")
    p.add_argument("--target-cam", default=None, choices=CAMERAS,
                   help="default role for infer.py (default: widest view from the input)")
    p.add_argument("--max-dt-ms", type=float, default=45.0,
                   help="max |time offset| of any camera from the input, per row")
    p.add_argument("--max-gap-ms", type=float, default=110.0,
                   help="max gap between consecutive frames (catches dropped frames)")
    p.add_argument("--out")
    p.add_argument("--overlay", action="store_true", help="write lidar-overlay check images")
    a = p.parse_args()

    if a.list:
        nus = NuScenes(a.dataroot, a.version, channels={"CAM_FRONT"}, need_pose=False)
        n_have = 0
        for s in nus.scenes:
            ch = nus.chain(s, "CAM_FRONT")
            dur = (ch[-1]["timestamp"] - ch[0]["timestamp"]) / 1e6
            have = all(os.path.isfile(os.path.join(nus.root, sd["filename"])) for sd in (ch[0], ch[-1]))
            n_have += have
            if have or a.all:
                mark = "images" if have else "------"
                print(f"{s['name']:12s} {mark}  {len(ch):4d} front frames  {dur:5.1f}s  "
                      f"{s.get('description', '')}")
        print(f"{n_have} of {len(nus.scenes)} scenes have images on disk"
              + ("" if a.all else " (--all to list every scene)"))
        return
    if not (a.scene and a.out):
        p.error("--scene and --out are required (or use --list)")
    cams = CAMERAS if a.cameras == "all" else tuple(c.strip() for c in a.cameras.split(","))
    unknown = set(cams) - set(CAMERAS)
    if unknown:
        p.error(f"unknown camera(s) {sorted(unknown)}; choose from {CAMERAS}")
    if a.input_cam not in cams:
        cams = (a.input_cam,) + cams
    if a.target_cam and a.target_cam not in cams:
        p.error(f"--target-cam {a.target_cam} is not in --cameras")
    nus = NuScenes(a.dataroot, a.version, scenes={a.scene},
                   channels=set(cams) | ({"LIDAR_TOP"} if a.overlay else set()))
    if (a.num_frames - 1) % 4 or a.num_frames > 41:
        print(f"warning: --num-frames {a.num_frames} is not 1+4k <= 41; infer.py will trim it")

    scene, picked, report, fps, info = build_clip(nus, a.scene, a.start_sec, a.num_frames,
                                                  a.input_cam, cameras=cams,
                                                  max_dt_ms=a.max_dt_ms, max_gap_ms=a.max_gap_ms)
    geom = write_clip(nus, scene, picked, a.out, a.input_cam, fps, report, info,
                      target_cam=a.target_cam)
    g = info["gap_ms_min_median_max"]
    print(f"wrote {a.out}: {a.scene} from {info['start_sec']:.2f}s, {a.num_frames} frames, "
          f"{fps:.1f} fps average (gaps {g[0]:.0f}/{g[1]:.0f}/{g[2]:.0f} ms min/median/max), "
          f"input {a.input_cam}")
    print(f"  {'target':16s} {'angle':>7s} {'baseline':>9s} {'sync mean/max':>15s}")
    for c, (ang, base) in geom.items():
        m, mx = report[c]
        print(f"  {c:16s} {ang:6.1f}d {base:8.2f}m {m:7.1f}/{mx:4.0f} ms")
    if a.overlay:
        counts, dt = overlay(nus, scene, a.out)
        print(f"  overlay: lidar {dt:+.0f} ms from frame 0; points per camera {counts} "
              f"-> {os.path.join(a.out, '_check')}")


if __name__ == "__main__":
    main()
