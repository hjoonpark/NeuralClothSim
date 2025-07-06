import struct, numpy as np, os, trimesh, pathlib

# ------------------------------------------------------------------
# 1. Low-level PC2 reader  (makes a writeable copy of the data!)
# ------------------------------------------------------------------
def read_pc2(path: str):
    with open(path, 'rb') as f:
        if not f.read(12).decode().startswith("POINTCACHE2"):
            raise ValueError(f"{path}: bad PC2 signature")

        version, n_pts = struct.unpack('<ii', f.read(8))
        start, rate    = struct.unpack('<ff', f.read(8))
        n_frames       = struct.unpack('<i',  f.read(4))[0]

        raw = np.frombuffer(f.read(n_frames * n_pts * 12), np.float32)
        pos = raw.reshape(n_frames, n_pts, 3).copy()   # <-- .copy() → writable

    return dict(positions=pos, num_points=n_pts, num_frames=n_frames)

# ------------------------------------------------------------------
# 2. OBJ writer
# ------------------------------------------------------------------
def write_obj(path, vertices, faces):
    with open(path, 'w') as f:
        f.writelines(f"v {x} {y} {z}\n" for x, y, z in vertices)
        f.writelines("f " + " ".join(str(i+1) for i in face) + "\n" for face in faces)

# ------------------------------------------------------------------
# 3. Main converter with root-motion transfer
# ------------------------------------------------------------------
def convert_pair(body_pc2_path, shirt_pc2_path,
                 body_obj_path, shirt_obj_path,
                 out_body_dir, out_shirt_dir):

    # --- read caches ---
    body_pc2  = read_pc2(body_pc2_path)
    shirt_pc2 = read_pc2(shirt_pc2_path)

    # --- load static meshes ---
    body_mesh  = trimesh.load(body_obj_path,  process=False)
    shirt_mesh = trimesh.load(shirt_obj_path, process=False)

    if len(body_mesh.vertices)  != body_pc2['num_points']:
        raise ValueError("vertex-count mismatch for body")
    if len(shirt_mesh.vertices) != shirt_pc2['num_points']:
        raise ValueError("vertex-count mismatch for shirt")

    n_frames = shirt_pc2['num_frames']
    if n_frames != body_pc2['num_frames']:
        raise ValueError("body and shirt have different frame counts")

    # --- compute per-frame translation of the shirt (world vs. object) ---
    shirt_static_centroid = shirt_mesh.vertices.mean(axis=0)
    shirt_frame_centroid  = shirt_pc2['positions'].mean(axis=1)     # (F, 3)
    deltas = shirt_frame_centroid - shirt_static_centroid           # (F, 3)

    print(f"[INFO] first 5 Δ translations:\n{deltas[:5]}")

    # --- add those deltas to body vertices ---
    body_pc2['positions'] += deltas[:, None, :]   # broadcast to all body verts

    # --- create output dirs ---
    pathlib.Path(out_body_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(out_shirt_dir).mkdir(parents=True, exist_ok=True)

    # --- write sequences ---
    for f in range(n_frames):
        write_obj(os.path.join(out_body_dir,  f"frame_{f:04d}.obj"),
                  body_pc2['positions'][f],  body_mesh.faces)
        write_obj(os.path.join(out_shirt_dir, f"frame_{f:04d}.obj"),
                  shirt_pc2['positions'][f], shirt_mesh.faces)

    print("✅ Finished: body now follows shirt root motion.")

# ------------------------------------------------------------------
# 4.  Run with your exact paths
# ------------------------------------------------------------------
if __name__ == "__main__":
    convert_pair(
        body_pc2_path  = "/nobackup/joon/1_Projects/NeuralClothSim/results/smpl/body.pc2",
        shirt_pc2_path = "/nobackup/joon/1_Projects/NeuralClothSim/results/smpl/tshirt.pc2",
        body_obj_path  = "/nobackup/joon/1_Projects/NeuralClothSim/body_models/smpl_female_neutral/body.obj",
        shirt_obj_path = "/nobackup/joon/1_Projects/NeuralClothSim/body_models/smpl_female_neutral/tshirt.obj",
        out_body_dir   = "../results/smpl/body",
        out_shirt_dir  = "../results/smpl/tshirts"
    )
