import bpy, os, sys, math, json, csv, traceback
from mathutils import Vector, Matrix
import datetime as _dt

TIMESTAMP         = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_ROOT             = os.path.expanduser(f"/home/tigersec-alkim/Desktop/MultiEV/experiments/parameter_sweeps/full_sweep_coarse_spatial_illumination_micro_fps_{TIMESTAMP}")
NUM_HALTON_COARSE    = 64
NUM_HALTON_MICRO     = 96
TOP_K_FOR_MICRO      = 16
INTEGRATED_FRAMES_T  = 12
K_SUBSAMPLES         = 6
YOLO_MODEL_PATH      = os.environ.get("YOLO11_MODEL", "yolo11n.pt")
YOLO_CONF            = 0.25
YOLO_IMGSZ           = 640
SCORE_ALPHA          = 0.5

FLASHLIGHT_COLLECTION_KEYS = ["flashlight", "flashlight light"]
TARGET_MATERIAL_NAME = "Material.001"
FLARED_MOD_NAME      = "Flared2"
SOCKET_90_NAME       = "Socket_90"
SOCKET_142_NAME      = "Socket_142"
S90_RANGE            = (12.0, 20.0)
S142_RANGE           = (0.0, 0.75)

DEFAULTS = {
    "flashlight_x":   -26.8715,
    "flashlight_z":     1.75988,
    "car_y":           33.0105,
    "incidence_z_deg": 180.0,
    "sun_wm2":          3.0,
    "flash_strength":  300.0,
}

RANGES = {
    "flashlight_x":   (-27.3836, -19.7505),
    "flashlight_z":   (  0.201205, 4.05466),
    "car_y":          ( 32.5423,  44.0463),
    "incidence_z_deg": (-270.0,    -90.0),
    "sun_wm2":        (  1.0,      10.0),
    "flash_strength": (100.0,    1000.0),
}

TEMPORAL_DEFAULTS = {
    "fps":               24.0,
    "shutter_time_ms":   20.0,
    "flash_freq_hz":      8.0,
    "duty_cycle":         0.15,
    "phase":              0.0,
    "K_subsamples":       K_SUBSAMPLES,
    "T_frames":           INTEGRATED_FRAMES_T,
}

TEMPORAL_RANGES = {
    "fps":             (18.0, 30.0),
    "shutter_time_ms": ( 5.0, 35.0),
    "flash_freq_hz":   ( 2.0, 20.0),
    "duty_cycle":      ( 0.05, 0.5),
    "phase":           ( 0.0, 1.0),
}

PARAM_KEYS = [
    "flashlight_x","flashlight_z","car_y","incidence_z_deg",
    "sun_wm2","flash_strength",
    "fps","shutter_time_ms","flash_freq_hz","duty_cycle","phase",
]

SPATIAL_KEYS      = ["flashlight_x","flashlight_z","car_y","incidence_z_deg"]
ILLUM_KEYS        = ["sun_wm2","flash_strength"]
CAMERA_KEYS       = ["fps"]
OTHER_TEMP_KEYS   = ["shutter_time_ms","flash_freq_hz","duty_cycle","phase"]

os.makedirs(OUT_ROOT, exist_ok=True)

import sys

LOG_PATH = os.path.join(OUT_ROOT, "autopilot_log.txt")

log_file = open(LOG_PATH, "w", buffering=1)

class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self._streams:
            s.flush()

_orig_stdout = sys.stdout
_orig_stderr = sys.stderr

sys.stdout = _Tee(_orig_stdout, log_file)
sys.stderr = _Tee(_orig_stderr, log_file)

def enable_gpu_for_cycles(prefer=("CUDA","OPTIX","HIP","METAL","ONEAPI")):
    prefs = bpy.context.preferences
    if "cycles" not in prefs.addons:
        print(" Cycles addon not found; CPU fallback.")
        return False
    cp = prefs.addons["cycles"].preferences
    supported = []
    for b in ("CUDA","OPTIX","HIP","METAL","ONEAPI"):
        try:
            cp.compute_device_type = b
            if cp.compute_device_type == b:
                supported.append(b)
        except Exception:
            pass
    if not supported:
        print(" No GPU backend supported by this Blender build.")
        return False
    selected = None
    for b in prefer:
        if b in supported:
            try:
                cp.compute_device_type = b
                if cp.compute_device_type == b:
                    selected = b
                    break
            except Exception:
                continue
    if not selected:
        print(" Could not set preferred backend; CPU fallback.")
        return False
    try:
        cp.get_devices()
        devs = []
        for d in cp.devices:
            d.use = True
            if getattr(d,"use",False):
                devs.append(d.name)
        if not devs:
            print(f" No devices enabled under {selected}.")
            return False
        bpy.context.scene.cycles.device = "GPU"
        print(f" Cycles GPU backend: {selected}; devices: {devs}")
        return True
    except Exception as e:
        print(f" Device enumeration failed: {e}")
        return False

def make_fast_render_settings():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    gpu_ok = enable_gpu_for_cycles()
    print(" Rendering on GPU." if gpu_ok else " Rendering on CPU.")

    scene.cycles.samples = 3
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.1
    scene.cycles.max_bounces = 2
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transmission_bounces = 1
    scene.cycles.volume_bounces = 0
    scene.cycles.transparent_max_bounces = 1
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = 'Raw'
    scene.view_settings.exposure = 0.0
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_depth = '16'
    scene.render.image_settings.exr_codec = 'ZIP'
    scene.render.use_motion_blur = False
    print(" Fast render settings applied.")

def disable_compositor_clamping():
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    changed = 0
    for n in tree.nodes:
        if hasattr(n, "use_clamp") and n.use_clamp:
            n.use_clamp = False
            changed += 1
    print(f" Compositor clamp disabled on {changed} node(s).")

def find_collections_by_keys(keys):
    keys = [k.lower() for k in keys]
    hits = []
    for col in bpy.data.collections:
        name_l = col.name.lower()
        if any(k in name_l for k in keys):
            hits.append(col)
    return hits

def collect_objects_recursive(col):
    objs = list(col.objects)
    for child in col.children:
        objs.extend(collect_objects_recursive(child))
    return objs

def collect_group_objects(keys):
    cols = find_collections_by_keys(keys)
    if not cols:
        raise RuntimeError(f"No collections found matching any of: {keys}")
    objs = []
    for c in cols:
        objs.extend(collect_objects_recursive(c))
    seen = set()
    out = []
    for o in objs:
        if o.as_pointer() not in seen:
            out.append(o); seen.add(o.as_pointer())
    return out

def has_emission_node(material):
    if not material or not material.use_nodes:
        return False
    return any(node.type == 'EMISSION' for node in material.node_tree.nodes)

def find_emission_mesh_in_objects(objs):
    for obj in objs:
        if obj.type == 'MESH':
            for ms in obj.material_slots:
                if has_emission_node(ms.material):
                    return obj, ms.material
    return None, None

def find_light_in_objects(objs):
    for obj in objs:
        if obj.type == 'LIGHT' and obj.data and obj.data.type in {'POINT', 'SPOT', 'AREA'}:
            return obj
    return None

def get_flashlight_group_and_main():
    group_objs = collect_group_objects(FLASHLIGHT_COLLECTION_KEYS)
    emissive_mesh, emissive_mat = find_emission_mesh_in_objects(group_objs)
    light_obj = find_light_in_objects(group_objs)
    if light_obj:
        main_obj = light_obj
    elif emissive_mesh:
        main_obj = emissive_mesh
    else:
        meshes = [o for o in group_objs if o.type == 'MESH']
        if not meshes:
            if not group_objs:
                raise RuntimeError("No objects found in flashlight groups.")
            main_obj = group_objs[0]
        else:
            def vol(o):
                d = getattr(o, "dimensions", None)
                if not d: return 0.0
                return max(d.x,1e-6)*max(d.y,1e-6)*max(d.z,1e-6)
            main_obj = max(meshes, key=vol)
    return group_objs, main_obj, emissive_mat, light_obj

def get_car_collection_objects():
    for col in bpy.data.collections:
        if "car" in col.name.lower():
            return collect_objects_recursive(col)
    return []

def get_car_core_objects():
    obs = get_car_collection_objects()
    return [o for o in obs if o.type != 'CAMERA']

def get_sun_light():
    for light in bpy.data.lights:
        if light.type == 'SUN':
            return light
    return None

def force_quaternion_mode(objs):
    for o in objs:
        try: o.rotation_mode = 'QUATERNION'
        except Exception: pass

def world_loc(obj) -> Vector:
    return obj.matrix_world.translation.copy()

def world_yaw_from_matrix(mw: Matrix) -> float:
    R = mw.to_3x3()
    x_axis = R @ Vector((1,0,0))
    y_axis = R @ Vector((0,1,0))
    x_xy = Vector((x_axis.x, x_axis.y))
    y_xy = Vector((y_axis.x, y_axis.y))
    v = y_xy if y_xy.length >= x_xy.length else x_xy
    if v.length < 1e-12: return 0.0
    return math.atan2(v.y, v.x)

def apply_group_rigid_transform_world(objs, pivot_world: Vector, dtrans_world: Vector, dyaw_rad: float):
    R = Matrix.Rotation(dyaw_rad, 4, 'Z')
    T_p = Matrix.Translation(pivot_world)
    T_np = Matrix.Translation(-pivot_world)
    T_d = Matrix.Translation(dtrans_world)
    for o in objs:
        o.matrix_world = T_d @ T_p @ R @ T_np @ o.matrix_world
    bpy.context.view_layer.update()

def set_flashlight_absolute_world(group_objs, main_obj, target_x, target_z, target_z_deg):
    force_quaternion_mode(group_objs)
    pivot = world_loc(main_obj)
    curr_yaw = world_yaw_from_matrix(main_obj.matrix_world)
    target_yaw = math.radians(target_z_deg)
    dyaw = target_yaw - curr_yaw
    dtrans = Vector((target_x - pivot.x, 0.0, target_z - pivot.z))
    apply_group_rigid_transform_world(group_objs, pivot, dtrans, dyaw)
    new_pivot = world_loc(main_obj)
    print(f"[flashlight(union)] xz({new_pivot.x:.3f},{new_pivot.z:.3f}) yaw={math.degrees(world_yaw_from_matrix(main_obj.matrix_world)):.2f}")

def set_car_absolute_y_world(target_y):
    car_objs = get_car_core_objects()
    if not car_objs: return
    centroid = Vector((0.0, 0.0, 0.0))
    for o in car_objs:
        centroid += world_loc(o)
    centroid /= len(car_objs)
    dy = target_y - centroid.y
    if abs(dy) < 1e-12: return
    dtrans = Vector((0.0, dy, 0.0))
    for o in car_objs:
        mw = o.matrix_world.copy()
        mw.translation += dtrans
        o.matrix_world = mw
    bpy.context.view_layer.update()

def set_sun_irradiance_wm2(wm2_value):
    sun = get_sun_light()
    if sun:
        sun.energy = float(wm2_value)

def get_material(name: str):
    mat = bpy.data.materials.get(name)
    if not mat:
        raise RuntimeError(f"Material '{name}' not found.")
    if not mat.use_nodes:
        mat.use_nodes = True
    return mat

def set_principled_emission_strength(mat, value: float) -> int:
    v = float(value); set_count = 0
    nt = mat.node_tree
    for n in nt.nodes:
        if n.type == 'BSDF_PRINCIPLED':
            try:
                n.inputs[28].default_value = v
                set_count += 1
            except Exception:
                sock = n.inputs.get('Emission Strength')
                if sock is not None:
                    sock.default_value = v; set_count += 1
    return set_count

def find_modifier_by_name(mod_name: str):
    for obj in bpy.data.objects:
        mod = obj.modifiers.get(mod_name)
        if mod is not None:
            return obj, mod
    raise RuntimeError(f"Modifier '{mod_name}' not found on any object.")

def clamp(x, lo, hi): return max(lo, min(hi, x))
def lerp(a, b, t):    return a + (b - a) * t

def map_power_to_sockets(power: float):
    pmin, pmax = RANGES["flash_strength"]
    t = 0.0 if pmax <= pmin else (float(power) - pmin) / (pmax - pmin)
    t = clamp(t, 0.0, 1.0)
    v90  = lerp(S90_RANGE[0],  S90_RANGE[1],  t)
    v142 = lerp(S142_RANGE[0], S142_RANGE[1], t)
    return clamp(v90, *S90_RANGE), clamp(v142, *S142_RANGE)

def set_flared_sockets(v90: float, v142: float):
    obj, mod = find_modifier_by_name(FLARED_MOD_NAME)
    mod[SOCKET_90_NAME]  = float(v90)
    mod[SOCKET_142_NAME] = float(v142)
    return obj.name

def set_flashlight_emission_and_glare(base_strength_value: float, scale: float = 1.0):
    current = float(base_strength_value) * float(scale)
    mat = get_material(TARGET_MATERIAL_NAME)
    set_principled_emission_strength(mat, current)
    s90, s142 = map_power_to_sockets(current)
    _ = set_flared_sockets(s90, s142)
    bpy.context.view_layer.update()
    return current, s90, s142

def square_wave_intensity(t, freq_hz, duty, phase):
    if freq_hz <= 0.0: return 1.0
    period = 1.0 / freq_hz
    tau = (t + phase * period) % period
    return 1.0 if tau < (duty * period) else 0.0

def frame_time_bounds(frame_index, fps, shutter_time_ms):
    frame_center = frame_index / fps
    dt = (shutter_time_ms / 1000.0) * 0.5
    return (frame_center - dt, frame_center + dt)

def temporal_supersample_times(frame_index, fps, shutter_time_ms, K):
    t0, t1 = frame_time_bounds(frame_index, fps, shutter_time_ms)
    if K <= 1: return [0.5 * (t0 + t1)]
    step = (t1 - t0) / K
    return [t0 + (i + 0.5) * step for i in range(K)]

def load_pixels_from_file(path):
    img = bpy.data.images.load(path)
    try:
        try: img.colorspace_settings.name = 'Linear'
        except Exception: pass
        img.update()
        pixels = list(img.pixels[:])
        return pixels
    finally:
        bpy.data.images.remove(img, do_unlink=True)

def average_pixel_buffers(buffers):
    if not buffers: return []
    n = len(buffers[0])
    out = [0.0] * n
    K = float(len(buffers))
    for buf in buffers:
        if len(buf) != n:
            raise RuntimeError(f"Inconsistent buffer lengths: {len(buf)} vs {n}")
        for i in range(n):
            out[i] += buf[i]
    invK = 1.0 / K
    for i in range(n):
        out[i] *= invK
    return out

def save_pixels_as_exr(pixels, width, height, path_no_ext):
    name = "IntegratedFrameTemp"
    img = bpy.data.images.get(name)
    if img is None or img.size[0] != width or img.size[1] != height:
        if img is not None:
            bpy.data.images.remove(img)
        img = bpy.data.images.new(name=name, width=width, height=height, alpha=True, float_buffer=True)
    img.pixels.foreach_set(pixels)
    img.filepath_raw = path_no_ext
    img.file_format = 'OPEN_EXR'
    img.save_render(path_no_ext)
    return path_no_ext

def render_integrated_frame(frame_idx, base_power, temporal_params, out_dir, tag_prefix):
    scene = bpy.context.scene
    width, height = scene.render.resolution_x, scene.render.resolution_y
    fps             = temporal_params["fps"]
    shutter_time_ms = temporal_params["shutter_time_ms"]
    flash_freq_hz   = temporal_params["flash_freq_hz"]
    duty_cycle      = temporal_params["duty_cycle"]
    phase           = temporal_params["phase"]
    K               = int(temporal_params["K_subsamples"])

    sub_times = temporal_supersample_times(frame_idx, fps, shutter_time_ms, K)
    sub_buffers, sub_paths = [], []

    for k, t_abs in enumerate(sub_times):
        scale = square_wave_intensity(t_abs, flash_freq_hz, duty_cycle, phase)
        _current, _s90, _s142 = set_flashlight_emission_and_glare(base_power, scale)
        sub_path = os.path.join(out_dir, f"{tag_prefix}_frame{frame_idx:03d}_sub{k:02d}.exr")
        scene.render.filepath = sub_path
        bpy.ops.render.render(write_still=True)
        buf = load_pixels_from_file(sub_path)
        sub_buffers.append(buf); sub_paths.append(sub_path)

    avg = average_pixel_buffers(sub_buffers)
    out_path = os.path.join(out_dir, f"{tag_prefix}_frame{frame_idx:03d}_INTEGRATED.exr")
    save_pixels_as_exr(avg, width, height, out_path)
    return out_path, sub_paths

def _halton_single(index, base):
    f = 1.0; r = 0.0; i = index
    while i > 0:
        f = f / base
        r = r + f * (i % base)
        i = i // base
    return r

def halton_sequence(dim, n, start_index=1, bases=None):
    default_primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71]
    if bases is None:
        bases = default_primes[:dim]
    return [[_halton_single(t, bases[d]) for d in range(dim)]
            for t in range(start_index, start_index+n)]

def sample_from_range(u, lo, hi): return lo + u * (hi - lo)

_YOLO = None
def _ensure_pip_and_install(pkg):
    try:
        import importlib
        importlib.import_module(pkg)
        return True
    except Exception:
        pass
    try:
        import ensurepip, subprocess
        ensurepip.bootstrap()
        py = sys.executable
        subprocess.check_call([py, "-m", "pip", "install", "-U", pkg])
        import importlib
        importlib.import_module(pkg)
        return True
    except Exception as e:
        print(f" Could not install {pkg}: {e}")
        return False

def _ensure_ultralytics():
    global _YOLO
    if _YOLO is not None:
        return _YOLO
    ok = _ensure_pip_and_install("ultralytics")
    if not ok:
        print(" Ultralytics unavailable. Skipping J.")
        return None
    try:
        from ultralytics import YOLO
        try:
            _YOLO = YOLO(YOLO_MODEL_PATH)
        except Exception:
            _YOLO = YOLO("yolo11n.pt")
        print(" YOLOv11 ready.")
        return _YOLO
    except Exception as e:
        print(f" Failed to load YOLOv11: {e}")
        _YOLO = None
        return None

def _exr_to_rgb8(path):
    import numpy as np
    img = bpy.data.images.load(path)
    try:
        try: img.colorspace_settings.name = 'Linear'
        except Exception: pass
        img.update()
        w,h = img.size
        buf = img.pixels[:]
        import array
        a = array.array('f', buf)
        rgb = []
        for i in range(0, len(a), 4):
            r,g,b = a[i], a[i+1], a[i+2]
            r = 0.0 if r<0 else (1.0 if r>1 else r)
            g = 0.0 if g<0 else (1.0 if g>1 else g)
            b = 0.0 if b<0 else (1.0 if b>1 else b)
            r = r**(1/2.2); g = g**(1/2.2); b = b**(1/2.2)
            rgb.extend((int(r*255+0.5), int(g*255+0.5), int(b*255+0.5)))
        arr = np.frombuffer(bytes(rgb), dtype=np.uint8)
        arr = arr.reshape(h, w, 3)
        return arr
    finally:
        bpy.data.images.remove(img, do_unlink=True)

def _score_frame_stop_sign(rgb8, yolo):
    h, w = rgb8.shape[:2]
    img_area = float(h*w)
    try:
        res = yolo.predict(source=[rgb8], imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False)
    except Exception as e:
        print(f" YOLO inference failed: {e}")
        return 0.0
    if not res:
        return 0.0
    r0 = res[0]
    boxes = getattr(r0, "boxes", None)
    names = getattr(yolo, "names", None)
    if boxes is None:
        return 0.0
    try:
        import numpy as np
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        clsi = boxes.cls.cpu().numpy().astype(int)
    except Exception:
        return 0.0
    best = 0.0
    for k in range(xyxy.shape[0]):
        name = None
        if names is not None and 0 <= clsi[k] < len(names):
            name = str(names[clsi[k]]).lower()
        if name is None or "stop" not in name:
            continue
        x1,y1,x2,y2 = xyxy[k]
        area = max(0.0, x2-x1) * max(0.0, y2-y1)
        ar = 0.0 if img_area<=0 else (area/img_area)
        s = float(conf[k]) * (ar ** SCORE_ALPHA)
        if s > best: best = s
    return best

def compute_J_for_sample(integrated_paths):
    yolo = _ensure_ultralytics()
    if yolo is None:
        return None
    scores = []
    for p in integrated_paths:
        if not os.path.exists(p): continue
        try:
            rgb8 = _exr_to_rgb8(p)
            s = _score_frame_stop_sign(rgb8, yolo)
            scores.append(s)
        except Exception as e:
            print(f" frame scoring failed for {p}: {e}")
    if not scores:
        return None
    return sum(scores)/len(scores)

def percentile(xs, p):
    if not xs: return None
    xs = sorted(xs)
    if p<=0: return xs[0]
    if p>=1: return xs[-1]
    i = p*(len(xs)-1)
    lo = int(i); hi = min(len(xs)-1, lo+1)
    if lo==hi: return xs[lo]
    return xs[lo] + (xs[hi]-xs[lo])*(i-lo)

def build_micro_ranges_from_top(samples_rows, stage):
    """
    samples_rows: list of dicts with keys:
      - 'J'
      - 'spatial' (dict)
      - 'temporal' (dict)
    stage: one of {"spatial","illumination","camera"}
    Returns: dim_ranges dict over all PARAM_KEYS
    """
    top = sorted(samples_rows, key=lambda r: r["J"], reverse=True)[:TOP_K_FOR_MICRO]

    vals = {k: [] for k in PARAM_KEYS}
    for r in top:
        s = r["spatial"]; t = r["temporal"]
        for k in SPATIAL_KEYS:
            if k in s: vals[k].append(float(s[k]))
        for k in ILLUM_KEYS:
            if k in s: vals[k].append(float(s[k]))
        if "fps" in t:
            vals["fps"].append(float(t["fps"]))
        for k in OTHER_TEMP_KEYS:
            if k in t: vals[k].append(float(t[k]))

    dim_ranges = {}
    for k in PARAM_KEYS:
        if k in DEFAULTS:
            v = float(DEFAULTS[k])
            dim_ranges[k] = (v, v)
        elif k in RANGES:
            dim_ranges[k] = RANGES[k]
        elif k in TEMPORAL_RANGES:
            dim_ranges[k] = TEMPORAL_RANGES[k]
        else:
            raise RuntimeError(f"No base range/default for '{k}'")

    def freeze_to_median(key):
        if vals[key]:
            m = percentile(vals[key], 0.50)
            dim_ranges[key] = (m, m)

    def narrow_p25_p75(key):
        if vals[key]:
            lo = percentile(vals[key], 0.25)
            hi = percentile(vals[key], 0.75)
            dim_ranges[key] = (lo, hi)

    if stage == "spatial":
        print("🔎 Building SPATIAL micro ranges from coarse top-K...")
        for k in SPATIAL_KEYS:
            narrow_p25_p75(k)

        for k in ILLUM_KEYS:
            freeze_to_median(k)

        freeze_to_median("fps")

        for k in OTHER_TEMP_KEYS:
            freeze_to_median(k)

    elif stage == "illumination":
        print("🔎 Building ILLUMINATION micro ranges from spatial micro top-K...")
        for k in SPATIAL_KEYS:
            freeze_to_median(k)

        for k in ILLUM_KEYS:
            narrow_p25_p75(k)

        freeze_to_median("fps")

        for k in OTHER_TEMP_KEYS:
            freeze_to_median(k)

    elif stage == "camera":
        print("🔎 Building CAMERA micro ranges from illumination micro top-K...")
        for k in SPATIAL_KEYS:
            freeze_to_median(k)

        for k in ILLUM_KEYS:
            freeze_to_median(k)

        for k in OTHER_TEMP_KEYS:
            freeze_to_median(k)

        narrow_p25_p75("fps")

    else:
        raise ValueError(f"Unknown micro stage '{stage}'")

    return dim_ranges

def run_sweep(mode_tag, dim_ranges, num_samples):
    print(f" Starting {mode_tag.upper()} sweep...")
    make_fast_render_settings()
    disable_compositor_clamping()

    group_objs, main_obj, _, _ = get_flashlight_group_and_main()
    set_flashlight_absolute_world(group_objs, main_obj,
                                  DEFAULTS["flashlight_x"],
                                  DEFAULTS["flashlight_z"],
                                  DEFAULTS["incidence_z_deg"])
    set_car_absolute_y_world(DEFAULTS["car_y"])
    set_sun_irradiance_wm2(DEFAULTS["sun_wm2"])
    set_flashlight_emission_and_glare(DEFAULTS["flash_strength"], 1.0)

    for k in PARAM_KEYS:
        if k not in dim_ranges:
            if k in DEFAULTS: dim_ranges[k] = (float(DEFAULTS[k]), float(DEFAULTS[k]))
            elif k in RANGES: dim_ranges[k] = RANGES[k]
            elif k in TEMPORAL_RANGES: dim_ranges[k] = TEMPORAL_RANGES[k]
            else: raise RuntimeError(f"Missing range for {k}")

    halton = halton_sequence(dim=len(PARAM_KEYS), n=num_samples, start_index=1)

    sweep_csv = os.path.join(OUT_ROOT, f"sweep_index__{mode_tag}.csv")
    new_index = not os.path.exists(sweep_csv)
    with open(sweep_csv, "a", newline="") as f:
        w = csv.writer(f)
        if new_index:
            w.writerow(["sample_id","sample_dir","mode"] + PARAM_KEYS + ["integrated_frames_semi"])

        sample_records = []
        for i, uvec in enumerate(halton, start=1):
            params = {}
            for k, u in zip(PARAM_KEYS, uvec):
                lo, hi = dim_ranges[k]
                params[k] = float(lo) if abs(hi-lo) < 1e-12 else sample_from_range(u, lo, hi)

            set_flashlight_absolute_world(group_objs, main_obj,
                                          params["flashlight_x"],
                                          params["flashlight_z"],
                                          params["incidence_z_deg"])
            set_car_absolute_y_world(params["car_y"])
            set_sun_irradiance_wm2(params["sun_wm2"])

            temporal = dict(TEMPORAL_DEFAULTS)
            temporal["fps"]             = params["fps"]
            temporal["shutter_time_ms"] = params["shutter_time_ms"]
            temporal["flash_freq_hz"]   = params["flash_freq_hz"]
            temporal["duty_cycle"]      = params["duty_cycle"]
            temporal["phase"]           = params["phase"]
            base_power = params["flash_strength"]

            out_dir = os.path.join(OUT_ROOT, f"{mode_tag}_sample_{i:04d}")
            os.makedirs(out_dir, exist_ok=True)
            tag = (f"{mode_tag[:1].upper()}{i:04d}"
                   f"__x{params['flashlight_x']:.3f}"
                   f"_z{params['flashlight_z']:.3f}"
                   f"_yaw{params['incidence_z_deg']:.1f}"
                   f"_carY{params['car_y']:.3f}"
                   f"_sun{params['sun_wm2']:.2f}"
                   f"_P{base_power:.1f}"
                   f"_fps{temporal['fps']:.2f}"
                   f"_sh{temporal['shutter_time_ms']:.1f}ms"
                   f"_ff{temporal['flash_freq_hz']:.2f}"
                   f"_dc{temporal['duty_cycle']:.3f}"
                   f"_ph{temporal['phase']:.3f}")

            with open(os.path.join(out_dir, "params.json"), "w") as js:
                json.dump({
                    "mode": mode_tag,
                    "spatial": {
                        "flashlight_x": params["flashlight_x"],
                        "flashlight_z": params["flashlight_z"],
                        "car_y": params["car_y"],
                        "incidence_z_deg": params["incidence_z_deg"],
                        "sun_wm2": params["sun_wm2"],
                        "flash_strength": base_power,
                    },
                    "temporal": temporal,
                    "ranges_used": dim_ranges
                }, js, indent=2)

            frames_csv_path = os.path.join(out_dir, "frames.csv")
            integrated_paths = []
            with open(frames_csv_path, "w", newline="") as fcsv:
                fw = csv.writer(fcsv)
                fw.writerow(["frame_index","integrated_exr","subsample_exrs_semi"])
                T = int(TEMPORAL_DEFAULTS["T_frames"])
                for fidx in range(T):
                    integrated_path, sub_paths = render_integrated_frame(
                        frame_idx=fidx, base_power=base_power,
                        temporal_params=temporal, out_dir=out_dir, tag_prefix=tag
                    )
                    integrated_paths.append(integrated_path)
                    fw.writerow([fidx, integrated_path, ";".join(sub_paths)])
                    print(f"   {mode_tag} {i}/{num_samples} frame {fidx+1}/{T}  {os.path.basename(integrated_path)}")

            J = compute_J_for_sample(integrated_paths)
            if J is not None:
                with open(os.path.join(out_dir, "objective.txt"), "w") as fh:
                    fh.write(f"J = {J}\n")
                sample_records.append({
                    "J": J,
                    "spatial": {
                        "flashlight_x": params["flashlight_x"],
                        "flashlight_z": params["flashlight_z"],
                        "car_y": params["car_y"],
                        "incidence_z_deg": params["incidence_z_deg"],
                        "sun_wm2": params["sun_wm2"],
                        "flash_strength": base_power,
                    },
                    "temporal": temporal,
                    "dir": out_dir
                })

            w.writerow([i, out_dir, mode_tag] + [params[k] for k in PARAM_KEYS] + [";".join(integrated_paths)])

    print(f" {mode_tag.capitalize()} sweep done. Index  {sweep_csv}")
    return sample_records

def main_autopilot():
    try:
        coarse_ranges = {**RANGES, **TEMPORAL_RANGES}
        coarse_rows = run_sweep("coarse", coarse_ranges, NUM_HALTON_COARSE)

        if not coarse_rows:
            print(" No J scores from coarse (YOLO unavailable or no detections). Skipping micro stages.")
            return

        ranges_spatial = build_micro_ranges_from_top(coarse_rows, stage="spatial")
        spatial_rows = run_sweep("micro_spatial", ranges_spatial, NUM_HALTON_MICRO)
        if not spatial_rows:
            print(" Spatial micro sweep produced no J scores. Stopping.")
            return

        ranges_illum = build_micro_ranges_from_top(spatial_rows, stage="illumination")
        illum_rows = run_sweep("micro_illumination", ranges_illum, NUM_HALTON_MICRO)
        if not illum_rows:
            print(" Illumination micro sweep produced no J scores. Stopping.")
            return

        ranges_camera = build_micro_ranges_from_top(illum_rows, stage="camera")
        camera_rows = run_sweep("micro_camera", ranges_camera, NUM_HALTON_MICRO)
        if not camera_rows:
            print(" Camera micro sweep produced no J scores. Stopping.")
            return

        top = sorted(camera_rows, key=lambda r: r["J"], reverse=True)[:5]
        print("\n Top final (camera micro) results:")
        for r in top:
            s = r["spatial"]; t = r["temporal"]
            print(f" J={r['J']:.4f}  flash_strength={s['flash_strength']:.1f}  "
                  f"sun={s['sun_wm2']:.2f}  "
                  f"dc={t['duty_cycle']:.3f}  ff={t['flash_freq_hz']:.2f}  "
                  f"sh={t['shutter_time_ms']:.1f}ms  fps={t['fps']:.1f}  phase={t['phase']:.2f}  "
                  f"x={s['flashlight_x']:.3f} z={s['flashlight_z']:.3f} "
                  f"carY={s['car_y']:.3f} yaw={s['incidence_z_deg']:.1f}")
        print("\n Multi-stage Autopilot complete.")
    except Exception:
        print(" Autopilot failed:\n" + traceback.format_exc())

if __name__ == "__main__":
    main_autopilot()

