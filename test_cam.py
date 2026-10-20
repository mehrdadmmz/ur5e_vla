"""
Live viewer for all connected RealSense cameras (auto-detects N devices).

Tiles color (top row) + depth colormap (bottom row) for each camera in one
window. Use before a collection run to sanity-check the workspace and camera
angles. Press `q` or Esc to quit. Fully release the pipelines before starting
a collect script, or the cameras will show as busy.
"""
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import sys
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "collect", "config.yaml")


def list_realsense_devices():
    ctx = rs.context()
    devices = ctx.query_devices()
    serials = []
    for dev in devices:
        try:
            serials.append(dev.get_info(rs.camera_info.serial_number))
        except Exception:
            pass
    return serials


def camera_names(serials):
    """Order devices as camera1/2/3 from collect/config.yaml (USB order is arbitrary)."""
    with open(CONFIG_PATH) as f:
        configured = [str(s) for s in yaml.safe_load(f)["recording"]["camera_serials"]]
    ordered = [s for s in configured if s in serials] + [s for s in serials if s not in configured]
    names = [f"camera{configured.index(s) + 1}" if s in configured else "unknown" for s in ordered]
    return ordered, names

def make_pipeline(serial, width=640, height=480, fps=30):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)

    # Enable streams
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    profile = pipeline.start(config)

    # Align depth to color for nicer overlay / matching
    align = rs.align(rs.stream.color)

    # Depth scale (meters per unit)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    return pipeline, align, depth_scale

def depth_to_colormap(depth_frame, depth_scale, max_dist_m=2.0):
    depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
    depth = np.clip(depth, 0.0, max_dist_m)
    depth_u8 = (depth / max_dist_m * 255.0).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)

def main():
    serials = list_realsense_devices()
    if len(serials) < 1:
        print("No RealSense devices found")
        sys.exit(1)

    serials, names = camera_names(serials)
    print(f"Using {len(serials)} device(s):")
    for name, s in zip(names, serials):
        print(f"  {name} serial: {s}")

    cams = [make_pipeline(s) for s in serials]

    try:
        while True:
            color_imgs = []
            depth_imgs = []
            drop = False
            for (pipeline, align, depth_scale), serial, name in zip(cams, serials, names):
                frames = pipeline.wait_for_frames()
                frames = align.process(frames)
                c = frames.get_color_frame()
                d = frames.get_depth_frame()
                if not (c and d):
                    drop = True
                    break

                img_c = np.asanyarray(c.get_data())
                img_d = depth_to_colormap(d, depth_scale, max_dist_m=2.0)

                cv2.putText(img_c, f"{name} Color ({serial})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(img_d, f"{name} Depth", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                color_imgs.append(img_c)
                depth_imgs.append(img_d)

            if drop:
                continue

            top = np.hstack(color_imgs)
            bot = np.hstack(depth_imgs)
            grid = np.vstack([top, bot])

            cv2.imshow("RealSense: (Top) Color | (Bottom) Depth  -  Press q to quit", grid)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        for pipeline, _, _ in cams:
            try:
                pipeline.stop()
            except Exception:
                pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
