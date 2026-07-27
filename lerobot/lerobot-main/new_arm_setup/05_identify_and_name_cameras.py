#!/usr/bin/env python3
"""联合检测腕部 OpenCV 相机与全局 RealSense，交互指定角色并写入永久命名。"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RULES_PATH = Path(__file__).resolve().parent / "99-so101-cameras-v2.rules"
IDS_PATH = Path(__file__).resolve().parent / "camera_ids.json"
CONFIG_PATH = ROOT / "leader" / "leader_record_config_v2.json"

# 跳过笔记本内置摄像头（按 vendor / 名称关键词）
SKIP_VENDORS = {
    "30c9",  # Luxvisions / 本机 Integrated RGB Camera
}
SKIP_NAME_KEYWORDS = (
    "integrated",
    "ir camera",
    "infrared",
    "laptop",
    "builtin",
    "built-in",
)
# OpenCV 侧只允许这些外部 USB 相机做腕部/全局候选（不含 RealSense）
# 腕部相机当前为 1bcf:2281；若换相机可往这里加 vendor
ALLOW_OPENCV_VENDORS = {
    "1bcf",  # 腕部 USB 相机 Sunplus / JYU2C
}


@dataclass
class OpenCVCam:
    path: str
    vendor: str
    product: str
    serial: str
    product_name: str


@dataclass
class RealSenseCam:
    name: str
    serial: str
    usb: str
    asic_serial: str
    product_id: str


def udev_props(dev: str) -> dict[str, str]:
    out = subprocess.check_output(["udevadm", "info", "-q", "property", "-n", dev], text=True)
    props = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            props[k] = v
    return props


def is_laptop_camera(vendor: str, name: str) -> bool:
    if vendor in SKIP_VENDORS:
        return True
    lower = name.lower()
    return any(k in lower for k in SKIP_NAME_KEYWORDS)


def list_opencv_external() -> list[OpenCVCam]:
    cams: list[OpenCVCam] = []
    skipped: list[str] = []
    seen_serials: set[str] = set()
    for i in range(32):
        path = f"/dev/video{i}"
        if not Path(path).exists():
            continue
        try:
            idx = Path(f"/sys/class/video4linux/video{i}/index").read_text().strip()
        except Exception:
            continue
        if idx != "0":
            continue

        props = udev_props(path)
        vendor = props.get("ID_VENDOR_ID", "").lower()
        product = props.get("ID_MODEL_ID", "").lower()
        serial = props.get("ID_SERIAL_SHORT", "")
        name = props.get("ID_V4L_PRODUCT", path)

        # RealSense 走 pyrealsense2，不占用 /dev/video*
        if vendor == "8086":
            skipped.append(f"{path} RealSense(UVC) -> 改用 pyrealsense2")
            continue
        if is_laptop_camera(vendor, name):
            skipped.append(f"{path} 笔记本内置: {name} ({vendor}:{product})")
            continue
        if ALLOW_OPENCV_VENDORS and vendor not in ALLOW_OPENCV_VENDORS:
            skipped.append(f"{path} 非白名单外部相机: {name} ({vendor}:{product})")
            continue

        key = f"{vendor}:{product}:{serial}"
        if key in seen_serials:
            continue
        seen_serials.add(key)
        cams.append(OpenCVCam(path, vendor, product, serial, name))

    if skipped:
        print("已排除的相机：")
        for s in skipped:
            print(f"  - {s}")
    return cams


def list_realsense() -> list[RealSenseCam]:
    import pyrealsense2 as rs

    cams: list[RealSenseCam] = []
    for d in rs.context().devices:
        cams.append(
            RealSenseCam(
                name=d.get_info(rs.camera_info.name),
                serial=d.get_info(rs.camera_info.serial_number),
                usb=d.get_info(rs.camera_info.usb_type_descriptor),
                asic_serial=d.get_info(rs.camera_info.asic_serial_number),
                product_id=d.get_info(rs.camera_info.product_id),
            )
        )
    return cams


def grab_opencv(path: str, w=640, h=480) -> np.ndarray | None:
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    ok, frame = False, None
    for _ in range(8):
        ok, frame = cap.read()
        if ok and frame is not None:
            break
    cap.release()
    return frame if ok else None


def grab_realsense(serial: str, w=640, h=480) -> np.ndarray | None:
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, 30)
    try:
        pipeline.start(cfg)
        frame = None
        for _ in range(30):
            frames = pipeline.wait_for_frames(timeout_ms=2000)
            color = frames.get_color_frame()
            if color:
                frame = np.asanyarray(color.get_data())
                break
        pipeline.stop()
        return frame
    except Exception:
        try:
            pipeline.stop()
        except Exception:
            pass
        return None


def put_banner(img: np.ndarray, lines: list[str]) -> np.ndarray:
    out = img.copy()
    y = 28
    for line in lines:
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        y += 28
    return out


def main() -> None:
    print("=== 扫描相机（仅全局 D435i + 腕部 USB，排除笔记本内置）===")
    ocv = list_opencv_external()
    rs_cams = list_realsense()

    if not ocv:
        raise SystemExit(
            "未找到腕部 OpenCV 相机（白名单 vendor=1bcf）。请确认腕部相机已插入。"
        )
    if not rs_cams:
        raise SystemExit("未找到 RealSense 全局相机。请确认 D435i 已插入且 lsusb 能看到 8086。")

    if len(ocv) != 1 or len(rs_cams) != 1:
        print(
            f"\n警告：期望正好 1 路腕部 OpenCV + 1 路 RealSense，"
            f"当前 OpenCV={len(ocv)} RealSense={len(rs_cams)}。"
        )

    print("\n将用于命名的相机（仅这两类）：")
    print("OpenCV 腕部候选:")
    for i, c in enumerate(ocv):
        print(f"  O{i}: {c.path}  {c.vendor}:{c.product} serial={c.serial}  ({c.product_name})")
    print("RealSense 全局候选:")
    for i, c in enumerate(rs_cams):
        print(f"  R{i}: {c.name} serial={c.serial} usb={c.usb} product_id={c.product_id}")

    # 默认取第一路外部 USB 相机 + 第一路 RealSense 做联合预览
    o = ocv[0]
    r = rs_cams[0]
    if len(ocv) > 1 or len(rs_cams) > 1:
        print("\n检测到多路，默认预览 O0 + R0。若不对请先只保留两路目标相机。")

    print("\n正在打开预览窗口...")
    print("看画面后在终端选择角色：")
    print("  1 = OpenCV(O0) 是腕部，RealSense(R0) 是全局")
    print("  2 = RealSense(R0) 是腕部，OpenCV(O0) 是全局")
    print("  q = 退出不保存")
    print("预览窗口按任意键可刷新一帧；在终端输入选择。")

    while True:
        f_o = grab_opencv(o.path)
        f_r = grab_realsense(r.serial)
        if f_o is None:
            print(f"无法读取 {o.path}")
            return
        if f_r is None:
            print(f"无法读取 RealSense {r.serial}")
            return

        fo = put_banner(cv2.resize(f_o, (640, 480)), [f"O0 OpenCV {o.path}", f"{o.vendor}:{o.product}", o.serial])
        fr = put_banner(cv2.resize(f_r, (640, 480)), [f"R0 RealSense {r.serial}", r.name, f"USB {r.usb}"])
        both = np.hstack([fo, fr])
        cv2.imshow("LEFT=O0 OpenCV | RIGHT=R0 RealSense  (press any key to refresh)", both)
        cv2.waitKey(1)

        choice = input("选择 [1/2/q]，或直接回车刷新预览> ").strip().lower()
        if choice == "":
            continue
        if choice == "q":
            cv2.destroyAllWindows()
            print("已取消。")
            return
        if choice in {"1", "2"}:
            break
        print("无效输入，请输入 1 / 2 / q")

    cv2.destroyAllWindows()

    if choice == "1":
        wrist_kind, global_kind = "opencv", "realsense"
        wrist_ocv, global_rs = o, r
        wrist_rs, global_ocv = None, None
    else:
        wrist_kind, global_kind = "realsense", "opencv"
        wrist_rs, global_ocv = r, o
        wrist_ocv, global_rs = None, None

    print("\n你的选择：")
    print(f"  腕部(wrist) = {wrist_kind}")
    print(f"  全局(front) = {global_kind}")

    # 生成 udev：仅对 OpenCV USB 相机做 /dev/video-so101-* 永久名
    # RealSense 用序列号永久绑定（采集走 intelrealsense）
    rules: list[str] = [
        f"# SO101 cameras v2 generated {date.today().isoformat()}",
        "# wrist / front OpenCV nodes (ATTR index==0 only)",
    ]

    identity = {
        "date": date.today().isoformat(),
        "wrist": {"kind": wrist_kind},
        "front": {"kind": global_kind},
    }

    cameras_cfg: dict = {}

    def add_opencv_role(role: str, cam: OpenCVCam, symlink: str) -> None:
        rules.append(
            f'SUBSYSTEM=="video4linux", ATTR{{index}}=="0", '
            f'ATTRS{{idVendor}}=="{cam.vendor}", ATTRS{{idProduct}}=="{cam.product}", '
            f'ATTRS{{serial}}=="{cam.serial}", SYMLINK+="{symlink}", MODE="0666"'
        )
        identity[role].update(
            {
                "type": "opencv",
                "path": f"/dev/{symlink}",
                "vendor": cam.vendor,
                "product": cam.product,
                "serial": cam.serial,
                "raw_path_example": cam.path,
            }
        )
        cameras_cfg[role if role != "front" else "front"] = {
            "type": "opencv",
            "index_or_path": f"/dev/{symlink}",
            "width": 640,
            "height": 480,
            "fps": 30,
        }

    def add_rs_role(role: str, cam: RealSenseCam) -> None:
        identity[role].update(
            {
                "type": "intelrealsense",
                "serial_number_or_name": cam.serial,
                "name": cam.name,
                "usb": cam.usb,
                "asic_serial": cam.asic_serial,
                "product_id": cam.product_id,
            }
        )
        cameras_cfg[role if role != "front" else "front"] = {
            "type": "intelrealsense",
            "serial_number_or_name": cam.serial,
            "width": 640,
            "height": 480,
            "fps": 30,
            "use_depth": False,
        }

    if wrist_ocv is not None:
        add_opencv_role("wrist", wrist_ocv, "video-so101-wrist")
    if wrist_rs is not None:
        add_rs_role("wrist", wrist_rs)
    if global_ocv is not None:
        add_opencv_role("front", global_ocv, "video-so101-fixed")
    if global_rs is not None:
        add_rs_role("front", global_rs)

    RULES_PATH.write_text("\n".join(rules) + "\n", encoding="utf-8")
    IDS_PATH.write_text(json.dumps(identity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with CONFIG_PATH.open(encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["follower"]["cameras"] = cameras_cfg
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n已写入：\n  {IDS_PATH}\n  {RULES_PATH}\n  {CONFIG_PATH}")
    print("\n安装 OpenCV 相机永久名（需 sudo）：")
    print(f"  sudo cp {RULES_PATH} /etc/udev/rules.d/99-so101-cameras-v2.rules")
    print("  sudo udevadm control --reload-rules && sudo udevadm trigger")
    print("  ls -l /dev/video-so101-wrist /dev/video-so101-fixed 2>/dev/null || true")
    print("\n说明：RealSense 用序列号永久绑定，不依赖 /dev/videoX；下次插拔不会乱。")


if __name__ == "__main__":
    main()
