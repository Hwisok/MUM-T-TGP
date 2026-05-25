import socket
import struct
import threading
import time
import json
import queue
import os
import tempfile
import re
import difflib
from typing import Optional, Dict, Any, Tuple, List

import cv2
import numpy as np

import zmq
import msgpack

import sounddevice as sd
from scipy.io.wavfile import write as wav_write
from faster_whisper import WhisperModel

from PIL import Image, ImageDraw, ImageFont


VIDEO_HOST = "127.0.0.1"
VIDEO_PORT = 5000

METADATA_BIND_HOST = "127.0.0.1"
METADATA_PORT = 5001

RETARGET_HOST = "127.0.0.1"
RETARGET_PORT = 5002

WINDOW_NAME = "UAV Receiver"
DISPLAY_W = 1280
DISPLAY_H = 720

MAX_FRAME_QUEUE = 1
RAW_CLICK_DISPLAY_SEC = 0.5
GAZE_CURSOR_COLOR = (255, 0, 255)

PUPIL_REMOTE_IP = "127.0.0.1"
PUPIL_REMOTE_PORT = 50020
PUPIL_CONFIDENCE_THRESHOLD = 0.6
PUPIL_FUSION_MAX_DT = 0.08

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshots")
LOG_DIR = os.path.join(BASE_DIR, "logs")
CSV_LOG_PATH = os.path.join(LOG_DIR, "command_response_log.csv")

SOURCE_W = 1280
SOURCE_H = 720

FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"

CALIBRATION_POINTS_NORM = [
    (0.50, 0.50),
    (0.15, 0.15),
    (0.85, 0.15),
    (0.15, 0.85),
    (0.85, 0.85),
]

VOICE_PTT_DURATION_SEC = 2.5
VOICE_SAMPLE_RATE = 16000
VOICE_CHANNELS = 1

WHISPER_MODEL_SIZE = "small"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

GAZE_TARGET_MATCH_PX = 60
GAZE_AVG_WINDOW_SEC = 0.30
STT_MATCH_THRESHOLD = 0.58
ACK_POPUP_SEC = 2.5
SAVE_POPUP_SEC = 2.5
CAMERA_TARGET_REACHED_PX = 40

COMMAND_SYNONYMS = {
    "attack_here": [
        "여기공격", "여기 공격", "여길공격", "여길 공격",
        "이곳공격", "이곳 공격", "여기 타격", "이곳 타격"
    ],
    "attack_there": [
        "거기공격", "거기 공격", "저기공격", "저기 공격",
        "그곳공격", "그곳 공격", "거기 타격", "저기 타격"
    ],
    "track_here": [
        "여기추적", "여기 추적", "여길추적", "여길 추적",
        "이곳추적", "이곳 추적"
    ],
    "track_there": [
        "거기추적", "거기 추적", "저기추적", "저기 추적",
        "그곳추적", "그곳 추적"
    ],
    "hold": [
        "대기", "대기해", "대기 해", "대기하", "대기 하",
        "대기해라", "기다려", "유지해"
    ],
    "attack_generic": [
        "공격해", "공격 해", "공격", "타격해", "타격 해", "타격"
    ],
    "track_generic": [
        "추적해", "추적 해", "추적", "따라가", "계속추적", "계속 추적"
    ],
}

ACK_TEXT_MAP = {
    "ACK_ATTACK_NOW": "바로 공격합니다.",
    "ACK_TRACK_CURRENT": "계속 추적합니다.",
    "ACK_CONFIRM_ATTACK": "여기 공격 맞습니까?",
    "ACK_TRACK_HERE": "여기 추적합니다.",
    "ACK_UNKNOWN": "명령을 다시 확인합니다.",
}


def safe_load_font(size: int):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


FONT_SMALL = safe_load_font(18)
FONT_MEDIUM = safe_load_font(24)
FONT_BIG = safe_load_font(40)


def ensure_csv_log_header():
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        if not os.path.exists(CSV_LOG_PATH):
            with open(CSV_LOG_PATH, "w", encoding="utf-8-sig") as f:
                f.write(
                    "command_time,voice_text,gaze_x,gaze_y,current_target_before_x,current_target_before_y,"
                    "target_x,target_y,response_time,response_latency_sec,ack_key,ack_text,fused_action\n"
                )
    except Exception:
        pass


def csv_escape(value):
    s = str(value)
    s = s.replace('"', "'")
    return f"\"{s}\""


def append_command_csv(row: Dict[str, Any]):
    try:
        ensure_csv_log_header()
        with open(CSV_LOG_PATH, "a", encoding="utf-8-sig") as f:
            f.write(
                f"{row.get('command_time','')},"
                f"{csv_escape(row.get('voice_text',''))},"
                f"{row.get('gaze_x','')},"
                f"{row.get('gaze_y','')},"
                f"{row.get('current_target_before_x','')},"
                f"{row.get('current_target_before_y','')},"
                f"{row.get('target_x','')},"
                f"{row.get('target_y','')},"
                f"{row.get('response_time','')},"
                f"{row.get('response_latency_sec','')},"
                f"{row.get('ack_key','')},"
                f"{csv_escape(row.get('ack_text',''))},"
                f"{row.get('fused_action','')}\n"
            )
    except Exception:
        pass


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()

        self.latest_metadata: Dict[str, Any] = {}
        self.last_click: Optional[Tuple[int, int]] = None
        self.last_click_time: float = 0.0
        self.running: bool = True
        self.frames_received: int = 0
        self.status_text: str = "INIT"
        self.mouse_pos: Tuple[int, int] = (0, 0)

        self.left_pupil_norm: Optional[Tuple[float, float]] = None
        self.left_pupil_time: float = 0.0
        self.right_pupil_norm: Optional[Tuple[float, float]] = None
        self.right_pupil_time: float = 0.0

        self.calibration_mode: bool = False
        self.calibration_index: int = 0
        self.calibration_samples: List[List[Tuple[float, float]]] = [[] for _ in CALIBRATION_POINTS_NORM]
        self.affine_matrix: Optional[np.ndarray] = None

        self.last_voice_text: str = ""
        self.last_voice_time: float = 0.0
        self.last_fusion_result: Dict[str, Any] = {}

        self.gaze_history: List[Tuple[float, Tuple[int, int]]] = []

        self.last_intent_ack: str = ""
        self.last_intent_code: str = ""
        self.last_intent_ack_time: float = 0.0
        self.last_intent_seq: int = -1
        self.last_intent_ack_key: str = ""

        self.last_save_status: str = ""
        self.last_save_time: float = 0.0

        self.voice_busy: bool = False

        self.flash_gaze_point: Optional[Tuple[int, int]] = None
        self.flash_gaze_active: bool = False

        self.world_target_marker_active: bool = False
        self.pending_command_log: Optional[Dict[str, Any]] = None

    def update_metadata(self, metadata: Dict[str, Any]):
        with self.lock:
            self.latest_metadata = metadata

            code = metadata.get("intent_code", "")
            seq = metadata.get("intent_seq", None)
            ack_key = metadata.get("intent_ack_key", "")

            if isinstance(code, str):
                self.last_intent_code = code

            if isinstance(seq, (int, float)):
                seq_i = int(seq)
                if seq_i != self.last_intent_seq:
                    self.last_intent_seq = seq_i

                    ack_text = ""
                    if isinstance(ack_key, str):
                        ack_key = ack_key.strip()
                        self.last_intent_ack_key = ack_key
                        ack_text = ACK_TEXT_MAP.get(ack_key, ack_key)
                        if ack_text:
                            self.last_intent_ack = ack_text
                            self.last_intent_ack_time = time.time()

                    pending = None
                    if self.pending_command_log is not None:
                        pending = dict(self.pending_command_log)
                        self.pending_command_log = None

                    if pending is not None:
                        response_time = time.time()
                        target_pixel = metadata.get("target_pixel", None)
                        target_x = ""
                        target_y = ""
                        if isinstance(target_pixel, (list, tuple)) and len(target_pixel) >= 2:
                            target_x = target_pixel[0]
                            target_y = target_pixel[1]

                        append_command_csv({
                            "command_time": pending.get("command_time", ""),
                            "voice_text": pending.get("voice_text", ""),
                            "gaze_x": pending.get("gaze_x", ""),
                            "gaze_y": pending.get("gaze_y", ""),
                            "current_target_before_x": pending.get("current_target_before_x", ""),
                            "current_target_before_y": pending.get("current_target_before_y", ""),
                            "target_x": target_x,
                            "target_y": target_y,
                            "response_time": response_time,
                            "response_latency_sec": response_time - pending.get("command_time", response_time),
                            "ack_key": ack_key,
                            "ack_text": ack_text,
                            "fused_action": pending.get("fused_action", ""),
                        })

                    self.world_target_marker_active = True

    def get_metadata(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.latest_metadata)

    def set_click(self, x: int, y: int):
        with self.lock:
            self.last_click = (x, y)
            self.last_click_time = time.time()

    def get_click(self) -> Optional[Tuple[int, int]]:
        with self.lock:
            return self.last_click

    def get_click_age(self) -> float:
        with self.lock:
            if self.last_click is None:
                return float("inf")
            return time.time() - self.last_click_time

    def set_mouse_pos(self, x: int, y: int):
        with self.lock:
            self.mouse_pos = (x, y)

    def get_mouse_pos(self) -> Tuple[int, int]:
        with self.lock:
            return self.mouse_pos

    def set_eye_pupil_norm(self, eye_id: int, norm_xy: Tuple[float, float]):
        with self.lock:
            now = time.time()
            if eye_id == 0:
                self.left_pupil_norm = norm_xy
                self.left_pupil_time = now
            elif eye_id == 1:
                self.right_pupil_norm = norm_xy
                self.right_pupil_time = now

    def get_fused_pupil_norm(self) -> Tuple[Optional[Tuple[float, float]], float]:
        with self.lock:
            now = time.time()

            left_valid = self.left_pupil_norm is not None and (now - self.left_pupil_time) < 1.0
            right_valid = self.right_pupil_norm is not None and (now - self.right_pupil_time) < 1.0

            if left_valid and right_valid:
                dt = abs(self.left_pupil_time - self.right_pupil_time)
                if dt <= PUPIL_FUSION_MAX_DT:
                    nx = 0.5 * (self.left_pupil_norm[0] + self.right_pupil_norm[0])
                    ny = 0.5 * (self.left_pupil_norm[1] + self.right_pupil_norm[1])
                    return (nx, ny), max(self.left_pupil_time, self.right_pupil_time)

            if left_valid and not right_valid:
                return self.left_pupil_norm, self.left_pupil_time

            if right_valid and not left_valid:
                return self.right_pupil_norm, self.right_pupil_time

            if left_valid and right_valid:
                if self.left_pupil_time >= self.right_pupil_time:
                    return self.left_pupil_norm, self.left_pupil_time
                else:
                    return self.right_pupil_norm, self.right_pupil_time

            return None, 0.0

    def push_gaze_point(self, gaze_xy: Tuple[int, int], gaze_time: Optional[float] = None):
        with self.lock:
            ts = time.time() if gaze_time is None else gaze_time
            self.gaze_history.append((ts, gaze_xy))

            cutoff = ts - 2.0
            self.gaze_history = [(t, p) for t, p in self.gaze_history if t >= cutoff]

    def get_recent_gaze_average(self, window_sec: float) -> Optional[Tuple[int, int]]:
        with self.lock:
            now = time.time()
            pts = [xy for ts, xy in self.gaze_history if (now - ts) <= window_sec]
            if not pts:
                return None
            arr = np.array(pts, dtype=np.float64)
            mean_xy = arr.mean(axis=0)
            x = int(round(mean_xy[0]))
            y = int(round(mean_xy[1]))
            x = max(0, min(DISPLAY_W - 1, x))
            y = max(0, min(DISPLAY_H - 1, y))
            return x, y

    def stop(self):
        with self.lock:
            self.running = False

    def is_running(self) -> bool:
        with self.lock:
            return self.running

    def set_status(self, text: str):
        with self.lock:
            self.status_text = text

    def get_status(self) -> str:
        with self.lock:
            return self.status_text

    def start_calibration(self):
        with self.lock:
            self.calibration_mode = True
            self.calibration_index = 0
            self.calibration_samples = [[] for _ in CALIBRATION_POINTS_NORM]
            self.affine_matrix = None

    def reset_calibration(self):
        with self.lock:
            self.calibration_mode = False
            self.calibration_index = 0
            self.calibration_samples = [[] for _ in CALIBRATION_POINTS_NORM]
            self.affine_matrix = None

    def is_calibration_mode(self) -> bool:
        with self.lock:
            return self.calibration_mode

    def get_calibration_index(self) -> int:
        with self.lock:
            return self.calibration_index

    def advance_calibration_index(self):
        with self.lock:
            if self.calibration_index < len(CALIBRATION_POINTS_NORM) - 1:
                self.calibration_index += 1
            else:
                self.calibration_mode = False

    def add_calibration_sample(self, sample: Tuple[float, float]):
        with self.lock:
            idx = self.calibration_index
            if 0 <= idx < len(self.calibration_samples):
                self.calibration_samples[idx].append(sample)

    def get_calibration_samples(self):
        with self.lock:
            return [list(s) for s in self.calibration_samples]

    def set_affine_matrix(self, A: np.ndarray):
        with self.lock:
            self.affine_matrix = A

    def get_affine_matrix(self) -> Optional[np.ndarray]:
        with self.lock:
            return None if self.affine_matrix is None else self.affine_matrix.copy()

    def set_last_voice_text(self, text: str):
        with self.lock:
            self.last_voice_text = text
            self.last_voice_time = time.time()

    def get_last_voice_text(self) -> Tuple[str, float]:
        with self.lock:
            return self.last_voice_text, self.last_voice_time

    def set_last_fusion_result(self, result: Dict[str, Any]):
        with self.lock:
            self.last_fusion_result = dict(result)

    def get_last_fusion_result(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.last_fusion_result)

    def get_last_intent_ack(self) -> Tuple[str, str, float, str]:
        with self.lock:
            return self.last_intent_ack, self.last_intent_code, self.last_intent_ack_time, self.last_intent_ack_key

    def set_save_status(self, text: str):
        with self.lock:
            self.last_save_status = text
            self.last_save_time = time.time()

    def get_save_status(self) -> Tuple[str, float]:
        with self.lock:
            return self.last_save_status, self.last_save_time

    def set_voice_busy(self, v: bool):
        with self.lock:
            self.voice_busy = v

    def is_voice_busy(self) -> bool:
        with self.lock:
            return self.voice_busy

    def set_flash_gaze_point(self, pt: Optional[Tuple[int, int]]):
        with self.lock:
            self.flash_gaze_point = pt
            self.flash_gaze_active = pt is not None

    def get_flash_gaze_point(self) -> Optional[Tuple[int, int]]:
        with self.lock:
            if self.flash_gaze_active:
                return self.flash_gaze_point
            return None

    def clear_flash_gaze_point(self):
        with self.lock:
            self.flash_gaze_point = None
            self.flash_gaze_active = False

    def set_world_target_marker_active(self, active: bool):
        with self.lock:
            self.world_target_marker_active = active

    def is_world_target_marker_active(self) -> bool:
        with self.lock:
            return self.world_target_marker_active

    def set_pending_command_log(self, data: Dict[str, Any]):
        with self.lock:
            self.pending_command_log = dict(data)


def recv_exact(sock: socket.socket, size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Socket closed while receiving data.")
        data += chunk
    return data


class VideoReceiverThread(threading.Thread):
    def __init__(self, host: str, port: int, frame_queue: queue.Queue, state: SharedState):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.frame_queue = frame_queue
        self.state = state

    def push_latest_frame(self, frame):
        while True:
            try:
                self.frame_queue.put_nowait(frame)
                return
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    return

    def run(self):
        while self.state.is_running():
            sock = None
            try:
                self.state.set_status("VIDEO_CONNECTING")
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((self.host, self.port))
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.state.set_status("VIDEO_CONNECTED")

                while self.state.is_running():
                    header = recv_exact(sock, 4)
                    (frame_len,) = struct.unpack(">I", header)

                    if frame_len <= 0 or frame_len > 20 * 1024 * 1024:
                        raise ValueError("Invalid frame length")

                    jpeg_bytes = recv_exact(sock, frame_len)
                    np_buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)

                    if frame is None:
                        continue

                    self.state.frames_received += 1
                    self.push_latest_frame(frame)

            except Exception:
                self.state.set_status("VIDEO_RECONNECTING")
                time.sleep(1.0)
            finally:
                if sock is not None:
                    try:
                        sock.close()
                    except Exception:
                        pass


class MetadataListenerThread(threading.Thread):
    def __init__(self, bind_host: str, port: int, state: SharedState):
        super().__init__(daemon=True)
        self.bind_host = bind_host
        self.port = port
        self.state = state

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.bind_host, self.port))
        sock.settimeout(0.5)

        while self.state.is_running():
            try:
                data, _addr = sock.recvfrom(65535)
                text = data.decode("utf-8", errors="strict")
                metadata = json.loads(text)
                self.state.update_metadata(metadata)
            except socket.timeout:
                continue
            except Exception:
                continue

        sock.close()


class PupilCoreThread(threading.Thread):
    def __init__(self, ip: str, remote_port: int, state: SharedState):
        super().__init__(daemon=True)
        self.ip = ip
        self.remote_port = remote_port
        self.state = state

    def run(self):
        while self.state.is_running():
            ctx = None
            remote = None
            sub = None
            try:
                ctx = zmq.Context()
                remote = ctx.socket(zmq.REQ)
                remote.setsockopt(zmq.RCVTIMEO, 2000)
                remote.setsockopt(zmq.SNDTIMEO, 2000)
                remote.connect(f"tcp://{self.ip}:{self.remote_port}")

                remote.send_string("SUB_PORT")
                sub_port = remote.recv_string()

                sub = ctx.socket(zmq.SUB)
                sub.setsockopt(zmq.RCVTIMEO, 1000)
                sub.connect(f"tcp://{self.ip}:{sub_port}")
                sub.subscribe("pupil.")

                while self.state.is_running():
                    topic, payload = sub.recv_multipart()
                    message = msgpack.loads(payload, raw=False)

                    conf = float(message.get("confidence", 1.0))
                    if conf < PUPIL_CONFIDENCE_THRESHOLD:
                        continue

                    eye_id = message.get("id", None)
                    if eye_id is None:
                        continue

                    try:
                        eye_id = int(eye_id)
                    except Exception:
                        continue

                    norm_pos = message.get("norm_pos", None)
                    if isinstance(norm_pos, (list, tuple)) and len(norm_pos) >= 2:
                        try:
                            nx = float(norm_pos[0])
                            ny = float(norm_pos[1])
                            self.state.set_eye_pupil_norm(eye_id, (nx, ny))
                        except Exception:
                            pass

            except Exception:
                time.sleep(1.0)
            finally:
                try:
                    if sub is not None:
                        sub.close(0)
                except Exception:
                    pass
                try:
                    if remote is not None:
                        remote.close(0)
                except Exception:
                    pass
                try:
                    if ctx is not None:
                        ctx.term()
                except Exception:
                    pass


class RetargetSender:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_retarget(self, display_x: int, display_y: int):
        msg = {
            "cmd": "retarget",
            "timestamp": time.time(),
            "source": "python_receiver",
        }

        src_x = int(round(display_x * (SOURCE_W / DISPLAY_W)))
        src_y = int(round(display_y * (SOURCE_H / DISPLAY_H)))
        src_x = max(1, min(SOURCE_W, src_x))
        src_y = max(1, min(SOURCE_H, src_y))
        msg["src_pixel"] = [src_x, src_y]

        payload = json.dumps(msg).encode("utf-8")
        self.sock.sendto(payload, (self.host, self.port))

    def send_cancel(self):
        msg = {"cmd": "cancel_retarget", "timestamp": time.time()}
        payload = json.dumps(msg).encode("utf-8")
        self.sock.sendto(payload, (self.host, self.port))

    def send_fused_intent(self, payload_dict: Dict[str, Any]):
        payload = json.dumps(payload_dict).encode("utf-8")
        self.sock.sendto(payload, (self.host, self.port))


class VoiceProcessor:
    def __init__(self):
        self.model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE
        )

    def record_audio(self, duration_sec: float) -> np.ndarray:
        audio = sd.rec(
            int(duration_sec * VOICE_SAMPLE_RATE),
            samplerate=VOICE_SAMPLE_RATE,
            channels=VOICE_CHANNELS,
            dtype="float32"
        )
        sd.wait()
        if VOICE_CHANNELS == 1:
            audio = np.squeeze(audio, axis=1)
        return audio

    def transcribe_ptt(self, duration_sec: float = VOICE_PTT_DURATION_SEC) -> Tuple[str, float]:
        audio = self.record_audio(duration_sec)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_wav_path = tmp.name

        try:
            wav_int16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
            wav_write(temp_wav_path, VOICE_SAMPLE_RATE, wav_int16)

            segments, info = self.model.transcribe(
                temp_wav_path,
                language="ko",
                vad_filter=True,
                beam_size=5
            )

            text = " ".join(seg.text.strip() for seg in segments).strip()
            confidence = float(getattr(info, "language_probability", 0.0))
            return text, confidence
        finally:
            try:
                os.remove(temp_wav_path)
            except Exception:
                pass


class VoiceFusionWorker(threading.Thread):
    def __init__(self, state: SharedState, retarget_sender: RetargetSender, voice_processor: VoiceProcessor):
        super().__init__(daemon=True)
        self.state = state
        self.retarget_sender = retarget_sender
        self.voice_processor = voice_processor
        self.jobs = queue.Queue()

    def submit(self, gaze_point: Optional[Tuple[int, int]], current_target_pixel: Optional[Tuple[int, int]]):
        if self.state.is_voice_busy():
            return False
        self.jobs.put((gaze_point, current_target_pixel))
        return True

    def run(self):
        while self.state.is_running():
            try:
                gaze_point, current_target_pixel = self.jobs.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self.state.set_voice_busy(True)
                self.state.set_status("VOICE_RECORDING")

                text, stt_conf = self.voice_processor.transcribe_ptt(VOICE_PTT_DURATION_SEC)
                self.state.set_last_voice_text(text)

                voice_info = parse_voice_command(text)

                fused_payload = build_fused_intent(
                    voice_info=voice_info,
                    gaze_point=gaze_point,
                    current_target_pixel=current_target_pixel,
                    stt_confidence=stt_conf,
                )

                voice_text = fused_payload.get("voice", {}).get("raw_text", "")
                gaze_xy = fused_payload.get("gaze", {}).get("display_xy", None)
                fused_action = fused_payload.get("fusion", {}).get("fused_action", "")

                self.state.set_pending_command_log({
                    "command_time": time.time(),
                    "voice_text": voice_text,
                    "gaze_x": gaze_xy[0] if isinstance(gaze_xy, (list, tuple)) and len(gaze_xy) >= 2 else "",
                    "gaze_y": gaze_xy[1] if isinstance(gaze_xy, (list, tuple)) and len(gaze_xy) >= 2 else "",
                    "current_target_before_x": current_target_pixel[0] if current_target_pixel is not None else "",
                    "current_target_before_y": current_target_pixel[1] if current_target_pixel is not None else "",
                    "fused_action": fused_action,
                })

                self.state.set_last_fusion_result(fused_payload)
                self.state.set_world_target_marker_active(True)
                self.retarget_sender.send_fused_intent(fused_payload)
                self.state.set_status("VOICE_SENT")
            except Exception as e:
                self.state.set_status(f"VOICE_ERR: {str(e)[:40]}")
            finally:
                self.state.set_voice_busy(False)


def fmt_vec(v, n=3, prec=2):
    if not isinstance(v, (list, tuple)) or len(v) < n:
        return "[N/A]"
    try:
        return "[" + ", ".join(f"{float(v[i]):.{prec}f}" for i in range(n)) + "]"
    except Exception:
        return "[N/A]"


def dist_px(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def pil_draw_text(img_bgr, text, pos, font, fill=(255, 255, 255), stroke_fill=(0, 0, 0), stroke_width=2):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    draw.text(pos, text, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
    out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img_bgr[:, :, :] = out
    return img_bgr


def pil_draw_multiline(img_bgr, lines, x, y, font, line_h=22, fill=(235, 235, 235), stroke_fill=(0, 0, 0)):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    yy = y
    for line in lines:
        draw.text((x, yy), line, font=font, fill=fill, stroke_width=1, stroke_fill=stroke_fill)
        yy += line_h
    out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img_bgr[:, :, :] = out
    return img_bgr


def draw_text_block_pil(img, lines, x=14, y=22, line_h=22,
                        font=FONT_SMALL,
                        text_color=(235, 235, 235),
                        bg_color=(20, 20, 20),
                        alpha=0.36):
    overlay = img.copy()
    widths = []
    for line in lines:
        try:
            bbox = font.getbbox(line)
            widths.append(bbox[2] - bbox[0])
        except Exception:
            widths.append(len(line) * 10)

    max_w = max(widths) if widths else 0
    box_h = 12 + line_h * len(lines)
    box_w = max_w + 20

    cv2.rectangle(overlay, (x - 8, y - 16), (x - 8 + box_w, y - 16 + box_h), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)

    return pil_draw_multiline(img, lines, x, y, font=font, line_h=line_h, fill=text_color)


def draw_center_ack_popup(img: np.ndarray, text: str):
    if not text:
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    bbox = draw.textbbox((0, 0), text, font=FONT_BIG, stroke_width=2)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    pad_x = 32
    pad_y = 24
    box_w = tw + pad_x * 2
    box_h = th + pad_y * 2

    x1 = (img.shape[1] - box_w) // 2
    y1 = (img.shape[0] - box_h) // 2
    x2 = x1 + box_w
    y2 = y1 + box_h

    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 215, 255), 2)
    cv2.addWeighted(overlay, 0.62, img, 0.38, 0, img)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    draw.text((x1 + pad_x, y1 + pad_y), text, font=FONT_BIG,
              fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img[:, :, :] = out


def save_snapshot(image: np.ndarray, state: SharedState):
    try:
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SNAPSHOT_DIR, f"snapshot_{ts}_{int((time.time() % 1) * 1000):03d}.jpg")

        ok = cv2.imwrite(filename, image)
        if ok:
            state.set_save_status(f"Saved: {filename}")
            return

        ok2, buf = cv2.imencode(".jpg", image)
        if ok2:
            with open(filename, "wb") as f:
                f.write(buf.tobytes())
            state.set_save_status(f"Saved: {filename}")
        else:
            state.set_save_status("Snapshot save failed")
    except Exception as e:
        state.set_save_status(f"Snapshot error: {str(e)[:60]}")


def compute_affine(cal_samples: List[List[Tuple[float, float]]]) -> Optional[np.ndarray]:
    src_pts = []
    dst_pts = []

    for i, samples in enumerate(cal_samples):
        if len(samples) == 0:
            return None

        arr = np.array(samples, dtype=np.float64)
        mean_pt = arr.mean(axis=0)

        px, py = float(mean_pt[0]), float(mean_pt[1])

        tx = CALIBRATION_POINTS_NORM[i][0] * DISPLAY_W
        ty = CALIBRATION_POINTS_NORM[i][1] * DISPLAY_H

        src_pts.append([px, py, 1.0, 0.0, 0.0, 0.0])
        src_pts.append([0.0, 0.0, 0.0, px, py, 1.0])

        dst_pts.append(tx)
        dst_pts.append(ty)

    A = np.array(src_pts, dtype=np.float64)
    b = np.array(dst_pts, dtype=np.float64)

    try:
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return x.reshape(2, 3)
    except Exception:
        return None


def map_pupil_to_screen(state: SharedState) -> Optional[Tuple[int, int]]:
    pupil_norm, pupil_time = state.get_fused_pupil_norm()
    if pupil_norm is None or (time.time() - pupil_time) > 1.0:
        return None

    A = state.get_affine_matrix()
    if A is None:
        return None

    px, py = pupil_norm
    vec = np.array([px, py, 1.0], dtype=np.float64)

    try:
        out = A @ vec
        x = int(round(out[0]))
        y = int(round(out[1]))
        x = max(0, min(DISPLAY_W - 1, x))
        y = max(0, min(DISPLAY_H - 1, y))
        return x, y
    except Exception:
        return None


def clamp_src_pixel(display_xy: Tuple[int, int]) -> Tuple[int, int]:
    display_x, display_y = display_xy
    src_x = int(round(display_x * (SOURCE_W / DISPLAY_W)))
    src_y = int(round(display_y * (SOURCE_H / DISPLAY_H)))
    src_x = max(1, min(SOURCE_W, src_x))
    src_y = max(1, min(SOURCE_H, src_y))
    return src_x, src_y


def parse_current_target_pixel(metadata: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    candidate_keys = ["current_target_pixel", "target_pixel", "desired_target_pixel"]

    for k in candidate_keys:
        v = metadata.get(k, None)
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            try:
                x = int(round(float(v[0])))
                y = int(round(float(v[1])))
                x = max(0, min(DISPLAY_W - 1, x))
                y = max(0, min(DISPLAY_H - 1, y))
                return x, y
            except Exception:
                pass
    return None


def normalize_korean_command(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"\s+", "", t)
    t = t.replace(".", "").replace(",", "").replace("!", "").replace("?", "")
    return t


def best_command_match(text: str) -> Tuple[Optional[str], float, str]:
    norm = normalize_korean_command(text)

    if not norm:
        return None, 0.0, ""

    best_key = None
    best_score = -1.0
    best_phrase = ""

    for cmd_key, phrases in COMMAND_SYNONYMS.items():
        for phrase in phrases:
            cand = normalize_korean_command(phrase)
            score = difflib.SequenceMatcher(None, norm, cand).ratio()

            if norm in cand or cand in norm:
                score = max(score, 0.92)

            if score > best_score:
                best_score = score
                best_key = cmd_key
                best_phrase = phrase

    if best_score < STT_MATCH_THRESHOLD:
        return None, best_score, best_phrase

    return best_key, best_score, best_phrase


def parse_voice_command(text: str) -> Dict[str, Any]:
    raw = text.strip()
    norm = normalize_korean_command(raw)

    result = {
        "raw_text": raw,
        "normalized_text": norm,
        "voice_action": "unknown",
        "deictic": "none",
        "valid": False,
        "matched_command_key": None,
        "matched_phrase": "",
        "match_score": 0.0,
    }

    cmd_key, match_score, matched_phrase = best_command_match(raw)
    result["matched_command_key"] = cmd_key
    result["matched_phrase"] = matched_phrase
    result["match_score"] = match_score

    if cmd_key is None:
        return result

    if cmd_key == "attack_here":
        result["voice_action"] = "attack"
        result["deictic"] = "here"
    elif cmd_key == "attack_there":
        result["voice_action"] = "attack"
        result["deictic"] = "there"
    elif cmd_key == "track_here":
        result["voice_action"] = "track"
        result["deictic"] = "here"
    elif cmd_key == "track_there":
        result["voice_action"] = "track"
        result["deictic"] = "there"
    elif cmd_key == "hold":
        result["voice_action"] = "hold"
        result["deictic"] = "none"
    elif cmd_key == "attack_generic":
        result["voice_action"] = "attack"
        result["deictic"] = "none"
    elif cmd_key == "track_generic":
        result["voice_action"] = "track"
        result["deictic"] = "none"

    result["valid"] = True
    return result


def build_fused_intent(
    voice_info: Dict[str, Any],
    gaze_point: Optional[Tuple[int, int]],
    current_target_pixel: Optional[Tuple[int, int]],
    stt_confidence: float,
) -> Dict[str, Any]:
    gaze_valid = gaze_point is not None
    target_valid = current_target_pixel is not None

    gaze_matches_target = False
    if gaze_valid and target_valid:
        gaze_matches_target = dist_px(gaze_point, current_target_pixel) <= GAZE_TARGET_MATCH_PX

    voice_action = voice_info["voice_action"]
    deictic = voice_info["deictic"]

    fused_action = "unknown"
    execution_mode = "unknown"
    reasoning = ""

    if voice_action == "attack":
        if deictic == "there":
            if gaze_valid and gaze_matches_target:
                fused_action = "attack_current_target"
                execution_mode = "direct_execute"
                reasoning = "gaze_on_current_target + there + attack"
            else:
                fused_action = "confirm_then_attack"
                execution_mode = "search_move_confirm"
                reasoning = "there_attack but gaze not on current target"
        elif deictic == "here":
            if gaze_valid and target_valid and not gaze_matches_target:
                fused_action = "attack_gaze_candidate"
                execution_mode = "search_move_confirm"
                reasoning = "here_attack with gaze away from current target"
            elif gaze_valid and not target_valid:
                fused_action = "attack_gaze_candidate"
                execution_mode = "search_move_confirm"
                reasoning = "here_attack with gaze and no known current target"
            else:
                fused_action = "confirm_then_attack"
                execution_mode = "confirm_only"
                reasoning = "here_attack but no distinct gaze candidate"
        elif deictic == "none":
            fused_action = "confirm_then_attack"
            execution_mode = "confirm_only"
            reasoning = "generic attack"

    elif voice_action == "track":
        if deictic == "there":
            fused_action = "track_current_target"
            execution_mode = "direct_execute"
            reasoning = "there_track defaults to current target"
        elif deictic == "here":
            if gaze_valid and target_valid and not gaze_matches_target:
                fused_action = "track_gaze_candidate"
                execution_mode = "retarget_to_gaze"
                reasoning = "here_track with gaze away from current target"
            elif gaze_valid and not target_valid:
                fused_action = "track_gaze_candidate"
                execution_mode = "retarget_to_gaze"
                reasoning = "here_track with gaze and no known target"
            else:
                fused_action = "track_current_target"
                execution_mode = "direct_execute"
                reasoning = "here_track without distinct gaze -> current target"
        elif deictic == "none":
            fused_action = "track_current_target"
            execution_mode = "direct_execute"
            reasoning = "generic track defaults to current target"

    elif voice_action == "hold":
        fused_action = "hold_current_target"
        execution_mode = "direct_execute"
        reasoning = "hold command"

    return {
        "cmd": "multimodal_intent",
        "timestamp": time.time(),
        "source": "python_receiver",
        "voice": {
            "raw_text": voice_info["raw_text"],
            "normalized_text": voice_info["normalized_text"],
            "voice_action": voice_action,
            "deictic": deictic,
            "valid": voice_info["valid"],
            "matched_command_key": voice_info.get("matched_command_key"),
            "matched_phrase": voice_info.get("matched_phrase", ""),
            "match_score": voice_info.get("match_score", 0.0),
            "stt_confidence": stt_confidence,
        },
        "gaze": {
            "display_xy": list(gaze_point) if gaze_point is not None else None,
            "src_pixel": list(clamp_src_pixel(gaze_point)) if gaze_point is not None else None,
            "valid": gaze_valid,
            "avg_window_sec": GAZE_AVG_WINDOW_SEC,
        },
        "current_target": {
            "display_xy": list(current_target_pixel) if current_target_pixel is not None else None,
            "valid": target_valid,
        },
        "fusion": {
            "gaze_matches_current_target": gaze_matches_target,
            "pixel_threshold": GAZE_TARGET_MATCH_PX,
            "fused_action": fused_action,
            "execution_mode": execution_mode,
            "reasoning": reasoning,
        }
    }


def overlay_frame(frame: np.ndarray,
                  metadata: Dict[str, Any],
                  fps: float,
                  state: SharedState,
                  pupil_point: Optional[Tuple[int, int]]) -> np.ndarray:
    vis = cv2.resize(frame, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_LINEAR)

    last_click = state.get_click()
    click_age = state.get_click_age()
    if last_click is not None and click_age <= RAW_CLICK_DISPLAY_SEC:
        cx, cy = last_click
        cv2.drawMarker(vis, (cx, cy), (0, 0, 255),
                       markerType=cv2.MARKER_TILTED_CROSS, markerSize=16, thickness=2)
        cv2.circle(vis, (cx, cy), 14, (0, 0, 255), 2)

    if pupil_point is not None:
        gx, gy = pupil_point
        cv2.drawMarker(vis, (gx, gy), GAZE_CURSOR_COLOR,
                       markerType=cv2.MARKER_STAR, markerSize=16, thickness=2)
        cv2.circle(vis, (gx, gy), 12, GAZE_CURSOR_COLOR, 1)

    flash_gaze = state.get_flash_gaze_point()
    if flash_gaze is not None:
        fx, fy = flash_gaze
        cv2.drawMarker(vis, (fx, fy), (0, 255, 255),
                       markerType=cv2.MARKER_SQUARE, markerSize=18, thickness=2)
        cv2.circle(vis, (fx, fy), 16, (0, 255, 255), 1)

    current_target_pixel = parse_current_target_pixel(metadata)
    if current_target_pixel is not None and state.is_world_target_marker_active():
        tx, ty = current_target_pixel
        cv2.drawMarker(vis, (tx, ty), (0, 255, 255),
                       markerType=cv2.MARKER_DIAMOND, markerSize=18, thickness=2)
        cv2.circle(vis, (tx, ty), 14, (0, 255, 255), 2)

        center_pt = (DISPLAY_W // 2, DISPLAY_H // 2)
        if dist_px(center_pt, (tx, ty)) <= CAMERA_TARGET_REACHED_PX:
            state.set_world_target_marker_active(False)

    cv2.drawMarker(vis, (DISPLAY_W // 2, DISPLAY_H // 2), (255, 255, 255),
                   markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)

    last_voice_text, _ = state.get_last_voice_text()
    fusion_result = state.get_last_fusion_result()

    fused_action = fusion_result.get("fusion", {}).get("fused_action", "N/A")
    execution_mode = fusion_result.get("fusion", {}).get("execution_mode", "N/A")
    match_score = fusion_result.get("voice", {}).get("match_score", 0.0)
    matched_phrase = fusion_result.get("voice", {}).get("matched_phrase", "N/A")
    voice_valid = fusion_result.get("voice", {}).get("valid", False)

    ack_text, ack_code, ack_time, ack_key = state.get_last_intent_ack()
    if not ack_text:
        ack_text = "N/A"

    confirm_pending = metadata.get("confirm_pending", False)
    voice_busy = state.is_voice_busy()

    top_lines = [
        f"Status           {state.get_status()}",
        f"FPS              {fps:.2f}",
        f"Voice Busy       {voice_busy}",
        f"Confirm Pending  {confirm_pending}",
        f"UAV Position     {fmt_vec(metadata.get('uav_pos', None), 3, 2)}",
        f"Target Position  {fmt_vec(metadata.get('desired_target_world', None), 3, 2)}",
        f"Voice            {last_voice_text if last_voice_text else 'N/A'}",
        f"Voice Valid      {voice_valid}",
        f"Cmd Match        {match_score:.2f}",
        f"Matched Phrase   {matched_phrase}",
        f"Fused Action     {fused_action}",
        f"Exec Mode        {execution_mode}",
        f"UAV Ack Key      {ack_key if ack_key else 'N/A'}",
        f"UAV Ack          {ack_text}",
        f"Ack Code         {ack_code if ack_code else 'N/A'}",
    ]

    draw_text_block_pil(vis, top_lines, x=14, y=20, line_h=22, font=FONT_SMALL)

    if ack_text != "N/A" and (time.time() - ack_time) <= ACK_POPUP_SEC:
        draw_center_ack_popup(vis, ack_text)

    save_text, save_time = state.get_save_status()
    if save_text and (time.time() - save_time) <= SAVE_POPUP_SEC:
        vis = pil_draw_text(vis, save_text, (20, DISPLAY_H - 40), FONT_SMALL, fill=(0, 255, 0))

    if state.is_calibration_mode():
        idx = state.get_calibration_index()
        tx = int(CALIBRATION_POINTS_NORM[idx][0] * DISPLAY_W)
        ty = int(CALIBRATION_POINTS_NORM[idx][1] * DISPLAY_H)

        cv2.drawMarker(vis, (tx, ty), (0, 255, 0),
                       markerType=cv2.MARKER_CROSS, markerSize=28, thickness=3)
        cv2.circle(vis, (tx, ty), 20, (0, 255, 0), 2)

        cal_lines = [
            "Calibration Mode",
            f"Point {idx + 1} / {len(CALIBRATION_POINTS_NORM)}",
            "Look at green target and press SPACE",
            "Press C to restart, ENTER to fit, R to reset",
        ]
        draw_text_block_pil(vis, cal_lines, x=DISPLAY_W - 390, y=28, line_h=22,
                            font=FONT_SMALL, bg_color=(0, 40, 0), alpha=0.35)

    footer = "LClick: Retarget  RClick: Cancel  Space: Gaze+Voice Fusion / Cal Sample  C: Start Cal  Enter: Fit  R: Reset  S: Snapshot  Q: Quit"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.30
    thickness = 1
    (tw, th), _ = cv2.getTextSize(footer, font, font_scale, thickness)

    bx = 14
    by = DISPLAY_H - 10

    overlay = vis.copy()
    cv2.rectangle(overlay, (bx - 6, by - th - 8), (bx + tw + 8, by + 6), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.34, vis, 0.66, 0, vis)
    cv2.putText(vis, footer, (bx, by), font, font_scale, (225, 225, 225), thickness, cv2.LINE_AA)

    return vis


def main():
    ensure_csv_log_header()

    state = SharedState()
    frame_queue = queue.Queue(maxsize=MAX_FRAME_QUEUE)
    retarget_sender = RetargetSender(RETARGET_HOST, RETARGET_PORT)
    voice_processor = VoiceProcessor()
    voice_worker = VoiceFusionWorker(state, retarget_sender, voice_processor)

    video_thread = VideoReceiverThread(VIDEO_HOST, VIDEO_PORT, frame_queue, state)
    metadata_thread = MetadataListenerThread(METADATA_BIND_HOST, METADATA_PORT, state)
    pupil_thread = PupilCoreThread(PUPIL_REMOTE_IP, PUPIL_REMOTE_PORT, state)

    video_thread.start()
    metadata_thread.start()
    pupil_thread.start()
    voice_worker.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_W, DISPLAY_H)

    last_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(last_frame, "Waiting for video...", (150, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    def mouse_callback(event, x, y, flags, param):
        state.set_mouse_pos(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            state.set_click(x, y)
            retarget_sender.send_retarget(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            retarget_sender.send_cancel()

    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    fps = 0.0
    frame_counter = 0
    fps_timer = time.time()
    last_vis = cv2.resize(last_frame, (DISPLAY_W, DISPLAY_H))

    try:
        while True:
            try:
                frame = frame_queue.get(timeout=0.01)
                last_frame = frame
                frame_counter += 1
                state.clear_flash_gaze_point()
            except queue.Empty:
                frame = last_frame

            now = time.time()
            if now - fps_timer >= 1.0:
                fps = frame_counter / (now - fps_timer)
                frame_counter = 0
                fps_timer = now

            metadata = state.get_metadata()
            pupil_point = map_pupil_to_screen(state)

            if pupil_point is not None:
                state.push_gaze_point(pupil_point, time.time())

            vis = overlay_frame(last_frame, metadata, fps, state, pupil_point)
            last_vis = vis.copy()
            cv2.imshow(WINDOW_NAME, vis)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("s"):
                save_snapshot(last_vis, state)
            elif key == ord("c"):
                state.start_calibration()
            elif key == ord("r"):
                state.reset_calibration()
            elif key == 13:
                A = compute_affine(state.get_calibration_samples())
                if A is not None:
                    state.set_affine_matrix(A)
                    state.set_save_status("Calibration fitted")
                else:
                    state.set_save_status("Calibration fit failed")
            elif key == 32:
                if state.is_calibration_mode():
                    pupil_norm, pupil_time = state.get_fused_pupil_norm()
                    if pupil_norm is not None and (time.time() - pupil_time) < 1.0:
                        state.add_calibration_sample(pupil_norm)
                        state.advance_calibration_index()
                        state.set_save_status("Calibration sample added")
                    else:
                        state.set_save_status("No valid pupil sample")
                else:
                    if not state.is_voice_busy():
                        gaze_point = state.get_recent_gaze_average(GAZE_AVG_WINDOW_SEC)
                        state.set_flash_gaze_point(gaze_point)

                        metadata = state.get_metadata()
                        current_target_pixel = parse_current_target_pixel(metadata)
                        ok = voice_worker.submit(gaze_point, current_target_pixel)
                        if ok:
                            state.set_status("VOICE_QUEUED")
                        else:
                            state.set_save_status("Voice worker busy")
                    else:
                        state.set_save_status("Voice worker busy")

    finally:
        state.stop()
        time.sleep(0.2)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()