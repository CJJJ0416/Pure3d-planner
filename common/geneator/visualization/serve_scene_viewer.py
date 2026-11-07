#!/usr/bin/env python3
"""提供共享场景网页服务、RViz 启动和随机 GPMP 路线生成。"""
import argparse
import errno
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import uuid
from urllib.parse import urlsplit, parse_qs

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))
from stair_nav.core.io import read_json, write_json, sha256 as sha256_file
from stair_nav.visualization.viewer_selection import selection_record
from stair_nav.visualization.data import DATA_FILE, ensure_viewer_data, shared_asset
from stair_nav.visualization.random_maps import RandomMapGenerator


def rviz_environment():
    """启动时读取用户桌面环境，兼容网页服务早于图形登录启动的情况。"""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    try:
        desktop = subprocess.run(["systemctl", "--user", "show-environment"],
                                 check=True, capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return env
    # 只更新桌面连接变量；ROS、Python 和 CUDA 仍使用本服务的环境。
    names = {"DISPLAY", "WAYLAND_DISPLAY", "XAUTHORITY", "XDG_RUNTIME_DIR",
             "DBUS_SESSION_BUS_ADDRESS", "XDG_SESSION_TYPE"}
    for line in desktop.stdout.splitlines():
        name, separator, value = line.partition("=")
        if separator and name in names and value:
            env[name] = value
    return env


class RandomRouteGenerator:
    """管理一个可替换随机路线结果，以及一个尚未完成的后台任务。

    current 保存成功的 route.json/原生曲线，.pending 接收新结果，.previous 用于交换回滚。
    生成与相机渲染由独立子进程完成，只有进程退出并验证成功后才发布 ready。
    与地图管理器共用可重入锁，避免换图期间启动仍引用旧场景的新路线。
    """
    def __init__(self, scene, on_replace=None, lock=None):
        self.scene = Path(scene).resolve()
        self.scene_seed = read_json(self.scene/"scene.json")["seed"]
        self.folder = self.scene/"random_routes"
        self.current, self.pending = self.folder/"current", self.folder/".pending"
        self.previous = self.folder/".previous"
        self.process, self.job_id = None, None
        self.running = False
        self.lock = lock or threading.RLock()
        self.on_replace = on_replace
        self.last = dict(state="idle", message="优先在不同楼层随机选点")
        self.migrate()

    @staticmethod
    def remove(path):
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path)

    def migrate(self):
        """将旧的多任务目录迁移为一个 current，并恢复中断的交换。

        只把来源场景和 query ID 相符的完整结果作为候选，优先保留最新一个。
        旧重复预览附件和临时槽位随后清理；迁移统计写到缓存，不改固定 GPMP 查询。
        """
        self.folder.mkdir(parents=True, exist_ok=True)
        before = sum(p.stat().st_size for p in self.folder.rglob("*") if p.is_file())
        if not self.current.exists() and self.previous.exists():
            self.previous.rename(self.current)
        legacy, complete = [], []
        for path in self.folder.iterdir():
            if path.is_dir() and not path.is_symlink() and re.fullmatch(r"[0-9a-f]{32}", path.name):
                try:
                    report = read_json(path/"route.json")["report"]
                    if report["source_scene"] == str(self.scene) and report["query"] == "random_"+path.name:
                        legacy.append(path)
                        complete.append(path)
                except (OSError, ValueError, KeyError):
                    # 旧失败任务可能只有日志，没有可恢复的回放结果。
                    if (path/"generation.log").is_file():
                        legacy.append(path)
        if not self.current.exists() and complete:
            latest = max(complete, key=lambda p: (p/"route.json").stat().st_mtime_ns)
            latest.rename(self.current)
        for path in legacy:
            self.remove(path)
        self.remove(self.pending)
        self.remove(self.previous)
        # 旧版导出重复包含图像和页面，当前仅保留
        # JSON 回放、原生曲线与小型生成记录。
        trimmed_exports = False
        if self.current.exists():
            trimmed_exports = (self.current/"replay").exists()
            self.remove(self.current/"replay")
            data = read_json(self.current/"route.json")
            self.job_id = data["report"]["query"].removeprefix("random_")
            self.last = dict(state="ready", message="已载入当前随机路线")
        write_json(self.folder/"job.json", dict(self.last, job_id=self.job_id))
        after = sum(p.stat().st_size for p in self.folder.rglob("*") if p.is_file())
        if legacy or trimmed_exports:
            write_json(ROOT/".cache/viewer/random_route_storage_migration.json",
                       dict(scene=str(self.scene), before_bytes=before, after_bytes=after,
                            removed_legacy_jobs=len(legacy), retained_query="random_"+self.job_id if self.job_id else None))

    def status(self):
        """返回当前任务与仍可用旧结果的状态，必要时收尾已退出进程。

        临时状态已 ready 但进程未退出时仍报告 rendering；新任务失败或未完成时，
        current_result_url 仍可指向上一条完整路线，网页可继续回放。
        """
        with self.lock:
            if self.running and self.process.poll() is not None:
                self.finish(self.process, self.job_id)
            result = dict(self.last, scene=str(self.scene), scene_seed=self.scene_seed, job_id=self.job_id)
            if self.running:
                try:
                    result.update(read_json(self.pending/"status.json"))
                except (OSError, ValueError):
                    result.update(state="planning", message="正在启动随机路线生成…")
                # 状态文件先写 ready 仍不算完成，必须等渲染子进程退出。
                if result["state"] == "ready":
                    result.update(state="rendering", message="正在完成渲染并替换上一条路线…")
            result["log_url"] = ("/random_routes/.pending/generation.log" if self.running else
                                 "/random_routes/current/generation.log" if result["state"] == "ready" else
                                 "/random_routes/last_generation.log")
            if self.current.exists():
                current_query = read_json(self.current/"route.json")["report"]["query"]
                result.update(current_query=current_query,
                              current_result_url="/random_routes/current/route.json?v="+current_query)
            if result["state"] == "ready":
                result["result_url"] = result["current_result_url"]
            return result

    def launch(self):
        """启动固定随机路线脚本，并注册等待退出的守护线程。

        当前任务仍运行时直接返回其状态，不重复启动。新任务使用 UUID 标识及独立
        进程组，所有产物先写 .pending，当前成功路线在生成期间保持可用。
        """
        with self.lock:
            if self.running:
                if self.process.poll() is None:
                    return self.status()
                self.finish(self.process, self.job_id)
            self.remove(self.pending)
            self.pending.mkdir()
            self.job_id = uuid.uuid4().hex
            self.last = dict(state="planning", message="正在随机生成，完成后替换当前路线…")
            env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
            try:
                with (self.pending/"generation.log").open("w") as log:
                    self.process = subprocess.Popen([sys.executable, str(ROOT/"common/geneator/planning/generate_random_route.py"),
                        str(self.scene), str(self.pending), "--job-id", self.job_id], cwd=ROOT, env=env,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            except OSError:
                self.remove(self.pending)
                self.last = dict(state="error", message="无法启动随机路线生成，上一条路线已保留")
                raise
            self.running = True
            write_json(self.folder/"job.json", dict(self.last, job_id=self.job_id))
            threading.Thread(target=self.watch, args=(self.process, self.job_id), daemon=True).start()
            return self.status()

    def finish(self, process, job_id, error=None):
        """核对进程、任务 ID、退出码、ready 状态及路线所属场景后安装结果。

        新曲线验证成功才交换 current/.pending；重命名失败会恢复旧目录。
        安装后触发可选的 RViz 选线回调，回调读取失败不撤销已经成功生成的路线，
        而是在状态中提示刷新。最后释放 running，并保存用于前端轮询的 job.json。
        """
        with self.lock:
            if not self.running or process is not self.process or job_id != self.job_id:
                return
            try:
                status = read_json(self.pending/"status.json")
                if error or process.returncode != 0 or status["state"] != "ready":
                    raise RuntimeError(error or status.get("message", "生成失败，上一条路线已保留"))
                route = read_json(self.pending/"route.json")
                if route["report"]["query"] != "random_"+job_id or route["report"]["source_scene"] != str(self.scene):
                    raise ValueError("生成结果与本次任务不一致")
                self.remove(self.previous)
                if self.current.exists():
                    self.current.rename(self.previous)
                try:
                    self.pending.rename(self.current)
                except BaseException:
                    if self.previous.exists():
                        self.previous.rename(self.current)
                    raise
                self.remove(self.previous)
                self.last = status
                if self.on_replace:
                    try:
                        self.on_replace(route["report"]["query"])
                    except (OSError, ValueError, KeyError):
                        self.last = dict(status, message=status["message"]+"；请刷新 RViz 当前路线")
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                if (self.pending/"generation.log").exists():
                    shutil.copyfile(self.pending/"generation.log", self.folder/"last_generation.log")
                self.remove(self.pending)
                self.last = dict(state="error", message=str(exc))
            finally:
                self.running = False
                write_json(self.folder/"job.json", dict(self.last, job_id=self.job_id))

    def watch(self, process, job_id):
        """最多等待 240 秒；超时终止任务进程组，随后统一执行失败收尾。"""
        error = None
        try:
            process.wait(timeout=240)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            error = "生成超过 4 分钟，上一条路线已保留，请再次随机"
        self.finish(process, job_id, error)

    def close(self):
        """服务退出时停止本任务的进程组，等待退出后保留上一个完整路线结果。"""
        if self.running:
            if self.process.poll() is None:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait()
            self.finish(self.process, self.job_id, "生成已中止，上一条路线已保留")


class RvizLauncher:
    """按当前场景维护 ROS 发布器/RViz 进程与网页选线握手文件。

    每个场景路径对应独立缓存文件；selection 表达期望路线，ready 由 ROS 发布端
    回写。只有发布器 PID 及选中 query 均匹配时，状态才从 starting 变为 running。
    """
    def __init__(self, scene):
        self.token = secrets.token_urlsafe(32)
        self.process = None
        self.lock = threading.RLock()
        self.configure_scene(scene)

    def configure_scene(self, scene):
        """为场景建立日志/握手缓存路径，选择第一条预设路线并写入 selection。"""
        self.scene = Path(scene).resolve()
        scene = self.scene
        key = hashlib.sha256(str(scene).encode()).hexdigest()[:16]
        self.cache = ROOT/".cache/viewer"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.log = self.cache/f"rviz_{key}.log"
        self.ready = self.cache/f"rviz_{key}.ready.json"
        self.selection = self.cache/f"rviz_{key}.selection.json"
        model = read_json(scene/"scene.json")
        self.scene_seed = model["seed"]
        self.selected = selection_record(scene, model["queries"][0]["id"])
        write_json(self.selection, self.selected)

    def switch_scene(self, scene):
        """关闭旧场景的发布进程，更新场景缓存；若此前已打开则自动重启显示。"""
        with self.lock:
            reopen = self.process is not None and self.process.poll() is None
            self.close()
            self.process = None
            self.configure_scene(scene)
            if reopen:
                try:
                    self.launch()
                except OSError as exc:
                    self.log.write_text(str(exc))

    def status(self):
        """结合子进程存活状态、ready 文件 PID 与 query 确认 RViz 同步状态。

        不把残留 ready 文件或另一进程的状态视为当前场景已就绪，返回状态附带前端操作 token。
        """
        state, message, ready = "idle", "尚未打开 RViz", None
        if self.process is not None:
            code = self.process.poll()
            if code is None:
                state, message = "starting", "正在打开 RViz…"
                if self.ready.exists():
                    try:
                        ready = json.loads(self.ready.read_text())
                        if ready.get("publisher_pid") == self.process.pid:
                            state, message = "running", "RViz 已启动，可查看当前场景及 VST 图层"
                            if ready.get("selected_query") != self.selected["query"]:
                                state, message = "starting", "正在同步 RViz 当前路线…"
                            else:
                                message = "RViz 当前路线："+self.selected["label"]
                    except (OSError, ValueError):
                        pass
            elif code == 0:
                state, message = "closed", "RViz 已关闭，可再次打开"
            else:
                state, message = "error", "RViz 启动失败，请查看本地启动日志"
        return dict(state=state, message=message, scene=str(self.scene), scene_seed=self.scene_seed, token=self.token,
                    ready=ready, log=str(self.log), selected_query=self.selected["query"])

    def select(self, query):
        """仅接受可解析的已保存 query，原子写入 selection 并返回同步状态。"""
        with self.lock:
            selected = selection_record(self.scene, query)
            write_json(self.selection, selected)
            self.selected = selected
            return self.status()

    def launch(self):
        """复用已有发布进程，或通过固定 rviz.sh 入口启动新进程组并传入握手路径。"""
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                return self.status()
            self.ready.unlink(missing_ok=True)
            env = rviz_environment()
            with self.log.open("w") as log:
                self.process = subprocess.Popen([str(ROOT/"rviz.sh"), str(self.scene),
                    "--ready-file", str(self.ready), "--selection-file", str(self.selection)], cwd=ROOT, env=env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            return self.status()

    def close(self):
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()


class SceneSession:
    """协调共享网页、当前地图、随机路线与 RViz 的同一场景视图。

    以同一 RLock 保护地图交换和路由读取；服务重启优先恢复完整 current 地图。
    地图切换时重新建立该场景的随机路线管理器，并让 RViz 跟随新的源目录。
    """
    def __init__(self, scene):
        scene = Path(scene).resolve()
        self.lock = threading.RLock()
        self.maps = RandomMapGenerator(scene, self.lock, self.install_scene,
            lambda: hasattr(self, "generator") and self.generator.running)
        active = self.maps.current if self.maps.complete(self.maps.current) else scene
        ensure_viewer_data(active, remove_legacy_html=(active/"random_map.json").is_file())
        self.launcher = RvizLauncher(active)
        self.generator = RandomRouteGenerator(active, self.replace_selected, self.lock)
        self.lidar = None

    def lidar_preview(self):
        # 仅在页面请求时加载雷达及其可选依赖，普通相机回放不提前初始化射线查询。
        if self.lidar is None:
            from stair_nav.visualization.lidar import LidarPreview
            self.lidar = LidarPreview()
        return self.lidar

    def replace_selected(self, query):
        if self.launcher.selected["query"].startswith("random_"):
            self.launcher.select(query)

    def install_scene(self, directory):
        """安装后迁移/检查回放数据，切换 RViz 及随机路线管理器到新目录。

        由地图管理器在共享锁内调用，不在这里重新生成地图或相机图像。
        """
        ensure_viewer_data(directory, remove_legacy_html=(directory/"random_map.json").is_file())
        generator = RandomRouteGenerator(directory, self.replace_selected, self.lock)
        self.launcher.switch_scene(directory)
        self.generator = generator

    def close(self):
        self.maps.close()
        self.generator.close()
        self.launcher.close()


class SceneHandler(SimpleHTTPRequestHandler):
    """本地 HTTP 页面、共享资源和受限生成操作的路由入口。

    共享 JS/CSS/HTML 使用 ETag，场景 JSON 使用 no-store；场景切换与读取共享锁。
    POST 同时检查本机 Host/Origin、会话 token 和小型 JSON 负载，只接受固定操作
    及路线 ID，不能让请求指定服务端任意脚本或文件路径。
    """
    def __init__(self, *args, session, **kwargs):
        self.session = session
        super().__init__(*args, directory=str(session.launcher.scene), **kwargs)

    @property
    def launcher(self):
        return self.session.launcher

    @property
    def generator(self):
        return self.session.generator

    def local_request(self):
        """核对 Host 为当前端口的 localhost/127.0.0.1，Origin 若存在也必须一致。"""
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        return host in hosts and (origin is None or origin == "http://"+host)

    def reply(self, code, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def serve_file(self, path, content_type, shared=False):
        """发送共享资源或一代完整的场景数据，支持 HEAD 而不发送正文。

        shared=True 时按文件指纹提供 ETag/304，其他数据明确 no-store。
        调用方在读取可替换场景数据时持有会话锁；函数保持已打开文件流直到发送完成。
        """
        try:
            with Path(path).open("rb") as stream:
                etag = '"'+sha256_file(path)+'"' if shared else None
                if shared and self.headers.get("If-None-Match") == etag:
                    self.send_response(304)
                    self.send_header("ETag", etag)
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(os.fstat(stream.fileno()).st_size))
                self.send_header("Cache-Control", "no-cache" if shared else "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                if etag:
                    self.send_header("ETag", etag)
                self.end_headers()
                if self.command != "HEAD":
                    shutil.copyfileobj(stream, self.wfile)
        except FileNotFoundError:
            self.reply(404, dict(message="查看资源不存在"))

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        """把固定页面/资源/API 路由到当前会话；读取当前随机路线时持锁避免目录混读。"""
        if not self.local_request():
            return self.reply(403, dict(message="仅支持当前本地查看页面"))
        endpoint = urlsplit(self.path).path
        if endpoint == "/api/health":
            return self.reply(200, dict(service="glocal2-scene-viewer", state="ok", pid=os.getpid(),
                                        root=str(ROOT)))
        if endpoint in ("/", "/index.html", "/viewer/index.html"):
            return self.serve_file(shared_asset("index.html"), "text/html; charset=utf-8", shared=True)
        if endpoint.startswith("/viewer/"):
            name = endpoint.removeprefix("/viewer/")
            if name not in ("viewer.css", "viewer.js", "lidar.js", "plotly.min.js"):
                return self.reply(404, dict(message="查看资源不存在"))
            content_type = "text/css" if name.endswith(".css") else "text/javascript"
            return self.serve_file(shared_asset(name), content_type+"; charset=utf-8", shared=True)
        if endpoint == "/api/viewer-data":
            with self.session.lock:
                return self.serve_file(self.launcher.scene/DATA_FILE, "application/json; charset=utf-8")
        if endpoint in ("/api/lidar", "/api/lidar/frames"):
            try:
                with self.session.lock:
                    sensor = self.session.lidar_preview()
                    if endpoint == "/api/lidar":
                        result = sensor.describe(self.launcher.scene)
                    else:
                        params = parse_qs(urlsplit(self.path).query, strict_parsing=True)
                        if set(params) != {"scene_key", "query", "start", "count", "pitch"} or any(len(v) != 1 for v in params.values()):
                            raise ValueError("LiDAR 请求参数无效")
                        p = {key: values[0] for key, values in params.items()}
                        result = sensor.frames(self.launcher.scene, p["scene_key"], p["query"],
                                               int(p["start"]), int(p["count"]), int(p["pitch"]))
                return self.reply(200, result)
            except RuntimeError as exc:
                return self.reply(409, dict(message=str(exc)))
            except (ValueError, KeyError) as exc:
                return self.reply(400, dict(message=str(exc)))
            except (OSError, ImportError) as exc:
                return self.reply(503, dict(message="LiDAR 预览不可用", detail=str(exc)))
        if urlsplit(self.path).path == "/api/rviz":
            return self.reply(200, self.launcher.status())
        if urlsplit(self.path).path == "/api/random-route":
            return self.reply(200, dict(**self.generator.status(), token=self.launcher.token))
        if urlsplit(self.path).path == "/api/random-map":
            return self.reply(200, dict(**self.session.maps.status(), token=self.launcher.token))
        if urlsplit(self.path).path.startswith("/random_routes/current/"):
            # 静态读取与目录切换共用锁，避免读到混合的新旧数据。
            with self.generator.lock:
                return super().do_HEAD() if self.command == "HEAD" else super().do_GET()
        return super().do_HEAD() if self.command == "HEAD" else super().do_GET()

    def do_POST(self):
        """解析受限的生成/选线请求，并返回 200/202、冲突或参数错误。

        只允许固定 API，检查本地来源、token、Content-Length 及 {query} 结构。
        地图生成与路线生成互斥；请求只负责启动任务，完整退出和安装状态由后续轮询确认。
        """
        endpoint = urlsplit(self.path).path
        if endpoint not in ("/api/rviz", "/api/rviz/selection", "/api/random-route", "/api/random-map"):
            return self.reply(404, dict(message="接口不存在"))
        if not self.local_request() or not secrets.compare_digest(
                self.headers.get("X-Viewer-Token", ""), self.launcher.token):
            return self.reply(403, dict(message="请从当前场景页面发起操作"))
        # 请求只允许选择已保存的路线 ID，不接受可执行程序或任意路径。
        if self.headers.get("Transfer-Encoding"):
            return self.reply(400, dict(message="不支持分块请求"))
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > 512 or (endpoint in ("/api/random-route", "/api/random-map") and length):
                return self.reply(400, dict(message="请求参数无效"))
            with self.session.lock:
                if endpoint == "/api/random-map":
                    return self.reply(202, self.session.maps.launch())
                if endpoint == "/api/random-route" and self.session.maps.running:
                    return self.reply(409, dict(message="正在生成新地图，请完成后再随机选点"))
            if length:
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict) or set(payload) != {"query"}:
                    return self.reply(400, dict(message="仅支持选择当前场景的路线 ID"))
                with self.generator.lock:
                    selected = self.launcher.select(payload["query"])
            else:
                selected = self.launcher.status()
            if endpoint == "/api/rviz/selection":
                return self.reply(200, selected)
            with self.session.lock:
                if endpoint == "/api/random-route" and self.session.maps.running:
                    return self.reply(409, dict(message="正在生成新地图，请完成后再随机选点"))
                return self.reply(202, self.launcher.launch() if endpoint == "/api/rviz" else self.generator.launch())
        except RuntimeError as exc:
            return self.reply(409, dict(message=str(exc)))
        except (ValueError, KeyError) as exc:
            return self.reply(400, dict(message=str(exc)))
        except OSError as exc:
            return self.reply(500, dict(state="error", message="无法启动本地操作", detail=str(exc)))


def main():
    """加载已有场景并启动仅绑定 127.0.0.1 的多线程查看服务。

    若入口为随机 current 地图，先还原其模板位置以恢复固定槽位管理。
    必须已有查看数据或可迁移旧 HTML；退出时关闭服务和它启动的后台任务。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path, help="Scene directory containing viewer_data.json or a legacy index.html")
    parser.add_argument("--port", type=int, default=8769)
    args = parser.parse_args()
    scene = args.scene.resolve()
    if scene.name == "current" and scene.parent.name == "random_maps" and (scene/"random_map.json").is_file():
        scene = scene.parent.parent
    for name in ("scene.json", "terrain.pcd"):
        if not (scene/name).is_file():
            parser.error(f"Missing saved scene file: {scene/name}")
    if not any((scene/name).is_file() for name in (DATA_FILE, "index.html")):
        parser.error("Missing viewer_data.json; run common/geneator/visualization/build_scene_viewer.py <scene> first")
    # 先占用监听端口，再恢复场景；重复启动不能清理正在使用的 .pending 任务。
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), SceneHandler)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            parser.exit(2, f"Port {args.port} is already in use; reuse the running viewer or stop its service first.\n")
        raise
    session = None
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def stop_service(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_service)
    try:
        session = SceneSession(scene)
        server.RequestHandlerClass = partial(SceneHandler, session=session)
        print(f"Scene: {session.launcher.scene}\nViewer: http://127.0.0.1:{server.server_port}/index.html\nRViz log: {session.launcher.log}", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if session is not None:
            session.close()
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
