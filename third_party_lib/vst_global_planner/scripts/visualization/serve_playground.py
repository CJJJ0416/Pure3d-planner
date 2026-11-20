#!/usr/bin/env python3
"""Use glocal2-nav's viewer with region-constrained playground route generation."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import threading
import uuid

ROOT=Path(__file__).resolve().parents[2]
BACKEND=ROOT.parents[1]
sys.path[:0]=[str(ROOT/"src"),str(BACKEND/"src")]
from vst_global_planner.route_viewer import read_json
from vst_global_planner.supervision import write_json_utf8

spec=importlib.util.spec_from_file_location("glocal_playground_viewer",BACKEND/"scripts/visualization/serve_scene_viewer.py")
viewer=importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


class PlaygroundRandomRoutes(viewer.RandomRouteGenerator):
    def launch(self):
        with self.lock:
            if self.running:
                if self.process.poll() is None: return self.status()
                self.finish(self.process,self.job_id)
            scene=read_json(self.scene/"scene.json")
            query=self.selected_query()
            if query.startswith("random_"):
                region=read_json(self.current/"route.json")["report"]["region_id"]
            else:
                region=next(q["region_id"] for q in scene["queries"] if q["id"]==query)
            self.remove(self.pending)
            self.pending.mkdir()
            self.job_id=uuid.uuid4().hex
            self.last=dict(state="planning",message="正在当前建筑内生成随机路线…")
            env=dict(os.environ,PYTHONUNBUFFERED="1",PYTHONUTF8="1",OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4")
            try:
                with (self.pending/"generation.log").open("w") as log:
                    self.process=subprocess.Popen([sys.executable,"-X","utf8",
                        str(ROOT/"scripts/planning/generate_viewer_random_route.py"),
                        str(self.scene),str(self.pending),"--region",region,"--job-id",self.job_id],
                        cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            except OSError:
                self.remove(self.pending)
                self.last=dict(state="error",message="无法启动随机路线生成，已保存路线保留")
                raise
            self.running=True
            write_json_utf8(self.folder/"job.json",dict(self.last,job_id=self.job_id))
            threading.Thread(target=self.watch,args=(self.process,self.job_id),daemon=True).start()
            return self.status()


class PlaygroundSession(viewer.SceneSession):
    def __init__(self,scene):
        super().__init__(scene)
        self.generator.selected_query=lambda:self.launcher.selected["query"]


def assets():
    folder=ROOT/".cache/viewer8769"
    folder.mkdir(parents=True,exist_ok=True)
    original=viewer.shared_asset
    html=original("index.html").read_text(encoding="utf-8")
    html=html.replace('id="policy-link"','id="policy-link" style="display:none"')
    html=html.replace('class="randombar"><button id="random-map"','class="randombar" style="display:none"><button id="random-map"')
    html=html.replace("随机两点并生成路线","在当前建筑内随机生成路线")
    html=html.replace("优先不同楼层 · 自动规划和相机回放","限定当前建筑 · 上下行 / 穿行随机选点")
    html=html.replace("随机路线使用当前场景的 VST 地图生成；仅有一个可用楼层时选择同层点。未执行路径几何碰撞验收、动力学仿真或策略训练。",
        "随机路线限定在当前建筑的 VST 裁剪地图内。已通过离线支撑与上部空间采样检查，未执行动力学仿真或学生策略。")
    html=html.replace('value="0.16"','value="0.45"')
    html=html.replace('<button id="reset">恢复视角</button>',
        '<button id="reset">恢复视角</button><button id="fit-all">全场景</button><button id="fit-region">当前建筑</button>')
    (folder/"index.html").write_text(html,encoding="utf-8")
    script=original("viewer.js").read_text(encoding="utf-8")
    script=script.replace("function region(p){","function region(p){\n if(D.report.region_label)return D.report.region_label;",1)
    script=script.replace("固定平台场景 · 可随机两点生成路线","固定 Playground · 随机路线限定当前建筑")
    marker="$('route').onchange=()=>{try{selectRoute(+$('route').value);}catch(error){$('random-status').textContent=error.message;}};"
    extra="""
$('fit-region').onclick=()=>{const s=D.figure.layout.scene;Plotly.relayout(plot,{'scene.xaxis.range':s.xaxis.range,'scene.yaxis.range':s.yaxis.range,'scene.zaxis.range':s.zaxis.range,'scene.aspectmode':'manual','scene.aspectratio':s.aspectratio,'scene.camera':s.camera});};
$('fit-all').onclick=()=>{const w=D.scene.width,d=D.scene.depth,h=D.scene.height+1.3;Plotly.relayout(plot,{'scene.xaxis.range':[-w/2,w/2],'scene.yaxis.range':[-d/2,d/2],'scene.zaxis.range':[-.3,h],'scene.aspectmode':'manual','scene.aspectratio':{x:1,y:d/w,z:h*1.7/w},'scene.camera':{eye:{x:1.15,y:-1.35,z:1.1}}});};
"""
    if marker not in script: raise RuntimeError("Shared viewer route-selection interface changed")
    script=script.replace(marker,marker+extra,1)
    (folder/"viewer.js").write_text(script,encoding="utf-8")
    viewer.shared_asset=lambda name:folder/name if name in ("index.html","viewer.js") else original(name)


if __name__=="__main__":
    assets()
    viewer.RandomRouteGenerator=PlaygroundRandomRoutes
    viewer.SceneSession=PlaygroundSession
    viewer.main()
