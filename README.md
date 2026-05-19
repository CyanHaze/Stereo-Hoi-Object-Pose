# Stereo FoundationPose — 双目 HOI 物体位姿估计

基于双目视频 + 已重建 mesh，估计准确、连续、稳定的物体 6D pose 序列。

**核心思路**：将 AGILE 的 setting 从单目扩展到双目 — 用 Fast-FoundationStereo 获取更可靠的双目深度，用 FoundationPose 在左右视角上独立 tracking，再融合两个视角的结果。

## 环境

| 环境 | 用途 |
|------|------|
| `conda activate ffs` | Fast-FoundationStereo 深度推理 |
| `conda activate diffusion` | WiLoR 手部 mesh 推理 + HOI 可视化 |
| FoundationPose Docker 容器 | FoundationPose tracking + 融合可视化 (`--vis`) |

> 纯融合（不加 `--vis`）不需要 Docker，任意有 numpy+scipy 的 Python 环境均可。

## 数据准备

每个 clip 目录需包含：

```
data/<clip>/
├── rgb/                    # 左视角帧  00000.jpg ...
├── right/                  # 右视角帧  00000.jpg ...
├── calib.json              # 双目标定 (K_left, K_right, baseline_m)
├── mask/object/            # 物体 mask  00000.png ...
├── mesh/clean_mesh.obj     # 带纹理物体 mesh
└── ffs/cam_K.txt           # 左相机内参 (3行 3x3)
```

帧编号 5 位 0-padding（`00000.jpg` ...），所有模态共享同一编号。

### 坐标系与单位

- 深度图：**uint16 PNG，单位毫米**（÷1000 = 米）
- 位姿 `ob_in_cam`：**4×4 齐次矩阵**，平移单位**米**
- 相机坐标系：OpenCV/ZED 约定（+X 右、+Y 下、+Z 前）
- 双目已 rectified，`K_left == K_right`，畸变全 0
- `calib.json::baseline_m` ≈ 6.3 cm，右相机在左相机 +X 方向
- WiLoR 预训练模型放在 `WiLoR/pretrained_models/` 下（detector.pt + wilor_final.ckpt）
- MANO 手部模型 `MANO_RIGHT.pkl` 放在 `WiLoR/mano_data/` 下（需从 mano.is.tue.mpg.de 手动下载）

### calib.json

```json
{
  "K_left":  [[fx,0,cx],[0,fy,cy],[0,0,1]],
  "K_right": [...],
  "D_left":  [0,...,0],
  "D_right": [...],
  "baseline_m": 0.0630453,
  "width": 1920, "height": 1080, "fps": 30
}
```

## Pipeline

```
双目视频
  ├── 物体位姿 ─────────────────────────────────────────
  │   [1] run_ffs_batch.py        → ffs/depth/ (左视图深度)
  │   [2] run_fp_track.py  left   → foundationpose_v2/run/
  │   [3] run_fp_track.py  right  → foundationpose_v2/run_right/
  │   [4] run_fusion.py           → foundationpose_v2/fused/
  │
  ├── 手部 mesh ────────────────────────────────────────
  │   [5] run_wilor_hand.py  left → wilor/left/*.npz
  │
  ├── 可视化 & 浏览 ────────────────────────────────────
  │   [6] build_comparison_video.py  → track.mp4 + 对比视频
  │   [7] render_hoi.py              → hoi/video_frames/*.png
  │   [8] hoi_viewer.py              → Viser 交互式 3D (localhost)
  │
  └── 网页 Demo ───────────────────────────────────────
      [9] export_web_demo.py --rgb   → web_demo/ 静态 Three.js
```

### Step 1: 双目深度

```bash
conda activate ffs
cd F:/Research/02_Projects/SRTP/Reproduction

# 测试前 10 帧
python scripts/run_ffs_batch.py --clip clip03 --end_frame 10

# 全序列 (自动跳过已有结果)
python scripts/run_ffs_batch.py --clip clip03

# 强制覆盖
python scripts/run_ffs_batch.py --clip clip03 --overwrite
```

输出 → `data/<clip>/ffs/depth/*.png` (uint16 mm)

### Step 2: FoundationPose 左视角 tracking

```bash
docker exec -it foundationpose bash
# FoundationPose Docker 容器内
cd /mnt/f/Research/02_Projects/SRTP/Reproduction

# 测试前 5 帧
python scripts/run_fp_track.py --clip clip03 --camera left --debug 1 --end_frame 5

# 全序列 + 保存可视化
python scripts/run_fp_track.py --clip clip03 --camera left --debug 2
```

输出 → `foundationpose_v2/run/ob_in_cam/*.txt` + `track_vis/` + `video_frames/`

### Step 3: FoundationPose 右视角 tracking

```bash
python scripts/run_fp_track.py --clip clip03 --camera right --debug 2
```

右视角深度由左视角 FFS 深度通过 stereo baseline **自动 warp**，不需要单独跑 FFS。

输出 → `foundationpose_v2/run_right/ob_in_cam/*.txt` + `track_vis/` + `video_frames/`

### Step 4: 多视角融合

**必须先跑完 Step 2 + Step 3**。`run_fusion.py` 不跑 FoundationPose，只读取已有 pose 文件。

```bash
# 推荐：outlier 剔除 + Gaussian 时域平滑
python scripts/run_fusion.py --clip clip03 --method average --smooth 7

# 纯融合 (无 outlier 剔除)
python scripts/run_fusion.py --clip clip03 --no_outlier

# 更严格的 outlier 阈值
python scripts/run_fusion.py --clip clip03 --outlier_trans 0.03 --outlier_rot 15

# 融合 + 可视化 (需要 FP Docker)
python scripts/run_fusion.py --clip clip03 --vis

# 消融实验
python scripts/run_fusion.py --clip clip03 --method left_only --vis
python scripts/run_fusion.py --clip clip03 --method right_only --vis
python scripts/run_fusion.py --clip clip03 --method left_main --vis
```

| `--method` | 说明 |
|---|---|
| `average` | 等权平均 (quaternion mean + translation mean) |
| `left_main` | 默认用 left，仅当左右一致时平均 (平移差<2cm & 旋转差<10°) |
| `left_only` | 仅用左视角 (消融基线) |
| `right_only` | 仅用右视角变换到左坐标系 (sanity check) |

| `--outlier_*` | 默认 | 说明 |
|---|---|---|
| `--outlier_trans` | 0.05 (5cm) | 左右平移差超过此值 → 剔除该帧右视角 |
| `--outlier_rot` | 30 (deg) | 左右旋转差超过此值 → 剔除该帧右视角 |
| `--no_outlier` | - | 关闭 outlier 剔除 |

| `--smooth` | 说明 |
|---|---|
| `--smooth 7` | 时域平滑窗口（奇数；默认 `gaussian` 核） |
| `--smooth_method moving_avg` | 改用均匀窗口平滑 |

### Step 5: 生成对比视频

**推荐在 FP Docker 容器内运行**（可同时渲染 + 合成；容器外只能合成已有帧）。

```bash
# 全流程：渲染缺失帧 + 合成 MP4 + 左右并排对比
python scripts/build_comparison_video.py --clip clip03

# 仅合成（不渲染，video_frames 已就绪时使用）
python scripts/build_comparison_video.py --clip clip03 --mode compose

# 仅渲染缺失帧
python scripts/build_comparison_video.py --clip clip03 --mode render

# 指定帧范围 / 帧率
python scripts/build_comparison_video.py --clip clip03 --start_frame 0 --end_frame 100 --fps 15
```

输出视频：
| 文件 | 内容 |
|---|---|
| `run/track.mp4` | 左视角 tracking 结果叠加 RGB |
| `run_right/track.mp4` | 右视角 tracking 结果叠加 RGB |
| `fused/track.mp4` | 融合后位姿叠加 RGB |
| `comparison_left_fused.mp4` | 左右并排（left-only \| fused） |

### Step 6: WiLoR 手部 mesh 推理

**需要先准备好 WiLoR 环境**（`conda activate diffusion`，依赖见 `WiLoR/requirements.txt`）。

```bash
conda activate diffusion
cd F:/Research/02_Projects/SRTP/Reproduction

# 测试前 10 帧（保存手部 mesh 叠加图）
python scripts/run_wilor_hand.py --clip clip03 --camera left --end_frame 10 --debug

# 全序列
python scripts/run_wilor_hand.py --clip clip03 --camera left --debug

# 手部筛选参数（过滤误检为人手的脸部等）
python scripts/run_wilor_hand.py --clip clip03 --camera left --max_hands 2 --hand_top_margin 0.12
```

| `--max_hands` | 默认 | 说明 |
|---|---|---|
| `--max_hands 2` | 2 | 每帧最多保留的手部检测数。按左右手分组，每组取最高置信度；超过时取置信度前 N |
| `--hand_top_margin 0.12` | 0.12 | 位置过滤：丢弃 bbox 中心在图像高度前 12% 的检测（通常为脸部误检）；设 0 关闭 |

每帧输出 `data/<clip>/wilor/left/<frame>.npz`，包含：

| 字段 | 形状 | 说明 |
|------|------|------|
| `verts_mano` | (N, 778, 3) | MANO 手部 mesh 顶点（米） |
| `verts_virt` | (N, 778, 3) | WiLoR 虚拟相机坐标（用于渲染） |
| `verts_cam` | (N, 778, 3) | metric 相机坐标（深度对齐后） |
| `joints` / `joints_2d` | (N, 21, 3) / (N, 21, 2) | 3D/2D 手部关节 |
| `wrist_3d` | (N, 3) | metric 手腕位置（相机坐标系，米） |
| `depth_ok` | (N,) | 深度对齐是否成功 |

### Step 7: 手+物合并渲染

**前置**：需先跑完 Step 2/4（物体 pose）和 Step 6（手部数据）。

```bash
conda activate diffusion
cd F:/Research/02_Projects/SRTP/Reproduction

# 测试前 10 帧
python scripts/render_hoi.py --clip clip03 --end_frame 10

# 全序列 + 合成 MP4
python scripts/render_hoi.py --clip clip03 --fps 30
```

输出 `data/<clip>/hoi/video_frames/*.png`（`--fps N` 时同时生成 `track.mp4`）。

渲染内容：右手（橙）、左手（青）、物体 mesh（绿半透明）、物体 3D bbox（绿线框）、手腕标记点（红点）。

### Step 8: 交互式 3D 浏览 (Viser)

**前置**：需先跑完 Step 4（融合 pose）和 Step 5（手部数据）。需要 `viser` 包（`pip install viser`）。数据加载和坐标转换逻辑由 `scripts/hoi_data.py` 提供。

```bash
conda activate diffusion
cd F:/Research/02_Projects/SRTP/Reproduction

# 启动 Viser 服务器，浏览器中自由旋转/缩放/平移视角
python scripts/hoi_viewer.py --clip clip03

# 自定义端口、播放速度
python scripts/hoi_viewer.py --clip clip03 --port 8081 --fps 15
```

打开浏览器访问 `http://localhost:8080` 即可交互查看。

**场景内容**：物体 mesh（绿半透明）、物体 3D bbox（绿线框）、左手 mesh（橙）、右手 mesh（青）、手腕标记点（红）、相机原点坐标轴、参考网格。

**操作**：鼠标左键旋转、右键平移、滚轮缩放；下方面板控制播放/暂停、帧滑块、各元素显隐与透明度。

数据加载和坐标转换逻辑提取在 `scripts/hoi_data.py`，被 `hoi_viewer.py`、`export_web_demo.py` 等共用。

### Step 9: 网页 Demo

**前置**：需先跑完 Step 4（融合 pose）和 Step 5（手部数据）。

```bash
conda activate diffusion
cd F:/Research/02_Projects/SRTP/Reproduction

# 导出 clip03（每3帧采样，含 RGB 缩略图）
python scripts/export_web_demo.py --clip clip03 --step 3 --rgb

# 启动本地 HTTP 服务
cd web_demo && python -m http.server 8080
```

浏览器打开 `http://localhost:8080`，页面包括：

- **Hero**：标题 + feature cards
- **3D 画布**：物体 mesh + bbox + 左手 mesh + 手腕圆点，OrbitControls 自由视角
- **RGB Overlay**：Opacity 滑块控制原始 RGB 底图透明度，可对比 3D 重建与视频
- **控制栏**：播放/暂停、逐帧、FPS 调节、Object/BBox/Hands/Wrist 显隐开关
- **180 帧缩略图条**：底部可翻页缩略图条（每页 15 帧），单击跳帧，红色边框标注当前帧
- **键盘**：Space 播放/暂停，← → 逐帧

> 更改 `--step` 可调整导出帧数（`--step 6` ≈ 180 帧）。

## 输出结构

```
data/<clip>/foundationpose_v2/
├── run/                     # 左视角 tracking
│   ├── ob_in_cam/*.txt      # 4x4 位姿矩阵 (左相机坐标系, 米)
│   ├── track_vis/*.png      # mesh 渲染视图
│   ├── video_frames/*.png   # RGB 叠加视图
│   └── track.mp4            # RGB 叠加视频
├── run_right/               # 右视角 tracking (结构同上)
│   └── track.mp4
├── fused/                   # 融合结果
│   ├── ob_in_cam/*.txt      # 4x4 位姿矩阵 (左相机坐标系)
│   ├── track_vis/*.png      # (需 --vis)
│   ├── video_frames/*.png   # (需 --vis)
│   └── track.mp4            # 融合位姿视频
└── comparison_left_fused.mp4  # 左右并排对比 (left | fused)

data/<clip>/wilor/
├── left/                            # 左手视角手部数据
│   └── *.npz                        # 每帧手部 mesh + 关节 + metric 位置
└── video_frames/                    # 手部 mesh 叠加图 (--debug)

data/<clip>/hoi/
└── video_frames/                    # 手+物合并渲染帧 (*.png)

web_demo/
├── index.html                       # 学术风格页面 (Three.js)
└── static/
    ├── js/viewer.js                 # 核心 JS：加载资产 + 逐帧渲染
    └── results/<clip>/
        ├── metadata.json            # 帧列表、路径
        ├── object_mesh.glb          # 物体 mesh (静态)
        ├── object_frames.json       # 每帧 4×4 矩阵 + bbox 线段
        ├── hand_frames.json         # 每帧手部顶点 (Three.js 坐标)
        ├── mano_faces.json          # MANO 面索引
        ├── rgb_index.json           # RGB/缩略图索引
        ├── rgb/                     # 全尺寸 RGB 帧 (1920×1080)
        └── thumbnails/              # 200px 缩略图 (180 帧条)
```

## 脚本

| 脚本 | 功能 | 环境 |
|------|------|------|
| `scripts/run_ffs_batch.py` | 批量双目深度估计 | conda ffs |
| `scripts/run_fp_track.py` | FoundationPose 单视角 tracking | FP Docker |
| `scripts/run_fusion.py` | 左右 pose 多视角融合 | 任意 Python① |
| `scripts/run_demo_save_depth.py` | FFS 单帧推理 (被 run_ffs_batch 调用) | conda ffs |
| `scripts/convert_depth_npy_to_png.py` | depth npy → uint16 PNG 转换 | 任意 Python |
| `scripts/build_comparison_video.py` | 渲染+合成对比视频 | 任意 Python② |
| `scripts/run_wilor_hand.py` | WiLoR 批量手部 mesh 推理 + metric 对齐 | conda diffusion |
| `scripts/render_hoi.py` | 手+物合并渲染（mesh + bbox 叠加） | conda diffusion |
| `scripts/hoi_viewer.py` | Viser 交互式 3D 浏览器（自由视角） | conda diffusion |
| `scripts/hoi_data.py` | 共享模块：数据加载、坐标转换、MANO faces | conda diffusion |
| `scripts/export_web_demo.py` | 导出静态网页资产（GLB + JSON + 缩略图） | conda diffusion |

> ① `--vis` 需要 FP Docker
> ② `render` 模式需 `trimesh`(可选，AABB fallback)；`compose` 模式需 `ffmpeg`

## 安装依赖

### conda diffusion 环境（WiLoR 手部推理 + HOI 可视化）

```bash
conda activate diffusion
pip install smplx==0.1.28 pytorch-lightning timm einops xtcocotools \
    hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils rich \
    webdataset yacs scikit-image pyrender gradio ultralytics==8.1.34
pip install --no-build-isolation "chumpy @ git+https://github.com/mattloper/chumpy"
```

### MANO 模型（手动下载）

1. 注册 [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de)
2. 下载 `mano_v*_*.zip`，解压得到 `MANO_RIGHT.pkl`
3. 放到 `WiLoR/mano_data/MANO_RIGHT.pkl`

## 技术说明

### 右视角深度 (Step 3)

右视角深度由左视角 FFS 深度通过 stereo baseline **forward warp** 得到，不单独跑 FFS：

```
u_right = u_left - fx * baseline / Z
```

多左像素映射到同一右像素时取**最小深度**（最近表面优先）。空洞由 nearest-neighbor inpainting 填充。左边缘 ~140 px 的条带在左相机中不可见，该区域深度值为估计值，可能导致该区域的 tracking 质量下降。

### WiLoR 坐标系对齐 (Step 5)

WiLoR 在虚拟相机坐标系（fv ≈ 37500）中输出 MANO hand mesh。对齐到真实 metric 相机坐标系时采用**手腕锚定**而非各向异性缩放：

```
verts_metric = verts_mano - joints[0] + wrist_3d
```

MANO 顶点保持真实米尺度。FFS 深度用于确定手腕的 metric 3D 位置，MANO 手 mesh 整体锚定到该点。

### 渲染坐标系一致性 (Step 7-8)

所有 3D 渲染（2D overlay 和 3D viewer）中，手部和物体均在**左相机坐标系**（OpenCV 约定：+X 右、+Y 下、+Z 前）。`render_hoi.py` 中手部通过真实 K 投影（与物体同一相机模型）；`hoi_viewer.py` 和 web demo 中将坐标转换为 Viser/Three.js 约定（+X 右、+Y 上、+Z 后，即 `to_viser(X,Y,Z) = (X, -Y, -Z)`）。

## 参考

- [Fast-FoundationStereo](https://github.com/NVlabs/FoundationStereo)
- [FoundationPose](https://github.com/NVlabs/FoundationPose)
- [WiLoR](https://github.com/rolpotamias/WiLoR) — End-to-end 3D hand localization and reconstruction (CVPR 2025)
- [MANO](https://mano.is.tue.mpg.de) — Hand model
