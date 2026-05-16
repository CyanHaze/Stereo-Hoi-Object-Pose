# Stereo FoundationPose — 双目 HOI 物体位姿估计

基于双目视频 + 已重建 mesh，估计准确、连续、稳定的物体 6D pose 序列。

**核心思路**：将 AGILE 的 setting 从单目扩展到双目 — 用 Fast-FoundationStereo 获取更可靠的双目深度，用 FoundationPose 在左右视角上独立 tracking，再融合两个视角的结果。

## 环境

| 环境 | 用途 |
|------|------|
| `conda activate ffs` | Fast-FoundationStereo 深度推理 |
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
  → [1] run_ffs_batch.py  → ffs/depth/ (左视图深度, uint16 mm)
  → [2] run_fp_track.py --camera left   → foundationpose_v2/run/ob_in_cam/*.txt
  → [3] run_fp_track.py --camera right  → foundationpose_v2/run_right/ob_in_cam/*.txt
  → [4] run_fusion.py   → foundationpose_v2/fused/ob_in_cam/*.txt
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
| `scripts/run_wilor_hand.py` | WiLoR 批量手部 mesh 推理 | conda diffusion |

> ① `--vis` 需要 FP Docker
> ② `render` 模式需 `trimesh`(可选，AABB fallback)；`compose` 模式需 `ffmpeg`

## 参考

- [Fast-FoundationStereo](https://github.com/NVlabs/FoundationStereo)
- [FoundationPose](https://github.com/NVlabs/FoundationPose)
