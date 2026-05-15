# SRTP 复现 — 双目 RGB-D 物体位姿追踪

基于 FoundationStereo + FoundationPose 的双目 6DoF 物体位姿追踪优化管线。

## 目录结构

```
Reproduction/
├── reference.pdf                  # 参考论文
├── Fast-FoundationStereo/         # 立体匹配 → 深度图
├── FoundationPose/                # 6DoF 物体位姿估计与追踪
├── data/
│   ├── clip03/                    # 1231 帧 ZED-M 双目序列
│   └── clip04/                    # 1323 帧 ZED-M 双目序列
├── scripts/
│   ├── run_ffs_batch.py           # Step 2: FFS 批量深度生成
│   ├── run_fp_track.py            # Step 2.5: FoundationPose 追踪 (left/right)
│   ├── run_demo_save_depth.py     # FFS 单帧推理（被 run_ffs_batch 调用）
│   └── convert_depth_npy_to_png.py # 深度格式转换 (npy → uint16 PNG)
└── README.md
```

## 数据格式 (clip03/clip04)

```
clipXX/
├── calib.json                     # 双目内参 + baseline (≈6.3cm)
├── rgb/*.jpg                      # 左 RGB (1920×1080, rectified)
├── right/*.jpg                    # 右 RGB (1920×1080, rectified)
├── ffs/
│   ├── cam_K.txt                  # 左相机 3×3 内参
│   └── depth/*.png                # FFS 深度图 (uint16 mm) — 核心中间产物
├── mask/
│   ├── object/*.png               # 物体 mask (uint8, 0/255)
│   └── hand/*.png                 # 手部 mask
├── mesh/
│   ├── clean_mesh.obj             # 目标物体 3D 网格
│   ├── clean_mesh.mtl
│   └── clean_texture.png
├── foundationpose/run/            # 师兄的单目 FP 基线结果
│   ├── ob_in_cam/NNNNN.txt        # 每帧 4×4 位姿矩阵
│   ├── track_vis/NNNNN.png        # 追踪可视化叠加图
│   └── scales/unified_scale.txt   # mesh 缩放因子
└── foundationpose_v2/
    ├── run/ob_in_cam/NNNNN.txt    # 你的 FP 左目输出
    └── run_right/ob_in_cam/...    # 你的 FP 右目输出
```

## 环境

### FoundationPose (Docker)

```bash
cd FoundationPose/docker
bash run_container.sh              # 启动容器（已挂载 /mnt:/mnt）
docker exec -it foundationpose bash  # 进入容器
```

### Fast-FoundationStereo (conda)

```bash
conda activate ffs
cd Fast-FoundationStereo
```

## Pipeline 总览

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: FFS 深度生成                                           │
│  run_ffs_batch.py                                               │
│  ┌──────────┐  ┌───────────┐  ┌───────────┐                    │
│  │ rgb/*.jpg │  │ right/*.jpg│  │ cam_K.txt │  calib.json      │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘                  │
│        └──────────────┬──────────────┘                          │
│                       ▼                                         │
│              Fast-FoundationStereo                              │
│                       │                                         │
│                       ▼                                         │
│               ffs/depth/*.png  (uint16 mm, left-camera)         │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2.5: FoundationPose 追踪                                  │
│  run_fp_track.py --camera left|right                            │
│                                                                 │
│  Left camera:                                                   │
│    rgb/*.jpg + ffs/depth/*.png + mask/object/*.png → pose_left │
│                                                                 │
│  Right camera:                                                  │
│    right/*.jpg + ffs/depth/*.png (warped via baseline)          │
│                + mask/object/*.png (warped) → pose_right        │
│                                                                 │
│  输出: foundationpose_v2/run[_right]/ob_in_cam/NNNNN.txt        │
└─────────────────────────────────────────────────────────────────┘
```

## 运行步骤

### Step 2: FFS 批量深度生成

从左右目 RGB 生成深度图，保存为 `ffs/depth/*.png`（uint16 mm）。

```bash
conda activate ffs
cd F:/Research/02_Projects/SRTP/Reproduction

# 前 10 帧测试
python scripts/run_ffs_batch.py --clip clip03 --end_frame 10

# 全量（自动跳过已存在的 depth PNG）
python scripts/run_ffs_batch.py --clip clip03

# 强制覆盖已有 depth
python scripts/run_ffs_batch.py --clip clip03 --overwrite

# 仅预览（不实际运行）
python scripts/run_ffs_batch.py --clip clip03 --dry_run
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--clip` | `clip03` | 数据子目录 |
| `--start_frame` | `0` | 起始帧 |
| `--end_frame` | `-1` | 结束帧（-1=全部） |
| `--overwrite` | `false` | 覆盖已有 depth PNG |
| `--dry_run` | `false` | 仅列出待处理帧 |
| `--scale` | `1.0` | 图像缩放比例 |
| `--save_intermediate` | `false` | 保存 disp.npy / cloud.ply |

输出：
- `data/{clip}/ffs/depth/*.png` — uint16 mm 深度图（**核心产物**，直接喂给 FP）

### Step 2.5: FoundationPose 位姿追踪

以 `ffs/depth/*.png` 为深度输入，运行 FoundationPose 做 6DoF 追踪。

```bash
# 进入 FoundationPose Docker 容器
docker exec -it foundationpose bash
cd /mnt/f/Research/02_Projects/SRTP/Reproduction

# === 左目追踪 ===
python scripts/run_fp_track.py --clip clip03 --camera left --debug 1 --end_frame 5
python scripts/run_fp_track.py --clip clip03 --camera left --debug 2   # 全量

# === 右目追踪（深度由左目 + baseline 自动 warp，无需额外数据）===
python scripts/run_fp_track.py --clip clip03 --camera right --debug 1 --end_frame 5
python scripts/run_fp_track.py --clip clip03 --camera right --debug 2
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--clip` | `clip03` | 数据子目录 |
| `--camera` | `left` | `left`（左目）/ `right`（右目，深度由 baseline warp） |
| `--shorter_side` | `800` | 下采样分辨率（8GB 显存: 800；>12GB: 1080） |
| `--debug` | `1` | 0=无输出；1=弹窗预览；2=保存 track_vis |
| `--start_frame` | `0` | 起始帧 |
| `--end_frame` | `-1` | 结束帧（-1=全部） |
| `--debug_dir` | auto | 输出目录（默认 `foundationpose_v2/run[_right]/`） |
| `--est_refine_iter` | `5` | 初始帧估计迭代次数 |
| `--track_refine_iter` | `2` | 追踪帧迭代次数 |
| `--zfar` | `2.0` | 深度远裁剪面（米） |
| `--mesh_scale` | auto | mesh 缩放因子（自动从 baseline 读取） |

输出：
- `foundationpose_v2/run/ob_in_cam/NNNNN.txt` — 左目 4×4 位姿
- `foundationpose_v2/run_right/ob_in_cam/NNNNN.txt` — 右目 4×4 位姿
- `foundationpose_v2/run[_right]/track_vis/NNNNN.png` — 可视化 (`--debug 2`)

### Step 3: 双目几何约束位姿精修（TODO）

以左右目独立位姿为初值，引入双目标定参数进行多视角联合优化。

### Step 4: 单目 vs 双目 对比评价（TODO）

右目重投影误差、时序抖动、定性对比。

## 关键数据流说明

- **唯一中间产物**：`ffs/depth/*.png`（uint16 mm 左相机深度）
- **FFS batch** 产出 → `ffs/depth/*.png`
- **FP left** 直接消费 → `ffs/depth/*.png`
- **FP right** 通过 stereo baseline 将左深度 warp 到右视图（不需要 disparity / raw_rerun）
- **深度 warp 原理**：rectified stereo 中，左像素 `(u,v)` 深度 Z → 右像素 `(u - fx·baseline/Z, v)`

## 已知修改

对 `FoundationPose/Utils.py` 做了两处适配（mesh 缺少 UV 坐标时降级到 vertex color）：

- L106: 增加 `and mesh.visual.uv is not None` 条件
- L119-123: `vertex_colors` 访问改用 `getattr` 安全读取
