import open3d as o3d
import numpy as np
import os, glob, h5py
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def render_views_array(off_path, num_elev=3, num_azim=2, radius=1.0, img_size=(32,32), normalize=True):
    """
    渲染单个 OFF 文件 → [num_views, H, W, 3]
    使用 GPU 离屏渲染（EGL headless）
    参数:
        off_path: .off 文件路径
        num_elev, num_azim: 相机在球面上的采样数量
        radius: 相机到物体中心的距离 (在归一化后坐标系下)
        img_size: 渲染图像大小
        normalize: 是否将 mesh 归一化到对角线长度=1
    """
    mesh = o3d.io.read_triangle_mesh(off_path)
    if mesh.is_empty():
        return None
    mesh.compute_vertex_normals()

    # ==== 自动归一化 ====
    if normalize:
        bbox = mesh.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        scale = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())  # 对角线长度
        mesh.translate(-center)  # 平移到原点
        mesh.scale(1.0 / scale, center=(0, 0, 0))  # 缩放到单位大小
        center = np.array([0.0, 0.0, 0.0])  # 归一化后中心就是原点
    else:
        center = mesh.get_center()

    H, W = img_size
    renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    renderer.scene.add_geometry("mesh", mesh, mat)

    images = []
    for i in range(num_elev):
        elev = np.pi * (i + 1) / (num_elev + 1)
        for j in range(num_azim):
            azim = 2 * np.pi * j / num_azim
            x = radius * np.sin(elev) * np.cos(azim)
            y = radius * np.cos(elev)
            z = radius * np.sin(elev) * np.sin(azim)
            cam_pos = np.array([x, y, z]) + center

            renderer.scene.camera.look_at(center, cam_pos, [0, 1, 0])
            img = np.asarray(renderer.render_to_image())[:, :, :3].astype(np.uint8)
            images.append(img)

    renderer.scene.clear_geometry()
    return np.stack(images, axis=0)  # [num_views, H, W, 3]


def process_one(args):
    """渲染单个 OFF 文件，返回列表：每个视角一个数组"""
    off_path, label, split, idx = args
    try:
        # ✅ 默认 normalize=True
        views = render_views_array(off_path, num_elev=3, num_azim=2, normalize=True)
        if views is None:
            return None
        result = []
        for v_idx, view in enumerate(views):
            result.append((view, label, 0 if split=="train" else 1, idx, v_idx))
        return result
    except Exception as e:
        print(f"出错: {off_path}, {e}")
        return None


def build_h5_per_view(modelnet_dir, output_dir, num_workers=None, img_size=(32,32)):
    os.makedirs(output_dir, exist_ok=True)
    categories = sorted([d for d in os.listdir(modelnet_dir) if os.path.isdir(os.path.join(modelnet_dir, d))])
    cat2label = {cat: i for i, cat in enumerate(categories)}

    tasks = []
    idx = 0
    for split in ["train", "test"]:
        for cat in categories:
            off_files = glob.glob(os.path.join(modelnet_dir, cat, split, "*.off"))
            for off_path in off_files:
                tasks.append((off_path, cat2label[cat], split, idx))
                idx += 1

    N = len(tasks)
    num_views = 6
    H, W = img_size

    # 为每个视角创建一个 h5 文件
    h5_files = []
    for v in range(num_views):
        path = os.path.join(output_dir, f"view_{v}.h5")
        f = h5py.File(path, 'w')
        f.create_dataset('data', shape=(N, H, W, 3), dtype=np.uint8, compression='gzip')
        f.create_dataset('label', shape=(N,), dtype=np.int64)
        f.create_dataset('split', shape=(N,), dtype=np.int64)
        h5_files.append(f)

    num_workers = num_workers or max(1, cpu_count()-1)
    print(f"使用 {num_workers} 个进程并行渲染，总文件数={N}")

    with Pool(num_workers) as pool:
        for res_list in tqdm(pool.imap_unordered(process_one, tasks), total=N):
            if res_list is None:
                continue
            for view, label, split, idx, v_idx in res_list:
                h5_files[v_idx]['data'][idx] = view
                h5_files[v_idx]['label'][idx] = label
                h5_files[v_idx]['split'][idx] = split

    # 关闭所有 h5 文件
    for f in h5_files:
        f.close()

    print(f"完成保存 {num_views} 个视角数据集到 {output_dir}")


if __name__ == "__main__":
    modelnet_dir = "/projappl/project_2015015/ModelNet10"
    output_dir = "./Data_Sim/ModelNet10_3x2"
    build_h5_per_view(modelnet_dir, output_dir, num_workers=8, img_size=(32,32))
