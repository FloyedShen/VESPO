import os
import time
from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
from PIL import Image
from typing import Dict, Any, Optional
import requests
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
import glob
import shutil


def reverse_geocode(
        service_url: str,
        lat: float,
        lon: float,
        format: str = "json",
        addressdetails: int = 1,
        max_retries: int = -1,
        retry_delay: float = 1.0
) -> Optional[Dict[str, Any]]:
    """
    Call reverse geocoding service to get address info from coordinates
    Retries until successful or max_retries is reached

    Args:
        service_url: Base URL of the geocoding service
        lat: Latitude
        lon: Longitude
        format: Response format (default: json)
        addressdetails: Include address details (default: 1)
        max_retries: Maximum number of retry attempts (default: 5)
        retry_delay: Delay between retries in seconds (default: 2.0)
    """
    endpoint = f"{service_url.rstrip('/')}/reverse"
    params = {
        'lat': lat,
        'lon': lon,
        'format': format,
        'addressdetails': addressdetails,
        'accept-language': 'en'
    }

    attempt = 0
    while max_retries == -1 or attempt < max_retries:
        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            attempt += 1
            if max_retries != -1 and attempt >= max_retries:
                print(f"Request failed after {max_retries} attempts (lat={lat}, lon={lon}): {e}")
                return None

            print(f"Request failed (attempt {attempt}, lat={lat}, lon={lon}): {e}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)

    return None


def process_geochain_sample(sample: Dict[str, Any], service_url: str, geogusor_dir: str) -> Optional[Dict[str, Any]]:
    """
    处理单个 geochain 样本

    Returns None if the sample should be discarded (e.g., corrupted image or missing file)
    """
    # 构建图像路径
    image_path = os.path.join(
        geogusor_dir,
        'vistas',
        'train_val',
        sample['city'],
        sample['sub_folder'],
        'images',
        f"{sample['key']}.jpg"
    )

    # 检查图像是否存在
    if not os.path.exists(image_path):
        # Discard: image file not found
        return None

    # 加载图像
    try:
        image = Image.open(image_path)
    except Exception as e:
        # Discard: cannot load image (corrupted or invalid format)
        return None

    # 调用逆地理编码服务 (with infinite retry, should not fail)
    # If it does fail after many retries, we keep the sample with empty address fields
    geo_result = reverse_geocode(service_url, sample['lat'], sample['lon'])

    # 提取地址信息 (use empty dict if geocoding failed)
    address = geo_result.get('address', {}) if geo_result is not None else {}

    # 构建返回字典
    processed_sample = {
        # Required fields
        'image': image,
        'lon': sample['lon'],
        'lat': sample['lat'],
        'image_source': 'vistas',
        'source': 'geochain',
        'messages': {}, # Placeholder for messages

        # Nominatim address fields
        'road': address.get('road', '').strip().lower(),
        'suburb': address.get('suburb', '').strip().lower(),
        'ISO3166-2-lvl10': address.get('ISO3166-2-lvl10', '').strip().lower(),
        'city': address.get('city', sample.get('city', '')).strip().lower(),
        'postcode': address.get('postcode', '').strip().lower(),
        'country': address.get('country', '').strip().lower(),

        # Dataset custom fields
        'locatability_score': sample['locatability_score'],
        'class_mapping': sample.get('class_mapping', ''),
    }

    return processed_sample


def preprocess_geochain(
        dataset_split: str,
        service_url: str,
        geogusor_dir: str = None,
        num_workers: int = None,
        chunk_size: int = 500,
        save_merged: bool = True,
        resume: bool = True,
        overwrite: bool = False
):
    """
    流式处理数据集，分块并行处理并保存，支持断点续传

    参数:
        dataset_split: 数据集分割名称
        service_url: 逆地理编码服务的URL地址
        geogusor_dir: GEOGUESSR_DIR 环境变量的路径
        num_workers: 并行工作进程数，默认为 CPU 核心数
        chunk_size: 每个chunk的大小
        save_merged: 是否在处理完成后合并所有chunks并保存完整数据集，默认为True
        resume: 是否从已有的chunks继续处理（断点续传），默认为True
        overwrite: 是否覆盖已存在的数据（True会删除所有已有chunks重新开始），默认为False

    返回:
        如果 save_merged=True: (merged_path, chunk_count)
        如果 save_merged=False: (chunks_dir, chunk_count)
    """
    # 获取数据目录
    if geogusor_dir is None:
        geogusor_dir = os.environ.get('GEOGUESSR_DIR')
        if geogusor_dir is None:
            raise ValueError("请设置 GEOGUESSR_DIR 环境变量或传入 geogusor_dir 参数")

    # 设置工作进程数
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)

    print("加载 geochain 数据集...")
    geochain = load_dataset('sahitiy51/geochain', split=dataset_split)
    print(f"Size of geochain/{dataset_split}: {len(geochain)}")

    output_dir = f"{geogusor_dir}/processed/geochain/"
    chunks_dir = os.path.join(output_dir, f"{dataset_split}_chunks")

    # 处理 overwrite 和 resume 逻辑
    existing_chunks = []
    start_chunk_idx = 0

    if os.path.exists(chunks_dir):
        existing_chunks = sorted(glob.glob(os.path.join(chunks_dir, "chunk_*")))

        if overwrite:
            # 删除所有已有chunks，重新开始
            print(f"检测到 overwrite=True，删除已有的 {len(existing_chunks)} 个chunks...")
            shutil.rmtree(chunks_dir)
            existing_chunks = []
            os.makedirs(chunks_dir, exist_ok=True)
            print("重新开始处理...")
        elif resume and existing_chunks:
            # 从最后一个chunk继续
            start_chunk_idx = len(existing_chunks)
            print(f"检测到 {len(existing_chunks)} 个已有chunks")
            print(f"从 chunk {start_chunk_idx} 继续处理（断点续传）...")
        elif not resume and existing_chunks:
            # 不允许 resume，但有已有数据，报错
            raise ValueError(
                f"检测到已有 {len(existing_chunks)} 个chunks，但 resume=False。\n"
                f"请设置 resume=True 继续处理，或 overwrite=True 重新开始。"
            )

    os.makedirs(chunks_dir, exist_ok=True)

    process_func = partial(process_geochain_sample,
                           service_url=service_url,
                           geogusor_dir=geogusor_dir)

    print(f"\nprocessing {dataset_split} split...")

    # 从断点位置开始
    chunk_count = start_chunk_idx
    start_sample_idx = start_chunk_idx * chunk_size

    if start_sample_idx > 0:
        print(f"跳过前 {start_sample_idx} 个样本（已处理）")

    # 分块处理
    for start_idx in range(start_sample_idx, len(geochain), chunk_size):
        end_idx = min(start_idx + chunk_size, len(geochain))
        chunk = geochain.select(range(start_idx, end_idx))

        chunk_samples = []

        # First, safely collect samples from chunk (handle corrupted images during iteration)
        chunk_data = []
        print(f"Loading chunk data ({start_idx}-{end_idx})...")
        for i in tqdm(range(len(chunk)), desc="Loading samples"):
            try:
                sample = chunk[i]
                chunk_data.append(sample)
            except Exception as e:
                print(f"\nError loading sample {start_idx + i}: {e}")
                continue

        # 并行处理当前chunk
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_func, sample): i
                       for i, sample in enumerate(chunk_data)}

            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"Processing chunk {chunk_count}"):
                try:
                    result = future.result()
                    if result is not None:
                        chunk_samples.append(result)
                except Exception as e:
                    print(f"\n处理样本时出错: {e}")

        # 立即保存当前chunk
        if chunk_samples:
            chunk_dataset = Dataset.from_list(chunk_samples)
            chunk_path = os.path.join(chunks_dir, f"chunk_{chunk_count:04d}")
            chunk_dataset.save_to_disk(chunk_path)
            print(f"保存 chunk {chunk_count} ({len(chunk_samples)} 样本)")
            chunk_count += 1

            # 释放内存
            del chunk_samples
            del chunk_dataset

    print(f"\n总共保存了 {chunk_count} 个chunks到: {chunks_dir}")

    # 如果需要合并所有chunks
    if save_merged and chunk_count > 0:
        print(f"\n正在合并所有 {chunk_count} 个chunks...")
        merged_path = os.path.join(output_dir, dataset_split)

        # 加载所有chunks
        chunk_datasets = []
        for i in tqdm(range(chunk_count), desc="加载chunks"):
            chunk_path = os.path.join(chunks_dir, f"chunk_{i:04d}")
            chunk_dataset = load_from_disk(chunk_path)
            chunk_datasets.append(chunk_dataset)

        # 合并所有chunks
        print("合并数据集...")
        full_dataset = concatenate_datasets(chunk_datasets)

        # 保存合并后的数据集
        print(f"保存完整数据集到: {merged_path}")
        full_dataset.save_to_disk(merged_path)

        print(f"完整数据集已保存，共 {len(full_dataset)} 个样本")
        print(f"可以使用 load_from_disk('{merged_path}') 直接加载")

        return merged_path, chunk_count

    return chunks_dir, chunk_count


def load_processed_geochain(
        dataset_split: str,
        geogusor_dir: str = None,
        from_chunks: bool = False
) -> Dataset:
    """
    加载已处理的 geochain 数据集

    参数:
        dataset_split: 数据集分割名称 (如 'test', 'mini_test')
        geogusor_dir: GEOGUESSR_DIR 环境变量的路径
        from_chunks: 是否从chunks加载（如果没有合并后的数据集，会自动尝试从chunks加载）

    返回:
        Dataset: 加载的数据集
    """
    # 获取数据目录
    if geogusor_dir is None:
        geogusor_dir = os.environ.get('GEOGUESSR_DIR')
        if geogusor_dir is None:
            raise ValueError("请设置 GEOGUESSR_DIR 环境变量或传入 geogusor_dir 参数")

    output_dir = f"{geogusor_dir}/processed/geochain/"
    merged_path = os.path.join(output_dir, dataset_split)
    chunks_dir = os.path.join(output_dir, f"{dataset_split}_chunks")

    # 尝试加载合并后的数据集
    if not from_chunks and os.path.exists(merged_path):
        print(f"从 {merged_path} 加载完整数据集...")
        return load_from_disk(merged_path)

    # 如果没有合并后的数据集或指定从chunks加载，则从chunks加载
    if os.path.exists(chunks_dir):
        print(f"从 {chunks_dir} 加载chunks...")
        chunk_paths = sorted(glob.glob(os.path.join(chunks_dir, "chunk_*")))

        if not chunk_paths:
            raise FileNotFoundError(f"在 {chunks_dir} 中没有找到任何chunks")

        print(f"找到 {len(chunk_paths)} 个chunks")

        # 加载所有chunks
        chunk_datasets = []
        for chunk_path in tqdm(chunk_paths, desc="加载chunks"):
            chunk_dataset = load_from_disk(chunk_path)
            chunk_datasets.append(chunk_dataset)

        # 合并所有chunks
        print("合并数据集...")
        full_dataset = concatenate_datasets(chunk_datasets)
        print(f"加载完成，共 {len(full_dataset)} 个样本")

        return full_dataset

    raise FileNotFoundError(
        f"未找到处理后的数据集。请先运行 preprocess_geochain 处理数据。\n"
        f"查找路径: {merged_path} 或 {chunks_dir}"
    )


# 使用示例
if __name__ == "__main__":
    # 服务地址
    service_url = "http://dsw-notebook-dsw-s9abjgcsvnls12nt68-8080.vpc-t4nptn1gzsxxk04su5mpc.instance-forward.dsw.ap-southeast-1.aliyuncs.com:8080"

    geogusor_dir = os.environ.get('GEOGUESSR_DIR')
    assert geogusor_dir is not None, "GEOGUESSR_DIR is required in environment variables."

    # 示例1: 处理 test 集（支持断点续传）
    result_test, chunk_count_test = preprocess_geochain(
        dataset_split='test',
        service_url=service_url,
        geogusor_dir=geogusor_dir,
        num_workers=16,  # 根据CPU核心数调整
        chunk_size=50000,  # 根据内存大小调整
        save_merged=True,  # 自动合并所有chunks
        resume=True  # 如果中断，重新运行会自动从断点继续（默认为True）
    )

    # 示例2: 处理 mini_test 集
    result_mini, chunk_count_mini = preprocess_geochain(
        dataset_split='mini_test',
        service_url=service_url,
        geogusor_dir=geogusor_dir,
        num_workers=16,
        chunk_size=50000,
        save_merged=True
    )

    # 示例3: 如果需要从头重新处理（删除已有数据）
    # result_test, chunk_count_test = preprocess_geochain(
    #     dataset_split='test',
    #     service_url=service_url,
    #     geogusor_dir=geogusor_dir,
    #     num_workers=16,
    #     chunk_size=10000,
    #     save_merged=True,
    #     overwrite=True  # 删除所有已有数据，从头开始
    # )

    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"\ntest 集: {chunk_count_test} 个 chunks -> {result_test}")
    print(f"mini_test 集: {chunk_count_mini} 个 chunks -> {result_mini}")

    print("\n如需加载数据:")
    print("# 方法1: 使用辅助函数（推荐）")
    print("from preprocess_geochain import load_processed_geochain")
    print("test_dataset = load_processed_geochain('test')")
    print()
    print("# 方法2: 直接使用 load_from_disk")
    print("from datasets import load_from_disk")
    print(f"test_dataset = load_from_disk('{result_test}')")
    print()
    print("\n断点续传说明:")
    print("- 如果处理过程中中断，重新运行脚本会自动从上次中断的地方继续")
    print("- 设置 resume=False 会在检测到已有数据时报错，防止意外覆盖")
    print("- 设置 overwrite=True 会删除所有已有数据，从头开始处理")

