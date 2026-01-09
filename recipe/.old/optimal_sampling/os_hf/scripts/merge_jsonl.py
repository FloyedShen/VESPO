#!/usr/bin/env python3
"""
合并所有 slice*.jsonl 文件到一个文件
"""
import glob
import os
from pathlib import Path


def merge_jsonl_files(pattern="slice*.jsonl", output_file="merged.jsonl", input_dir=None):
    """
    合并匹配模式的所有 JSONL 文件

    Args:
        pattern: 文件匹配模式，默认 "slice*.jsonl"
        output_file: 输出文件名，默认 "merged.jsonl"
        input_dir: 输入文件所在目录，默认为当前目录
    """
    # 构建完整的搜索路径
    if input_dir:
        search_pattern = os.path.join(input_dir, pattern)
    else:
        search_pattern = pattern

    # 获取所有匹配的文件
    files = sorted(glob.glob(search_pattern))

    if not files:
        print(f"未找到匹配 '{pattern}' 的文件")
        return

    print(f"找到 {len(files)} 个文件:")
    for f in files:
        print(f"  - {f}")

    # 合并文件
    total_lines = 0
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filepath in files:
            with open(filepath, 'r', encoding='utf-8') as infile:
                lines = 0
                for line in infile:
                    outfile.write(line)
                    lines += 1
                total_lines += lines
                print(f"已处理 {filepath}: {lines} 行")

    print(f"\n合并完成!")
    print(f"总共写入 {total_lines} 行到 {output_file}")
    print(f"输出文件大小: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="合并 JSONL 文件")
    parser.add_argument("-i", "--input-dir", default=None,
                        help="输入文件所在目录 (默认: 当前目录)")
    parser.add_argument("-p", "--pattern", default="slice*.jsonl",
                        help="文件匹配模式 (默认: slice*.jsonl)")
    parser.add_argument("-o", "--output", default="merged.jsonl",
                        help="输出文件名 (默认: merged.jsonl)")

    args = parser.parse_args()

    merge_jsonl_files(args.pattern, args.output, args.input_dir)