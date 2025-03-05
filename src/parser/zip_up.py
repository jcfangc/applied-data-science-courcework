import os
import zipfile
from pathlib import Path
from typing import NoReturn

from ..definition.const.core import (
    DATA_DIR,  # 假设 DATA_DIR 是 Path 对象 / Assuming DATA_DIR is a Path object
)


def zip_json_files(data_folder: Path, output_zip: Path) -> NoReturn:
    """
    将 data_folder 目录及其子目录下的所有 .json 文件打包到 output_zip 中。
    Zip all .json files in the data_folder and its subdirectories into output_zip.

    :param data_folder: 存放 JSON 文件的根目录 (Path)
                        The root directory containing JSON files (Path)
    :param output_zip: 输出 zip 文件的路径 (Path)
                       The path for the output zip file (Path)
    """
    with zipfile.ZipFile(
        output_zip.as_posix(), mode="w", compression=zipfile.ZIP_DEFLATED
    ) as zf:
        for root, _, files in os.walk(
            data_folder.as_posix()
        ):  # 转换为字符串 / Convert to string
            root_path = Path(
                root
            )  # 确保 root 也是 Path 对象 / Ensure root is a Path object
            for file in files:
                if file.lower().endswith(".json"):
                    file_path = (
                        root_path / file
                    )  # 使用 Path 拼接 / Concatenate using Path
                    arcname = file_path.relative_to(
                        data_folder
                    )  # 计算相对路径 / Compute relative path
                    zf.write(file_path.as_posix(), arcname.as_posix())


def main() -> NoReturn:
    data_folder = DATA_DIR  # 你的 data 文件夹 (Path) / Your data folder (Path)
    output_zip = (
        DATA_DIR / "all_json_codebook.zip"
    )  # 目标 zip 文件 / The target zip file
    zip_json_files(data_folder, output_zip)
    print(
        f"已将 {data_folder} 下的所有 JSON 文件打包到 {output_zip}"
    )  # 输出结果提示 / Print the result message


if __name__ == "__main__":
    main()
