"""原始文档存储（文件系统）。

职责：按知识库分目录保存上传的 PDF/Word 原件。
目录结构：data/uploaded_docs/{知识库名称}/{文件名}

不负责：文本解析、格式转换。
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import List, Optional

import config

_ILLEGAL_CHARS = re.compile(r'[\\/:*?"<>|\r\n\t]')


def safe_name(name: str, fallback: str = "kb") -> str:
    """把任意字符串转换为合法且稳定的文件/目录名。"""
    cleaned = _ILLEGAL_CHARS.sub("_", (name or "").strip())
    cleaned = cleaned.strip(" .")
    return cleaned or fallback


class DocumentStore:
    """原始文档的本地归档。"""

    def __init__(self, base_dir: Optional[str | Path] = None) -> None:
        self.base_dir = Path(base_dir or config.UPLOAD_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def kb_dir(self, kb_name: str) -> Path:
        directory = self.base_dir / safe_name(kb_name)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def save(self, src_path: str | Path, kb_name: str) -> Path:
        """把上传的临时文件归档到知识库目录，返回归档后的绝对路径。"""
        src = Path(src_path)
        if not src.exists():
            raise FileNotFoundError(f"上传文件不存在：{src}")

        ext = src.suffix.lower()
        stem = safe_name(src.stem, "document")
        target_dir = self.kb_dir(kb_name)
        target = target_dir / f"{stem}{ext}"

        index = 1
        while target.exists():
            target = target_dir / f"{stem}_{index}{ext}"
            index += 1

        shutil.copy2(src, target)
        return target.resolve()

    def delete(self, kb_name: str, filename: str) -> bool:
        path = self.base_dir / safe_name(kb_name) / safe_name(filename)
        if path.exists() and path.is_file():
            path.unlink()
            return True
        return False

    def delete_by_path(self, file_path: str | Path) -> bool:
        path = Path(file_path)
        if path.exists() and path.is_file():
            path.unlink()
            return True
        return False

    def exists(self, file_path: str | Path) -> bool:
        return Path(file_path).exists()

    def list_files(self, kb_name: str) -> List[Path]:
        directory = self.base_dir / safe_name(kb_name)
        if not directory.exists():
            return []
        return sorted(p for p in directory.iterdir() if p.is_file())

    # ------------------------------------------------------------------
    def export_copy(
        self,
        file_path: str | Path,
        export_dir: Optional[str | Path] = None,
        name: str = "",
    ) -> Optional[Path]:
        """把原件复制到导出目录，供浏览器下载（浏览器无法直接跳转 file:// 本地路径）。

        :param name: 指定导出文件名（隐私模式下用打码名，避免下载副本暴露原件名）。
        """
        src = Path(file_path)
        if not src.exists():
            return None
        target_dir = Path(export_dir or config.EXPORT_DIR)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / (name or src.name)
        if not target.exists() or target.stat().st_size != src.stat().st_size:
            shutil.copy2(src, target)
        return target.resolve()
