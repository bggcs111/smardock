"""路由决策器（纯代码）。

职责：判断用户是否选定了检索范围。
逻辑：
    指定了单篇文档 -> 范围已明确，直接返回该文档所属知识库
    勾选了知识库   -> 返回勾选的知识库列表
    都没有选       -> 直接返回空，系统不调用大模型，提示用户先选择知识库

说明：不做自动路由，完全由用户手动选择。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

NO_KB_MESSAGE = "请先在左侧勾选至少一个知识库，然后再提问。"


@dataclass
class RouteResult:
    kbs: List[str] = field(default_factory=list)
    ok: bool = False
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"kbs": list(self.kbs), "ok": self.ok, "message": self.message}


class Router:
    """纯条件判断的路由决策器。"""

    def __init__(self, available_provider: Optional[Any] = None) -> None:
        # available_provider: 可调用的知识库名称提供者，用于过滤已不存在的知识库
        self.available_provider = available_provider

    def route(
        self,
        selected_kbs: Optional[Iterable[str]] = None,
        scoped_doc: Optional[Dict[str, Any]] = None,
    ) -> RouteResult:
        """决定本次问答的检索范围。

        :param scoped_doc: 单篇文档问答时传入该文档记录（需含 ``kb_name``）；
                           此时忽略知识库多选框，因为文档已隐含其归属知识库。
        """
        if scoped_doc:
            kb_name = str(scoped_doc.get("kb_name") or "").strip()
            if not kb_name:
                return RouteResult(kbs=[], ok=False, message="所选文档缺少归属知识库信息，无法检索。")
            available = set(self._available_kbs())
            if available and kb_name not in available:
                return RouteResult(
                    kbs=[],
                    ok=False,
                    message=f"文档所属知识库 `{kb_name}` 已不存在，请重新上传文档。",
                )
            return RouteResult(kbs=[kb_name], ok=True, message="")

        selected = [str(kb).strip() for kb in (selected_kbs or []) if str(kb or "").strip()]
        # 去重并保持顺序
        selected = list(dict.fromkeys(selected))

        if not selected:
            return RouteResult(kbs=[], ok=False, message=NO_KB_MESSAGE)

        available = set(self._available_kbs())
        if available:
            valid = [kb for kb in selected if kb in available]
            missing = [kb for kb in selected if kb not in available]
            if not valid:
                return RouteResult(
                    kbs=[],
                    ok=False,
                    message=f"所选知识库已不存在：{'、'.join(missing)}。请重新勾选。",
                )
            if missing:
                return RouteResult(
                    kbs=valid,
                    ok=True,
                    message=f"以下知识库已不存在，已自动忽略：{'、'.join(missing)}",
                )
            return RouteResult(kbs=valid, ok=True, message="")

        return RouteResult(kbs=selected, ok=True, message="")

    def _available_kbs(self) -> List[str]:
        if self.available_provider is None:
            return []
        try:
            return list(self.available_provider())
        except Exception:
            return []
