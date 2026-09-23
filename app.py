"""文档智能问答与归档系统 —— Gradio 主程序。

第五层（交互层）的入口：负责组装界面并绑定事件处理器。
业务编排在 modules/ui/event_handlers.py，配色与样式在 modules/ui/theme.py。

布局：左侧可收起、可拖拽改宽的侧边栏（问答知识库固定可见 + 新建/上传/管理折叠）
      + 中间主内容区（对话流，输入框固定在底部）
      + 右侧可折叠、可拖拽改宽的引用来源面板。

启动方式：
    python app.py
或
    bash start.sh
"""
from __future__ import annotations

import atexit
import os
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402

# Gradio 的上传中转文件统一放进项目内的 data/tmp（必须在 import gradio 之前设置），
# 这样退出服务时可以连它一起清理，不污染系统临时目录。
os.environ.setdefault("GRADIO_TEMP_DIR", str(config.TEMP_DIR))
config.ensure_dirs()

import gradio as gr  # noqa: E402
from modules.ui.event_handlers import (  # noqa: E402
    CANON_KEYS,
    DOC_STATUS_EMPTY,
    DOC_TABLE_HEADERS,
    KB_MODE_CHOICES,
    QUERY_KEYS,
    REFS_EMPTY,
    SCOPE_ALL,
    SESSION_EMPTY,
    SESSION_KEYS,
    STATUS_EMPTY,
    SUMMARY_EMPTY,
    KnowledgeQAApp,
)
from modules.logutil import call_totals, get_logger, setup_logging  # noqa: E402
from modules.ui.theme import APP_CSS, RESIZE_HEAD, SCROLL_JS, build_theme  # noqa: E402


def _assert_binding(name: str, fn: object, inputs: object, outputs: list) -> None:
    """构建阶段校验事件绑定，把「入参数量/输出数量对不上」这类错位挡在启动前。

    历史教训：早期版本手工排布返回元组，数量与 outputs 不一致，
    导致状态文案被写进了其它组件（甚至被当成知识库名称落盘）。
    """
    import inspect

    params = [
        param
        for param in inspect.signature(fn).parameters.values()  # type: ignore[arg-type]
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
        and param.name != "progress"
    ]
    required = sum(1 for param in params if param.default is inspect.Parameter.empty)
    provided = len(inputs or [])  # type: ignore[arg-type]

    if not required <= provided <= len(params):
        raise RuntimeError(
            f"事件绑定 `{name}` 入参数量不匹配：绑定了 {provided} 个，"
            f"处理函数需要 {required}~{len(params)} 个"
        )
    if not outputs:
        raise RuntimeError(f"事件绑定 `{name}` 未声明任何输出组件")


def build_ui(app: KnowledgeQAApp) -> "gr.Blocks":
    with gr.Blocks(title="smardock · 文档智能问答工具") as demo:
        # ------------------------------------------------------------------
        # 左侧：可收起、可拖拽改宽的侧边栏
        # ------------------------------------------------------------------
        with gr.Sidebar(
            open=True,
            width=330,
            position="left",
            label="知识库与文档",
            elem_classes=["kv-sidebar", "kv-sidebar-left"],
        ):
            # 固定区：问答知识库始终可见，不随其它折叠区移动；列表过长时框内滚动
            with gr.Column(elem_classes=["kv-panel", "kv-kb-fixed"]):
                gr.Markdown("问答使用的知识库", elem_classes=["kv-section"])
                kb_checkbox = gr.CheckboxGroup(
                    show_label=False,
                    choices=[],
                    value=[],
                )

            # 低频操作分三个独立折叠区，避免一个 Accordion 里塞太多功能
            with gr.Accordion("新建知识库", open=False, elem_classes=["kv-accordion"]):
                gr.Markdown("新建知识库", elem_classes=["kv-section"])
                kb_name_box = gr.Textbox(
                    label="名称", placeholder="例如：劳动合同"
                )
                kb_mode_box = gr.Radio(
                    label="处理模式（建库时确定，库内文档一律遵循）",
                    choices=KB_MODE_CHOICES,
                    value="normal",
                )
                create_kb_btn = gr.Button("创建知识库", variant="secondary", size="sm")

            with gr.Accordion("上传文档并建立索引", open=False, elem_classes=["kv-accordion"]):
                gr.Markdown("上传文档并建立索引", elem_classes=["kv-section"])
                upload_kb = gr.Dropdown(label="上传到知识库", choices=[], value=None)
                gr.Markdown(
                    "文档处理模式和知识库模式一致。",
                    elem_classes=["kv-hint"],
                )
                file_input = gr.Files(
                    label="PDF / Word 文件",
                    file_count="multiple",
                    file_types=[".pdf", ".docx"],
                    type="filepath",
                )
                with gr.Row():
                    upload_btn = gr.Button("上传并建立索引", variant="primary", scale=3)
                    # 建立索引期间显示，用于中断本次上传（见下方事件绑定）
                    upload_stop_btn = gr.Button(
                        "停止", variant="stop", size="sm", scale=1, visible=False
                    )

                gr.Markdown("索引状态", elem_classes=["kv-section"])
                status_md = gr.Markdown(STATUS_EMPTY, elem_classes=["kv-status"])

            # 索引清单属于"看结果"而非"做操作"，默认折叠，让侧边栏保持安静
            with gr.Accordion("文档与索引", open=False, elem_classes=["kv-accordion"]):
                doc_table = gr.Dataframe(
                    headers=DOC_TABLE_HEADERS,
                    value=[],
                    interactive=False,
                    wrap=True,
                )

                gr.Markdown("重建索引", elem_classes=["kv-section"])
                gr.Markdown(
                    "只有在项目代码中的解析引擎或分块策略更新后，才需要重建索引。",
                    elem_classes=["kv-hint"],
                )
                rebuild_kb = gr.Dropdown(label="选择知识库", choices=[], value=None)
                with gr.Row():
                    # 重建索引按钮改小：文案缩短、scale 从 3 降到 2
                    rebuild_btn = gr.Button(
                        "重建索引", variant="secondary", size="sm", scale=2
                    )
                    # 重建期间显示，用于中断本次重建（见下方事件绑定）
                    rebuild_stop_btn = gr.Button(
                        "停止", variant="stop", size="sm", scale=1, visible=False
                    )
                doc_status_md = gr.Markdown(DOC_STATUS_EMPTY, elem_classes=["kv-status"])

            # 危险操作：默认折叠，且必须勾选确认框才会执行
            with gr.Accordion("危险操作：删除", open=False, elem_classes=["kv-danger"]):
                gr.Markdown(
                    "删除不可恢复，会同时清理已归档原件与向量索引。",
                    elem_classes=["kv-section"],
                )
                # 确认框对整个删除区生效（知识库与文档删除共用），
                # 放在最前面，避免被误以为只对文档删除有效
                confirm_checkbox = gr.Checkbox(
                    label="确认要删除，此操作不可恢复", value=False
                )
                gr.Markdown("删除知识库", elem_classes=["kv-section"])
                delete_kb_dropdown = gr.Dropdown(label="选择知识库", choices=[])
                delete_kb_btn = gr.Button("删除所选知识库", variant="stop", size="sm")
                gr.Markdown("删除单篇文档", elem_classes=["kv-section"])
                delete_doc_dropdown = gr.Dropdown(label="选择文档", choices=[])
                delete_doc_btn = gr.Button("删除选中文档", variant="stop", size="sm")

            with gr.Accordion("配置自检 / 知识库概览", open=False, elem_classes=["kv-accordion"]):
                gr.Markdown("知识库概览", elem_classes=["kv-section"])
                kb_overview_md = gr.Markdown()
                gr.Markdown("运行环境", elem_classes=["kv-section"])
                config_md = gr.Markdown()

        # ------------------------------------------------------------------
        # 中间：主内容区（对话流 + 固定在底部的输入区）
        # ------------------------------------------------------------------
        with gr.Column(elem_classes=["kv-main"]):
            gr.Markdown(
                "# smardock · 文档智能问答工具\n"
                "本地解析归档 · 隐私模式下云端只见占位符，保护隐私 · "
                "[GitHub 仓库](https://github.com/bggcs111/smardock)",
                elem_classes=["kv-title"],
            )
            scope_dropdown = gr.Dropdown(
                label="问答范围",
                choices=[("全部文档（按左侧勾选的知识库检索）", SCOPE_ALL)],
                value=SCOPE_ALL,
            )
            chatbot = gr.Chatbot(
                label="对话",
                height=430,
                layout="bubble",
                buttons=["copy"],
                sanitize_html=False,
                render_markdown=True,
                elem_classes=["kv-chat"],
                placeholder="例如：这份合同的付款方式和违约金是怎么约定的？",
            )
            with gr.Column(elem_classes=["kv-composer"]):
                with gr.Row():
                    question_box = gr.Textbox(
                        show_label=False,
                        placeholder="输入问题，回车发送",
                        lines=1,
                        scale=6,
                        container=False,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)
                    # 回答期间显示，用于中断当前回答（纯前端取消，见下方事件绑定）
                    stop_btn = gr.Button("停止", variant="stop", scale=1, visible=False)
                with gr.Row():
                    reuse_checkbox = gr.Checkbox(
                        label="相同问题复用历史答案",
                        value=True,
                        scale=3,
                        info="索引变更时自动失效",
                    )
                    new_chat_btn = gr.Button("＋ 新建对话", variant="secondary", size="sm", scale=1)
                gr.Markdown(
                    "问答口径跟随知识库模式：隐私库入库前脱敏、提问也脱敏、回答保留占位符；"
                    "普通库原文原样。",
                    elem_classes=["kv-hint"],
                )
            # 本次检索：简要信息，放在对话页底部
            summary_md = gr.Markdown(SUMMARY_EMPTY, elem_classes=["kv-status", "kv-summary"])

        # ------------------------------------------------------------------
        # 右侧：可折叠、可拖拽改宽的引用来源面板
        # ------------------------------------------------------------------
        with gr.Sidebar(
            open=True,
            width=360,
            position="right",
            label="会话与引用",
            elem_classes=["kv-sidebar", "kv-sidebar-right"],
        ):
            # 会话：按对话分类，点击任一会话即可切换并查看该对话
            with gr.Column(elem_classes=["kv-panel"]):
                gr.Markdown("会话", elem_classes=["kv-section"])
                session_md = gr.Markdown(SESSION_EMPTY, elem_classes=["kv-hint"])
                session_radio = gr.Radio(
                    show_label=False,
                    choices=[],
                    value=None,
                    elem_classes=["kv-history-list"],
                    info="点击任一会话即可切换并查看它的历史",
                )
                with gr.Row():
                    delete_session_btn = gr.Button(
                        "删除当前会话", variant="stop", size="sm", scale=1
                    )
                    history_clear_btn = gr.Button(
                        "清空全部历史", variant="stop", size="sm", scale=1
                    )

            with gr.Column(elem_classes=["kv-panel"]):
                gr.Markdown("引用来源", elem_classes=["kv-section"])
                references_md = gr.Markdown(REFS_EMPTY, elem_classes=["kv-refs"])
                downloads = gr.Files(
                    label="原件（点击下载）", interactive=False, visible=False
                )

            # 非可视组件：当前会话 + 滚动定位触发器
            session_state = gr.State("")
            scroll_box = gr.Textbox(value="", visible=False, elem_id="kv-scroll-trigger")

        # ----------------------------------------------------------------------
        # 事件绑定：列表型输出统一按 CANON_KEYS 顺序构造，
        # 新增组件只需改 CANON_KEYS，不会出现输出错位。
        # ----------------------------------------------------------------------
        component_map = {
            "kb_checkbox": kb_checkbox,
            "upload_kb": upload_kb,
            "scope_dropdown": scope_dropdown,
            "doc_table": doc_table,
            "delete_kb_dropdown": delete_kb_dropdown,
            "delete_doc_dropdown": delete_doc_dropdown,
            "kb_overview": kb_overview_md,
            "config_md": config_md,
            "status_md": status_md,
            "kb_name_box": kb_name_box,
            "confirm_checkbox": confirm_checkbox,
            "rebuild_kb": rebuild_kb,
            "doc_status_md": doc_status_md,
            "upload_btn": upload_btn,
            "upload_stop_btn": upload_stop_btn,
            "rebuild_btn": rebuild_btn,
            "rebuild_stop_btn": rebuild_stop_btn,
            "chatbot": chatbot,
            "question_box": question_box,
            "send_btn": send_btn,
            "stop_btn": stop_btn,
            "session_state": session_state,
            "session_radio": session_radio,
            "session_md": session_md,
            "references_md": references_md,
            "downloads": downloads,
            "summary_md": summary_md,
            "scroll_box": scroll_box,
        }
        missing = set(CANON_KEYS) - set(component_map)
        if missing:
            raise RuntimeError(f"UI 缺少 CANON_KEYS 中声明的组件：{sorted(missing)}")
        canon_outputs = [component_map[key] for key in CANON_KEYS]
        session_outputs = [component_map[key] for key in SESSION_KEYS]
        query_outputs = [component_map[key] for key in QUERY_KEYS]
        for key in CANON_KEYS + SESSION_KEYS + QUERY_KEYS:
            if component_map.get(key) is None:
                raise RuntimeError(f"清单中的组件 `{key}` 尚未创建")

        # 启动前校验每个绑定的入参数量（防止参数/输出错位）
        for _name, _fn, _inputs, _outputs in (
            ("refresh_ui", app.refresh_ui, None, canon_outputs),
            ("handle_kb_change", app.handle_kb_change, [kb_checkbox, scope_dropdown], [scope_dropdown]),
            (
                "handle_create_kb",
                app.handle_create_kb,
                [kb_name_box, kb_mode_box, kb_checkbox, scope_dropdown],
                canon_outputs,
            ),
            (
                "handle_delete_kb",
                app.handle_delete_kb,
                [delete_kb_dropdown, kb_checkbox, confirm_checkbox, scope_dropdown],
                canon_outputs,
            ),
            (
                "handle_upload",
                app.handle_upload,
                [file_input, upload_kb, kb_checkbox, scope_dropdown],
                canon_outputs,
            ),
            (
                "handle_delete_document",
                app.handle_delete_document,
                [delete_doc_dropdown, confirm_checkbox, kb_checkbox, scope_dropdown],
                canon_outputs,
            ),
            (
                "handle_query",
                app.handle_query,
                [
                    question_box,
                    kb_checkbox,
                    chatbot,
                    scope_dropdown,
                    reuse_checkbox,
                    session_state,
                ],
                query_outputs,
            ),
            ("finish_query", app.finish_query, [chatbot], query_outputs),
            ("stop_query", app.stop_query, None, query_outputs),
            ("stop_indexing", app.stop_indexing, None, canon_outputs),
            (
                "handle_rebuild",
                app.handle_rebuild,
                [rebuild_kb, kb_checkbox, scope_dropdown],
                canon_outputs,
            ),
            (
                "handle_history_clear",
                app.handle_history_clear,
                [kb_checkbox, scope_dropdown],
                canon_outputs + session_outputs,
            ),
            ("restore_history", app.restore_history, None, session_outputs),
            ("handle_new_session", app.handle_new_session, None, session_outputs),
            ("handle_history_jump", app.handle_history_jump, [session_radio], session_outputs),
            ("handle_delete_session", app.handle_delete_session, [session_state], session_outputs),
        ):
            _assert_binding(_name, _fn, _inputs, _outputs)

        demo.load(fn=app.refresh_ui, inputs=None, outputs=canon_outputs)
        # 问答记录默认存档：刷新页面后自动回到最近一次会话
        demo.load(fn=app.restore_history, inputs=None, outputs=session_outputs)

        # 问答范围跟随知识库勾选联动：未勾选知识库的文档不会出现在范围列表里
        kb_checkbox.change(
            fn=app.handle_kb_change,
            inputs=[kb_checkbox, scope_dropdown],
            outputs=[scope_dropdown],
        )

        create_kb_btn.click(
            fn=app.handle_create_kb,
            inputs=[kb_name_box, kb_mode_box, kb_checkbox, scope_dropdown],
            outputs=canon_outputs,
        )
        delete_kb_btn.click(
            fn=app.handle_delete_kb,
            inputs=[delete_kb_dropdown, kb_checkbox, confirm_checkbox, scope_dropdown],
            outputs=canon_outputs,
        )
        upload_event = upload_btn.click(
            fn=app.handle_upload,
            inputs=[file_input, upload_kb, kb_checkbox, scope_dropdown],
            outputs=canon_outputs,
            trigger_mode="once",
        )
        delete_doc_btn.click(
            fn=app.handle_delete_document,
            inputs=[delete_doc_dropdown, confirm_checkbox, kb_checkbox, scope_dropdown],
            outputs=canon_outputs,
        )
        rebuild_event = rebuild_btn.click(
            fn=app.handle_rebuild,
            inputs=[rebuild_kb, kb_checkbox, scope_dropdown],
            outputs=canon_outputs,
            trigger_mode="once",
        )

        # 索引任务的「停止」：cancels 让前端立刻掐断正在跑的事件（不再刷新进度），
        # 同时后端置取消信号让循环在下一个检查点退出，并复位按钮/写一句收尾状态。
        upload_stop_btn.click(
            fn=app.stop_indexing, inputs=None, outputs=canon_outputs,
            cancels=[upload_event],
        )
        rebuild_stop_btn.click(
            fn=app.stop_indexing, inputs=None, outputs=canon_outputs,
            cancels=[rebuild_event],
        )

        history_clear_btn.click(
            fn=app.handle_history_clear,
            inputs=[kb_checkbox, scope_dropdown],
            outputs=canon_outputs + session_outputs,
        )

        # 会话：新建（对话页按钮）/ 删除 / 点击切换，事件链末尾挂滚动脚本
        new_chat_btn.click(
            fn=app.handle_new_session, inputs=None, outputs=session_outputs,
            show_progress="hidden",
        ).then(fn=None, inputs=[scroll_box], js=SCROLL_JS)
        delete_session_btn.click(
            fn=app.handle_delete_session, inputs=[session_state], outputs=session_outputs,
            show_progress="hidden",
        ).then(fn=None, inputs=[scroll_box], js=SCROLL_JS)
        session_radio.change(
            fn=app.handle_history_jump, inputs=[session_radio], outputs=session_outputs,
            show_progress="hidden",
        ).then(fn=None, inputs=[scroll_box], js=SCROLL_JS)

        query_inputs = [
            question_box,
            kb_checkbox,
            chatbot,
            scope_dropdown,
            reuse_checkbox,
            session_state,
        ]
        # show_progress="hidden"：不渲染 Gradio 自带的进度条，
        # 生成进度由对话区的「正在检索…」与逐字输出体现。
        # trigger_mode="once"：本次回答结束前，同一触发方式（按钮 / 回车）不再受理，
        # 避免连点「发送」排出一串回答；回答期间输入区还会被锁定（见 handle_query）。
        send_event = send_btn.click(
            fn=app.handle_query, inputs=query_inputs, outputs=query_outputs,
            show_progress="hidden", trigger_mode="once",
        )
        submit_event = question_box.submit(
            fn=app.handle_query, inputs=query_inputs, outputs=query_outputs,
            show_progress="hidden", trigger_mode="once",
        )

        # 「停止」：cancels 掐断当前回答；同时挂一个后端处理器，先把输入区解冻
        # （前端取消要等 Gradio 关掉生成器，检索期间没有 yield，会锁住好几秒）。
        stop_btn.click(
            fn=app.stop_query, inputs=None, outputs=query_outputs,
            cancels=[send_event, submit_event],
        )

        # 收尾：正常结束与被「停止」取消都会走到，负责把输入区解锁、
        # 隐藏「停止」。被取消时收尾输出没机会发回前端，必须靠它兜底。
        for query_event in (send_event, submit_event):
            query_event.then(
                fn=app.finish_query, inputs=[chatbot], outputs=query_outputs,
                queue=False,
            )

    return demo


def _cleanup() -> None:
    """退出服务时清理临时缓存文件（导出副本 / Gradio 上传中转 / 字节码缓存）。"""
    get_logger().info("服务退出 本次运行API调用统计=%s", call_totals())
    cleaned = config.cleanup_temp_files()
    if cleaned:
        print(f"[清理] 已删除临时缓存：{'、'.join(cleaned)}")
    else:
        print("[清理] 没有需要清理的临时缓存。")


def main() -> None:
    config.ensure_dirs()
    log = setup_logging()
    log.info(
        "服务启动 问答模型=%s@%s 向量化=%s 数据目录=%s",
        config.LLM_MODEL, config.LLM_BASE_URL, config.EMBEDDING_MODEL, config.DATA_DIR,
    )
    app = KnowledgeQAApp()
    demo = build_ui(app)

    # 正常退出（含 Ctrl+C）都会走 atexit，这里再兜一层 finally
    atexit.register(app.shutdown)
    atexit.register(_cleanup)

    # 信号处理：Ctrl+C 立即退出，不再等 Gradio 优雅关停所有 SSE 连接
    # （那是 Ctrl+C 响应慢的根因——Gradio 会等所有进行中的请求完成）
    def _quick_exit(signum: int, _frame: object) -> None:
        sig_name = signal.Signals(signum).name
        print(f"\n[{sig_name}] 正在退出…")
        try:
            app.shutdown()
            _cleanup()
        except Exception:
            pass
        os._exit(0)

    signal.signal(signal.SIGINT, _quick_exit)
    signal.signal(signal.SIGTERM, _quick_exit)

    try:
        demo.queue().launch(
            server_name=config.SERVER_NAME,
            server_port=config.SERVER_PORT,
            share=config.SHARE,
            inbrowser=config.INBROWSER,
            allowed_paths=[str(config.DATA_DIR)],
            show_error=True,
            # Gradio 6 起 theme 与 css 必须传给 launch()，不能传给 Blocks()
            theme=build_theme(),
            css=APP_CSS,
            # 侧边栏拖拽改宽（Gradio 的 Sidebar 本身不支持调宽）
            head=RESIZE_HEAD,
        )
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
