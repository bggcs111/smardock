"""交互层主题：克制的工具型极简风格（Notion / Linear / Raycast 气质）。

设计令牌（CSS 变量）一览
--------------------------------------------------------------------------
分类      变量名                 取值                     用途
--------------------------------------------------------------------------
背景      --bg-page              #FAFAF9                 页面背景（暖白，禁纯白）
          --bg-panel             #FFFFFF                 面板／侧边栏背景
          --bg-subtle            #F7F7F5                 次级底（状态区、表头）
          --bg-code              #F6F7F8                 代码块／表格底色
文字      --text-primary         #1F2937                 正文（禁纯黑）
          --text-secondary       #6B7280                 次要文字／区块标题
          --text-tertiary        #9CA3AF                 占位符与空状态
主色      --accent               #4F6BED                 关键按钮、选中态、链接（面积极小）
          --accent-hover         #3B5BDB                 悬停态
          --accent-soft          #EEF1FD                 选中浅底／引用上标底
语义色    --success              #10B981                 成功（柔和，不大面积铺色）
          --warning              #F59E0B                 警告
          --danger               #EF4444                 危险操作
          --danger-soft          #FEF2F2                 危险操作悬停底
分割线    --line                 #E5E7EB                 极浅，仅用于必要分隔
圆角      --radius               10px                    全局统一，不混用
间距      --gap-1 … --gap-6      4 / 8 / 12 / 16 / 24 / 32 px
动效      --dur                  160ms                   150–200ms 区间内
          --ease                 cubic-bezier(.4,0,.2,1) 自然缓动，无弹跳缩放
阅读宽度  --reading              68ch                    正文最大宽度，避免长行疲劳
--------------------------------------------------------------------------

实现说明：
    * 主题对象（``gr.themes.Soft`` + ``.set()``）负责把变量灌进 Gradio 内置组件；
      自定义 CSS 负责布局、分区与问答细节。两者共用同一套取值。
    * Gradio 6 起 ``theme`` 与 ``css`` 必须传给 ``launch()``，不能传给 ``Blocks()``。
    * 未使用 ``gr.themes.GoogleFont``（需要访问 fonts.googleapis.com，
      离线或网络受限时会拖慢首屏）；改用含 Inter 的系统字体栈，
      如需联网加载 Inter，把 ``FONT_STACK`` 换成
      ``[gr.themes.GoogleFont("Inter"), *FALLBACK]`` 即可。
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# 令牌取值（与上方表格一一对应；改配色只需要动这里）
# ---------------------------------------------------------------------------
BG_PAGE = "#FAFAF9"
BG_PANEL = "#FFFFFF"
BG_SUBTLE = "#F7F7F5"
BG_CODE = "#F6F7F8"

TEXT_PRIMARY = "#1F2937"
TEXT_SECONDARY = "#6B7280"
TEXT_TERTIARY = "#9CA3AF"

ACCENT = "#4F6BED"
ACCENT_HOVER = "#3B5BDB"
ACCENT_SOFT = "#EEF1FD"

SUCCESS = "#10B981"
WARNING = "#F59E0B"
DANGER = "#EF4444"
DANGER_SOFT = "#FEF2F2"

LINE = "#E5E7EB"
RADIUS = "10px"

DURATION = "160ms"
EASING = "cubic-bezier(.4, 0, .2, 1)"
READING_WIDTH = "68ch"

# 拉丁字形走 Inter / system-ui，中文回退到系统黑体
FONT_STACK = [
    "Inter",
    "PingFang SC",
    "Microsoft YaHei",
    "Source Han Sans SC",
    "Noto Sans SC",
    "system-ui",
    "sans-serif",
]

APP_CSS = f"""
/* ==========================================================================
   0. 设计令牌 + 覆盖 Gradio 主题变量（让内置组件也跟随本风格）
   ========================================================================== */
.gradio-container {{
  --bg-page: {BG_PAGE};
  --bg-panel: {BG_PANEL};
  --bg-subtle: {BG_SUBTLE};
  --bg-code: {BG_CODE};
  --text-primary: {TEXT_PRIMARY};
  --text-secondary: {TEXT_SECONDARY};
  --text-tertiary: {TEXT_TERTIARY};
  --accent: {ACCENT};
  --accent-hover: {ACCENT_HOVER};
  --accent-soft: {ACCENT_SOFT};
  --success: {SUCCESS};
  --warning: {WARNING};
  --danger: {DANGER};
  --danger-soft: {DANGER_SOFT};
  --line: {LINE};
  --radius: {RADIUS};
  --gap-1: 4px;  --gap-2: 8px;  --gap-3: 12px;
  --gap-4: 16px; --gap-5: 24px; --gap-6: 32px;
  --dur: {DURATION};
  --ease: {EASING};
  --reading: {READING_WIDTH};

  /* 内置组件跟随本风格：去阴影、去渐变、统一圆角 */
  --body-background-fill: {BG_PAGE};
  --body-text-color: {TEXT_PRIMARY};
  --body-text-color-subdued: {TEXT_SECONDARY};
  --body-text-size: 15px;
  --background-fill-primary: {BG_PANEL};
  --background-fill-secondary: {BG_SUBTLE};
  --block-background-fill: transparent;
  --block-border-width: 0;
  --block-shadow: none;
  --block-radius: {RADIUS};
  --panel-background-fill: {BG_PANEL};
  --panel-border-color: {LINE};
  --panel-border-width: 0;
  --input-background-fill: #FFFFFF;
  --input-border-color: {LINE};
  --input-radius: {RADIUS};
  --color-accent: {ACCENT};
  --color-accent-soft: {ACCENT_SOFT};
  --link-text-color: {ACCENT_HOVER};
  --button-primary-background-fill: {ACCENT};
  --button-primary-background-fill-hover: {ACCENT_HOVER};
  --button-primary-text-color: #FFFFFF;
  --button-primary-border-color: {ACCENT};
  --button-secondary-background-fill: #FFFFFF;
  --button-secondary-background-fill-hover: {BG_SUBTLE};
  --button-secondary-text-color: {TEXT_PRIMARY};
  --button-secondary-border-color: {LINE};
  --button-large-radius: {RADIUS};
  --button-small-radius: {RADIUS};
  --border-color-primary: {LINE};
  --block-label-text-color: {TEXT_SECONDARY};
  --block-title-text-color: {TEXT_PRIMARY};
  --code-background-fill: {BG_CODE};
  --table-border-color: {LINE};
  --table-even-background-fill: {BG_SUBTLE};
  --shadow-drop: none;
  --shadow-drop-lg: none;

  background: {BG_PAGE} !important;
  color: {TEXT_PRIMARY};
  line-height: 1.7;
}}

/* 排版：15px 正文 / 1.7 行高 / 充足段距 */
.gradio-container p, .gradio-container li {{ line-height: 1.7; margin-bottom: var(--gap-2); }}
.gradio-container code, .gradio-container pre {{
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}}

/* 兜底：清掉 Gradio 默认的阴影与渐变，保持界面安静 */
.gradio-container .block, .gradio-container .form, .gradio-container .panel,
.gradio-container .gr-group, .gradio-container .gr-box {{
  box-shadow: none !important;
  background-image: none !important;
}}

/* ==========================================================================
   1. 面板：靠留白 + 极浅 1px 分割线分组，禁止重边框与阴影堆叠
   ========================================================================== */
.kv-panel {{
  background: var(--bg-panel);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: var(--gap-4);
  margin-bottom: var(--gap-4);
}}
.kv-panel:last-child {{ margin-bottom: 0; }}

/* 区块标题：靠字重与颜色深浅区分层级，不用色块和粗边框 */
.kv-section {{ margin-bottom: var(--gap-2) !important; }}
.kv-section h1, .kv-section h2, .kv-section h3, .kv-section p {{
  font-size: 13px !important;
  font-weight: 600 !important;
  color: {TEXT_SECONDARY} !important;
  letter-spacing: .02em;
  margin: 0 !important;
}}

/* 状态区：比页面底略暗一档，与输入控件区分开。
   索引日志可能很长，限高后内部滚动，避免把侧边栏撑乱或与其它内容叠在一起。 */
.kv-status {{
  background: var(--bg-subtle) !important;
  border-radius: var(--radius) !important;
  padding: var(--gap-3) var(--gap-4) !important;
  font-size: 14px;
  color: {TEXT_PRIMARY} !important;
  min-height: 20px;
  max-height: 280px;
  overflow-y: auto;
  overscroll-behavior: contain;
  word-break: break-word;
}}
.kv-status p {{ margin: 0 0 var(--gap-1) 0 !important; }}

/* 对话页底部的检索信息栏：只放一行简要信息，弱化存在感 */
.kv-summary {{
  max-height: none;
  margin-top: var(--gap-2);
  padding: var(--gap-2) var(--gap-3) !important;
  font-size: 12.5px;
  color: {TEXT_SECONDARY} !important;
  background: var(--bg-subtle) !important;
}}
.kv-summary p {{ margin: 0 !important; }}

/* 空状态：一句话，无插图 */
.kv-empty, .kv-empty p {{ color: {TEXT_TERTIARY} !important; font-size: 14px; }}

/* 辅助说明：比次要文字再弱一档，只用于解释性提示 */
.kv-hint, .kv-hint p {{
  color: {TEXT_TERTIARY} !important;
  font-size: 12.5px;
  line-height: 1.65;
  margin: 0 0 var(--gap-2) 0 !important;
}}

/* ==========================================================================
   2. 主内容区与输入区
   ========================================================================== */
.kv-main {{ padding: 0 var(--gap-2); }}

.kv-title h1 {{ font-size: 17px !important; font-weight: 600 !important; color: {TEXT_PRIMARY} !important; margin: 0 0 var(--gap-1) 0 !important; }}
.kv-title p {{ font-size: 13px !important; color: {TEXT_SECONDARY} !important; margin: 0 !important; }}

/* 输入区固定在底部，不随内容滚动 */
.kv-composer {{
  position: sticky;
  bottom: 0;
  z-index: 5;
  background: {BG_PAGE};
  border-top: 1px solid var(--line);
  padding-top: var(--gap-3);
  margin-top: var(--gap-3);
}}

/* 问答消息：对齐 + 背景微差区分，去掉气泡边框与阴影 */
.kv-chat .message {{
  border: none !important;
  box-shadow: none !important;
  border-radius: var(--radius) !important;
  max-width: var(--reading);
}}
.kv-chat .message.user {{ background: var(--accent-soft) !important; }}
.kv-chat .message.bot {{ background: transparent !important; }}

/* 引用上标：正文里的极简标记，hover 显示来源（原生 title 提示） */
.kv-cite {{
  font-size: 11px;
  line-height: 1;
  vertical-align: super;
  color: {ACCENT};
  background: var(--accent-soft);
  border-radius: 4px;
  padding: 1px 4px;
  margin: 0 2px 0 1px;
  cursor: default;
  transition: background var(--dur) var(--ease);
}}
.kv-cite:hover {{ background: #DDE4FB; }}

/* 代码块与表格：低饱和底色 + 等宽字体 */
.kv-chat pre {{
  background: var(--bg-code) !important;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: var(--gap-3);
  font-size: 13px;
  overflow-x: auto;
}}
.kv-chat code {{ background: var(--bg-code) !important; font-size: 13px; }}
.kv-chat table {{ border-collapse: collapse; font-size: 14px; }}
.kv-chat th, .kv-chat td {{ border: 1px solid var(--line); padding: 6px 10px; }}
.kv-chat thead th {{ background: var(--bg-subtle); font-weight: 600; color: {TEXT_SECONDARY}; }}

/* ==========================================================================
   3. 引用来源面板
   ========================================================================== */
/* 引用面板：条目可能很多，限高后内部滚动，保证每一条都能看到 */
.kv-refs {{
  font-size: 14px;
  color: {TEXT_PRIMARY} !important;
  max-height: 58vh;
  overflow-y: auto;
  overscroll-behavior: contain;
  padding-right: 4px;
}}
.kv-refs blockquote {{
  border-left: 2px solid var(--line);
  margin: var(--gap-2) 0;
  padding-left: var(--gap-3);
  color: {TEXT_SECONDARY};
}}
.kv-refs code {{
  background: var(--bg-code) !important;
  border: none;
  font-size: 12px;
  color: {TEXT_SECONDARY};
  word-break: break-all;
}}

/* ==========================================================================
   3b. 问答历史：条目之间用极浅分割线，信息密度高于引用区
   ========================================================================== */
.kv-history {{ font-size: 13.5px; color: {TEXT_PRIMARY} !important; }}
.kv-history p {{ margin: 0 0 var(--gap-1) 0 !important; }}
.kv-history hr {{
  border: none;
  border-top: 1px solid var(--line);
  margin: var(--gap-3) 0;
}}

/* 会话历史列表：做成可点击的紧凑卡片，而不是一排难点的单选圆钮 */
.kv-history-list {{ max-height: 280px; overflow-y: auto; }}
.kv-history-list label {{
  border: 1px solid var(--line);
  border-radius: var(--radius);
  background: #fff;
  padding: 6px 8px !important;
  margin: 0 !important;
  font-size: 12.5px;
  line-height: 1.5;
  transition: background var(--dur) var(--ease), border-color var(--dur) var(--ease);
}}
.kv-history-list label:hover {{ background: var(--bg-subtle); }}
.kv-history-list label:has(input:checked) {{
  border-color: var(--accent);
  background: var(--accent);
  color: #ffffff !important;
  font-weight: 600;
}}
.kv-history-list input[type="radio"] {{ display: none; }}

/* ==========================================================================
   3c. 生成过程中的进度只出现在对话区
       抑制 Gradio 给侧边栏各面板打的「进行中」遮罩 / 闪烁 / 进度条。
       注意：只重置透明度与动画，绝不能用 display:none 隐藏 .pending 元素本身，
       否则整个面板会消失。
   ========================================================================== */
.kv-sidebar .pending,
.kv-sidebar .generating {{
  opacity: 1 !important;
  animation: none !important;
  filter: none !important;
}}
.kv-sidebar .pending::before,
.kv-sidebar .pending::after,
.kv-sidebar .generating::before,
.kv-sidebar .generating::after {{
  display: none !important;
  content: none !important;
  animation: none !important;
}}
.kv-sidebar .progress-text,
.kv-sidebar .progress-bar,
.kv-sidebar .eta-bar,
.kv-chat .progress-text,
.kv-chat .eta-bar {{ display: none !important; }}

/* ==========================================================================
   4. 折叠区：统一为「1px 分割线 + 留白」的轻量容器
   ========================================================================== */
.kv-accordion, .kv-danger {{
  border: 1px solid var(--line) !important;
  border-radius: var(--radius) !important;
  background: {BG_PANEL} !important;
  padding: var(--gap-2) var(--gap-3) !important;
  margin-bottom: var(--gap-4) !important;
}}
.kv-danger .kv-section h3, .kv-danger .kv-section p {{ color: {DANGER} !important; }}
.kv-danger button.stop {{
  background: #FFFFFF !important;
  color: {DANGER} !important;
  border: 1px solid {DANGER} !important;
}}
.kv-danger button.stop:hover {{ background: {DANGER_SOFT} !important; }}

/* ==========================================================================
   5. 问答知识库选择器：位置固定不动，列表过长时限高并在框内滚动
   ========================================================================== */
.kv-kb-fixed {{
  max-height: 320px;
  overflow-y: auto;
  overscroll-behavior: contain;
}}
/* 兼容 Gradio 不同版本内部容器命名，命中其一即可获得内层滚动 */
.kv-kb-fixed .wrap,
.kv-kb-fixed [role="group"],
.kv-kb-fixed .options {{
  max-height: 176px;
  overflow-y: auto;
  overscroll-behavior: contain;
}}

/* ==========================================================================
   6. 侧边栏可拖拽改宽：把手贴在栏与主区之间的边界线上
   ========================================================================== */
.kv-resizer {{
  position: absolute;
  top: 0;
  bottom: 0;
  width: 8px;              /* 细，不抢视线 */
  z-index: 30;
  cursor: col-resize;
  background: transparent;
  touch-action: none;
  transition: background var(--dur) var(--ease);
}}
.kv-resizer-left {{ right: 0; }}
.kv-resizer-right {{ left: 0; }}
/* hover / 拖拽时只显示一条极浅提示，符合克制风格 */
.kv-resizer:hover, .kv-resizer.kv-dragging {{ background: var(--accent-soft); }}

/* ==========================================================================
   7. 交互细节：focus 清晰、动效 160ms 自然缓动、细进度条
   ========================================================================== */
.gradio-container *:focus-visible {{
  outline: 2px solid var(--accent);
  outline-offset: 1px;
}}
.gradio-container button, .gradio-container input, .gradio-container textarea,
.gradio-container .message, .gradio-container .kv-cite {{
  transition: background-color var(--dur) var(--ease),
              border-color var(--dur) var(--ease),
              color var(--dur) var(--ease);
}}
/* 加载状态用细进度条，避免旋转大动画 */
.gradio-container .progress-bar {{ height: 2px !important; }}
.gradio-container progress {{ height: 2px; }}

/* ==========================================================================
   8. 移动端：自动堆叠为单栏
   ========================================================================== */
@media (max-width: 768px) {{
  .kv-panel {{ padding: var(--gap-3); }}
  .kv-composer {{ position: static; }}
  .kv-chat .message {{ max-width: 100%; }}
  .kv-main {{ padding: 0; }}
  .kv-resizer {{ display: none; }}   /* 单栏下不需要拖拽改宽 */
}}
"""

#: 侧边栏拖拽改宽脚本。
#:
#: Gradio 的 Sidebar 不支持拖拽调宽，这里在左右侧边栏的边界线上挂一个把手，
#: 用 pointer 事件实时改宽度。为兼容 Gradio 的异步渲染，采用轮询挂载 + 幂等守卫；
#: 找不到元素时静默退出，不影响任何既有功能。
RESIZE_HEAD = """
<script>
(() => {
  const MIN_WIDTH = 240;
  const MAX_WIDTH = 560;

  function attach(selector, side) {
    const el = document.querySelector(selector);
    if (!el || el.querySelector(":scope > .kv-resizer")) return false;

    // 侧边栏若为静态定位，则补一个定位上下文，保证把手贴在栏边界上
    if (getComputedStyle(el).position === "static") {
      el.style.position = "relative";
    }

    const handle = document.createElement("div");
    handle.className = "kv-resizer kv-resizer-" + side;
    handle.title = "拖动可调整宽度";
    el.appendChild(handle);

    let startX = 0;
    let startWidth = 0;

    const onMove = (event) => {
      const delta = event.clientX - startX;
      const next = side === "left" ? startWidth + delta : startWidth - delta;
      el.style.width = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, next)) + "px";
      event.preventDefault();
    };

    const onUp = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      handle.classList.remove("kv-dragging");
      document.body.style.userSelect = "";
    };

    handle.addEventListener("pointerdown", (event) => {
      startX = event.clientX;
      startWidth = el.getBoundingClientRect().width;
      handle.classList.add("kv-dragging");
      document.body.style.userSelect = "none";
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
      event.preventDefault();
    });

    return true;
  }

  let tries = 0;
  const timer = setInterval(() => {
    tries += 1;
    const left = attach(".kv-sidebar-left", "left");
    const right = attach(".kv-sidebar-right", "right");
    if ((left && right) || tries > 25) clearInterval(timer);
  }, 400);
})();
</script>
"""

#: 历史定位用的前端脚本：优先滚动到带 data-kv-turn 锚点的那一轮，否则滚到底部。
#: 用法：事件链末尾接 ``.then(fn=None, inputs=[scroll_box], js=SCROLL_JS)``。
SCROLL_JS = """
(_value) => {
  const host = document.querySelector('.kv-chat');
  if (!host) return;

  // Gradio 的滚动容器层级随版本变化，这里递归找一个真正能滚的元素
  const findScroller = (node) => {
    if (!node || !node.children) return null;
    const style = window.getComputedStyle(node);
    if (/(auto|scroll)/.test(style.overflowY) && node.scrollHeight > node.clientHeight + 4) {
      return node;
    }
    for (const child of node.children) {
      const found = findScroller(child);
      if (found) return found;
    }
    return null;
  };

  const box = findScroller(host) || host;
  const anchor = host.querySelector('[data-kv-turn]');
  if (anchor) {
    const delta = anchor.getBoundingClientRect().top - box.getBoundingClientRect().top;
    box.scrollTop = box.scrollTop + delta - 16;
  } else {
    box.scrollTop = box.scrollHeight;
  }
}
"""


def build_theme() -> Any:
    """构建 Gradio 主题：以 Soft 为基底，覆盖为暖白 + 低饱和靛的克制配色。"""
    import gradio as gr

    try:
        return gr.themes.Soft(
            primary_hue="indigo",
            neutral_hue="gray",
            font=FONT_STACK,
        ).set(
            body_background_fill=BG_PAGE,
            body_text_color=TEXT_PRIMARY,
            body_text_color_subdued=TEXT_SECONDARY,
            body_text_size="15px",
            background_fill_primary=BG_PANEL,
            background_fill_secondary=BG_SUBTLE,
            block_background_fill="transparent",
            block_border_width="0px",
            block_shadow="none",
            block_radius=RADIUS,
            block_label_text_color=TEXT_SECONDARY,
            block_title_text_color=TEXT_PRIMARY,
            panel_background_fill=BG_PANEL,
            input_background_fill="#FFFFFF",
            input_border_color=LINE,
            border_color_primary=LINE,
            color_accent=ACCENT,
            link_text_color=ACCENT_HOVER,
            button_primary_background_fill=ACCENT,
            button_primary_background_fill_hover=ACCENT_HOVER,
            button_primary_text_color="#FFFFFF",
            button_secondary_background_fill="#FFFFFF",
            button_secondary_text_color=TEXT_PRIMARY,
            button_secondary_border_color=LINE,
            code_background_fill=BG_CODE,
            table_border_color=LINE,
            table_even_background_fill=BG_SUBTLE,
            shadow_drop="none",
            shadow_drop_lg="none",
        )
    except Exception:  # pragma: no cover - 主题参数不被支持时退回默认
        try:
            return gr.themes.Soft()
        except Exception:
            return gr.themes.Default()
