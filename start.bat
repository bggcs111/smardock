@echo off
rem 文档智能问答与归档系统 —— Windows 启动脚本
setlocal
cd /d "%~dp0"

rem 1) 准备虚拟环境（首次运行自动创建，避免污染全局 Python 环境）
if not exist ".venv" (
  echo [1/3] 创建虚拟环境 .venv
  python -m venv .venv
)

call ".venv\Scripts\activate.bat"

rem 2) 安装依赖（只在首次执行；如需重装请删除 .venv\.deps-ok）
if not exist ".venv\.deps-ok" (
  echo [2/3] 安装依赖，首次运行需要几分钟，请稍候...
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  if errorlevel 1 (
    echo 依赖安装失败，请检查网络连接或 Python 版本。
    exit /b 1
  )
  echo ok> ".venv\.deps-ok"
) else (
  echo [2/3] 依赖已就绪，跳过安装
)

if not exist ".env" (
  if exist ".env.example" (
    echo 提示：未找到 .env，已从 .env.example 复制一份，请填入 API Key 后重新启动。
    copy ".env.example" ".env" >nul
  )
)

echo [3/3] 启动服务
python app.py
endlocal
