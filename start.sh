#!/usr/bin/env bash
# 文档智能问答与归档系统 —— 启动脚本（macOS / Linux）
set -e

cd "$(dirname "$0")"

# 1) 准备虚拟环境（首次运行自动创建，避免污染全局 Python 环境）
if [ ! -d ".venv" ]; then
  echo "[1/3] 创建虚拟环境 .venv"
  python -m venv .venv
fi

# shellcheck disable=SC1091
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
else
  source .venv/Scripts/activate
fi

# 2) 安装依赖（只在首次执行；如需重装请删除 .venv/.deps-ok）
if [ ! -f ".venv/.deps-ok" ]; then
  echo "[2/3] 安装依赖，首次运行需要几分钟，请稍候..."
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  touch ".venv/.deps-ok"
else
  echo "[2/3] 依赖已就绪，跳过安装"
fi

if [ ! -f ".env" ] && [ -f ".env.example" ]; then
  echo "提示：未找到 .env，已从 .env.example 复制一份，请填入 API Key 后重新启动。"
  cp .env.example .env
fi

# 3) 启动服务
echo "[3/3] 启动服务"
python app.py
