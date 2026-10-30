@echo off
chcp 65001 >nul
cd /d %~dp0
start "联邦学习系统 v2.0" ".venv\Scripts\pythonw.exe" federated_gui_pyside.py
