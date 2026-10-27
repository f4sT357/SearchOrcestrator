# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification for the Windows GUI distribution."""

from pathlib import Path
from PyInstaller.utils.hooks import collect_all

project_root = Path(SPECPATH).parent.parent
runtime_packages = ("sentence_transformers", "transformers", "torch")

binaries = []
datas = [(str(project_root / "config.json"), ".")]
hiddenimports = []
for package in runtime_packages:
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

analysis = Analysis(
    [str(project_root / "gui" / "run.py")],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="SearchOrchestrator",
    console=False,
)
coll = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    name="SearchOrchestrator",
)