$ErrorActionPreference = 'Stop'
$packageRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $packageRoot '.venv\Scripts\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
function Invoke-PythonChecked {
    param([string[]]$Arguments)
    & $pythonExe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python 命令失败，退出码 $LASTEXITCODE。请查看上方错误信息。" }
}
