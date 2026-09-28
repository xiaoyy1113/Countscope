. (Join-Path $PSScriptRoot 'common.ps1')
if (-not (Test-Path -LiteralPath $pythonExe)) { throw '尚未安装独立环境，请先执行安装入口。' }
Invoke-PythonChecked -Arguments @((Join-Path $packageRoot 'check_environment.py'),'--inference')
