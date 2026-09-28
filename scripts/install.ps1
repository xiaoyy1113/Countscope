param([ValidateSet('cpu','gpu')][string]$Mode='cpu')
. (Join-Path $PSScriptRoot 'common.ps1')
Set-Location -LiteralPath $packageRoot
Write-Host 'SCOPE 环境安装：首次安装需要联网，仅安装到本文件夹。'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    $candidate = $null
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($version in @('-3.11','-3.12')) {
            try { $probe = & py $version -c "import sys,struct; print(sys.executable if struct.calcsize('P')==8 else '')" 2>$null } catch { continue }
            if ($LASTEXITCODE -eq 0 -and $probe) { $candidate = [string]($probe | Select-Object -Last 1); break }
        }
    }
    if (-not $candidate -and (Get-Command python -ErrorAction SilentlyContinue)) {
        $probe = & python -c "import sys,struct; print(sys.executable if sys.version_info[:2] in [(3,11),(3,12)] and struct.calcsize('P')==8 else '')" 2>$null
        if ($LASTEXITCODE -eq 0 -and $probe) { $candidate = [string]($probe | Select-Object -Last 1) }
    }
    if (-not $candidate) { throw '请先安装 Python 3.11 或 3.12（64位），安装时勾选 Add python.exe to PATH。详见使用说明。' }
    & $candidate -m venv (Join-Path $packageRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw '创建独立环境失败。' }
}
Invoke-PythonChecked -Arguments @('-m','pip','install','--upgrade','pip')
if ($Mode -eq 'gpu') { $index = 'https://download.pytorch.org/whl/cu128' } else { $index = 'https://download.pytorch.org/whl/cpu' }
Invoke-PythonChecked -Arguments @('-m','pip','install','--upgrade','--force-reinstall','torch==2.10.0','torchvision==0.25.0','--index-url',$index)
Invoke-PythonChecked -Arguments @('-m','pip','install','-r',(Join-Path $packageRoot 'requirements.txt'))
Invoke-PythonChecked -Arguments @((Join-Path $packageRoot 'check_environment.py'))
Write-Host '环境安装完成。请双击“02_启动工具.cmd”。'
