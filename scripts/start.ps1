param([switch]$CpuOnly, [switch]$NoBrowser)
. (Join-Path $PSScriptRoot 'common.ps1')
if (-not (Test-Path -LiteralPath $pythonExe)) { throw '尚未安装环境，请先运行 01_安装环境_CPU.cmd 或 01_安装环境_NVIDIA.cmd。' }
$config = Get-Content -LiteralPath (Join-Path $packageRoot 'config.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$url = 'http://127.0.0.1:' + $config.port
$state = $null
try { $state = Invoke-RestMethod -Uri ($url + '/api/state') -TimeoutSec 2 } catch { }
if ($state) {
    if ($state.package_id -ne 'scope-local-distribution-1.0' -or $state.installation -ne $packageRoot) { throw '端口被其他程序占用，请修改 config.json 的 port 后重试。' }
    if ($CpuOnly -and $state.device -ne 'cpu') { throw '当前已有 GPU 服务运行。请先运行“03_停止工具.cmd”，再启动 CPU 模式。' }
    if (-not $NoBrowser) { Start-Process $url }
    exit 0
}
$logs = Join-Path $packageRoot 'logs'
New-Item -ItemType Directory -Path $logs -Force | Out-Null
if ($CpuOnly) { $env:SCOPE_DEVICE='cpu' } else { Remove-Item Env:SCOPE_DEVICE -ErrorAction SilentlyContinue }
$child = Start-Process -FilePath $pythonExe -ArgumentList ('"' + (Join-Path $packageRoot 'app.py') + '"') -WorkingDirectory $packageRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logs 'server.out.log') -RedirectStandardError (Join-Path $logs 'server.err.log') -PassThru
@{pid=$child.Id;root=$packageRoot;port=$config.port} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logs 'server.json') -Encoding UTF8
for ($attempt=0; $attempt -lt 60; $attempt++) {
    Start-Sleep -Seconds 1
    try {
        $state = Invoke-RestMethod -Uri ($url+'/api/state') -TimeoutSec 1
        if ($state.package_id -eq 'scope-local-distribution-1.0' -and $state.installation -eq $packageRoot) { if (-not $NoBrowser) { Start-Process $url }; exit 0 }
    } catch { }
    if ($child.HasExited) { break }
}
throw '启动失败，请查看 logs\server.err.log 或运行环境检查。'
