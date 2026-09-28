. (Join-Path $PSScriptRoot 'common.ps1')
$record = Join-Path $packageRoot 'logs\server.json'
if (-not (Test-Path -LiteralPath $record)) { Write-Host '没有找到本工具的启动记录。'; exit 0 }
$entry = Get-Content -LiteralPath $record -Raw -Encoding UTF8 | ConvertFrom-Json
$proc = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + [int]$entry.pid)
if ($proc -and $proc.ExecutablePath -eq $pythonExe -and $proc.CommandLine.Contains((Join-Path $packageRoot 'app.py'))) {
    Stop-Process -Id $entry.pid
    Write-Host '工具已停止，图片和缓存保留。'
} elseif ($proc) { throw '进程与本工具不匹配，未执行停止。' } else { Write-Host '工具已经停止。' }
