. (Join-Path $PSScriptRoot 'common.ps1')
if (-not (Test-Path -LiteralPath $pythonExe)) { throw '请先安装环境。' }
$sourceFolder = Read-Host '请输入存放 SVS/TIF 的完整文件夹路径'
$outputFolder = Read-Host '请输入输出 PNG 的完整文件夹路径（不要与源目录相同）'
$size = Read-Host '图块边长 256 或 512，直接回车为 512'
if (-not $size) { $size = '512' }
$fraction = Read-Host '最低前景比例 0到1，回车为0.333333；输入0保留全部图块'
if (-not $fraction) { $fraction = '0.3333333333333333' }
Invoke-PythonChecked -Arguments @((Join-Path $packageRoot 'export_slides.py'),'--input',$sourceFolder,'--output',$outputFolder,'--size',$size,'--min-foreground',$fraction)
