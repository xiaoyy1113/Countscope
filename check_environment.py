from pathlib import Path
import argparse,hashlib,json,platform,struct,sys,time
root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--inference',action='store_true');args=parser.parse_args()
print('系统:',platform.platform(),'Python:',platform.python_version(),'位数:',struct.calcsize('P')*8)
import torch,torchvision,flask,numpy,PIL
print('PyTorch:',torch.__version__,'CUDA编译版本:',torch.version.cuda)
print('CUDA可用:',torch.cuda.is_available())
manifest=json.loads((root/'模型信息.json').read_text(encoding='utf-8'))
p=root/manifest['file'];actual=hashlib.sha256(p.read_bytes()).hexdigest()
if actual!=manifest['sha256']:raise RuntimeError('模型权重校验失败，请重新复制交付包。')
print('模型权重 SHA256: 校验通过')
import app
print('实际运行设备:',app.DEVICE,app.DEVICE_NOTE)
if app.DEVICE=='cuda':
    print('显卡:',torch.cuda.get_device_name(0),'显存GB:',round(torch.cuda.get_device_properties(0).total_memory/1024**3,1))
if args.inference:
    # Exercise the included architecture + weights, without medical sample data.
    model=app.load_model();t=time.time()
    with torch.inference_mode():
        x=torch.zeros((1,3,266,266),device=app.DEVICE)
        f=model.forward_features(x)['x_norm_patchtokens']
        assert tuple(f.shape)==(1,361,384) and bool(torch.isfinite(f).all())
    print('整图推理检查通过:',tuple(f.shape),'用时秒:',round(time.time()-t,2))
print('检查结束。GPU不可用时仍可使用CPU；CPU建议先测试少量图块。')
