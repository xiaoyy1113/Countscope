"""Batch WSI export with traceable per-slide coordinate layouts."""
from pathlib import Path
import argparse,hashlib,json,math
import numpy as np
from PIL import Image
import openslide

def export(source,base,size,minimum):
    suffix=hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:8]
    target=base/(source.stem+'_'+suffix)
    if target.exists():raise ValueError(f'输出已存在，未覆盖：{target}')
    png=target/'png';png.mkdir(parents=True)
    kept=0;count=0
    with openslide.OpenSlide(str(source)) as slide:
        width,height=slide.dimensions
        slide.get_thumbnail((1600,1600)).save(target/'overview.jpg')
        total=math.ceil(width/size)*math.ceil(height/size)
        Image.new('RGB',(size,size),'white').save(target/'background_white.png')
        with (target/'tiles.jsonl').open('w',encoding='utf-8') as ti,(target/'layout.jsonl').open('w',encoding='utf-8') as la:
            for y in range(0,height,size):
                for x in range(0,width,size):
                    vw,vh=min(size,width-x),min(size,height-y)
                    region=slide.read_region((x,y),0,(vw,vh)).convert('RGB')
                    a=np.asarray(region).astype(np.int16);low=a.min(2);spread=a.max(2)-low
                    fraction=float(np.mean(((spread>5)&(low<245))|(low<210)))
                    omit=fraction<minimum
                    name=f'png/x{x:06d}_y{y:06d}.png'
                    row=dict(file=name,x=x,y=y,width=size,height=size,valid_width=vw,valid_height=vh,
                             level=0,foreground_fraction=fraction,omitted=omit,
                             render_file='background_white.png' if omit else name)
                    if not omit:
                        canvas=Image.new('RGB',(size,size),'white');canvas.paste(region,(0,0))
                        canvas.save(target/name,compress_level=1);ti.write(json.dumps(row)+'\n');kept+=1
                    la.write(json.dumps(row)+'\n');count+=1
                    if count%500==0:print(source.name,count,'/',total,'保留',kept,flush=True)
        info=dict(source=str(source.resolve()),dimensions=[width,height],patch_size=size,level=0,
                  mpp_x=slide.properties.get('openslide.mpp-x'),original_grid_count=count,count=kept,
                  min_foreground=minimum,foreground_method='((maxRGB-minRGB>5)&(minRGB<245))|(minRGB<210)',
                  completed=True)
    (target/'metadata.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print('完成:',target,'保留',kept,'张',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--size',type=int,choices=[256,512],default=512);p.add_argument('--min-foreground',type=float,default=1/3)
    a=p.parse_args()
    if not 0<=a.min_foreground<=1:p.error('前景比例必须在0到1之间')
    if not a.input.is_dir():p.error('源文件夹不存在')
    if a.input.resolve()==a.output.resolve():p.error('输出目录必须与源目录不同')
    slides=sorted(x for x in a.input.iterdir() if x.is_file() and x.suffix.lower() in ('.svs','.tif','.tiff'))
    if not slides:p.error('当前层没有SVS/TIF文件')
    a.output.mkdir(parents=True,exist_ok=True);errors=[]
    for slide in slides:
        try:export(slide,a.output,a.size,a.min_foreground)
        except Exception as exc:errors.append(dict(file=str(slide),error=str(exc)));print('失败:',slide.name,str(exc),flush=True)
    if errors:
        (a.output/'export_errors.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2),encoding='utf-8')
        raise SystemExit(1)
