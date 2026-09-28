"""Local whole-image DINOv2 spatial-feature retrieval. No image uploads."""
from __future__ import annotations
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import secrets
import sqlite3
import threading
import time
import traceback
from datetime import datetime, timezone
from contextlib import closing

ROOT = Path(__file__).resolve().parent
os.environ['TORCH_HOME'] = str(ROOT / 'models')
from color_filter import summarize_rgb
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_file

CONFIG = json.loads((ROOT / 'config.json').read_text(encoding='utf-8-sig'))
PORT = int(os.environ.get('SCOPE_PORT', CONFIG.get('port',8775)))
torch.set_num_threads(int(CONFIG.get('cpu_threads',4)))
DEFAULT = Path(CONFIG['default_folder']).expanduser() if CONFIG.get('default_folder') else None
CACHE = ROOT / 'cache'
CACHE.mkdir(exist_ok=True)
ANNOTATIONS = ROOT / 'annotations.sqlite3'
TOKEN = secrets.token_urlsafe(24)
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16384
lock = threading.RLock()
cancel = threading.Event()
dataset = {'folder': '', 'files': [], 'generation': ''}
job = {'status': 'idle', 'phase': '请选择图片并点击一个位置', 'done': 0,
       'total': 0, 'results': [], 'errors': []}
model = None
requested_device = os.environ.get('SCOPE_DEVICE', CONFIG.get('device','auto'))
DEVICE = 'cpu'
DEVICE_NOTE = ''
if requested_device != 'cpu' and torch.cuda.is_available():
    try:
        probe = torch.ones((32,32),device='cuda')
        probe = probe @ probe
        torch.cuda.synchronize()
        DEVICE = 'cuda'
        del probe
    except Exception as exc:
        DEVICE_NOTE = '显卡测试未通过，已使用 CPU。请运行环境检查。'
elif requested_device == 'cuda':
    DEVICE_NOTE = '当前环境无法使用 CUDA，已使用 CPU。'
PACKAGE_ID = 'scope-local-distribution-1.0' 
MODEL_ID = 'dinov2_vits14_whole_v1'


def annotation_key(folder, name, x, y, width, height):
    raw=f'{folder}|{name}|{x}|{y}|{width}|{height}'
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]


def annotation_connection():
    connection=sqlite3.connect(ANNOTATIONS,timeout=30)
    connection.row_factory=sqlite3.Row
    connection.execute('''CREATE TABLE IF NOT EXISTS annotations (
        candidate_key TEXT PRIMARY KEY,
        folder TEXT NOT NULL,
        image_name TEXT NOT NULL,
        x INTEGER NOT NULL,
        y INTEGER NOT NULL,
        width INTEGER NOT NULL,
        height INTEGER NOT NULL,
        center_x REAL NOT NULL,
        center_y REAL NOT NULL,
        global_x INTEGER,
        global_y INTEGER,
        global_center_x REAL,
        global_center_y REAL,
        score REAL,
        review_label TEXT NOT NULL,
        query_json TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )''')
    connection.execute('''CREATE TABLE IF NOT EXISTS manual_points (
        point_key TEXT PRIMARY KEY,
        folder TEXT NOT NULL,
        image_name TEXT NOT NULL,
        x INTEGER NOT NULL,
        y INTEGER NOT NULL,
        width INTEGER NOT NULL,
        height INTEGER NOT NULL,
        center_x REAL NOT NULL,
        center_y REAL NOT NULL,
        global_x INTEGER,
        global_y INTEGER,
        global_center_x REAL,
        global_center_y REAL,
        updated_at TEXT NOT NULL
    )''')
    connection.execute('''CREATE TABLE IF NOT EXISTS patch_reviews (
        folder TEXT NOT NULL,
        image_name TEXT NOT NULL,
        review_status TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(folder,image_name)
    )''')
    return connection


def annotations_for(folder):
    with closing(annotation_connection()) as connection:
        rows=connection.execute(
            'SELECT * FROM annotations WHERE folder=? ORDER BY image_name,y,x',(folder,)
        ).fetchall()
    return [dict(row) for row in rows]


def manual_points_for(folder):
    with closing(annotation_connection()) as connection:
        rows=connection.execute(
            'SELECT * FROM manual_points WHERE folder=? ORDER BY image_name,y,x',(folder,)
        ).fetchall()
    return [dict(row) for row in rows]


def patch_reviews_for(folder):
    with closing(annotation_connection()) as connection:
        rows=connection.execute(
            'SELECT * FROM patch_reviews WHERE folder=? ORDER BY image_name',(folder,)
        ).fetchall()
    return [dict(row) for row in rows]


def candidate_coordinates(path, x, y, width, height):
    gx=gy=None
    match=re.search(r'x(\d+)_y(\d+)',path.stem)
    if match:gx=int(match[1])+x;gy=int(match[2])+y
    return gx,gy


def write_candidate_annotation(folder, path, x, y, width, height, score, label, config):
    key=annotation_key(folder,path.name,x,y,width,height)
    gx,gy=candidate_coordinates(path,x,y,width,height)
    now=datetime.now(timezone.utc).isoformat()
    values=(key,folder,path.name,x,y,width,height,x+width/2,y+height/2,gx,gy,
            None if gx is None else gx+width/2,None if gy is None else gy+height/2,
            float(score),label,json.dumps(config,ensure_ascii=False),now)
    with closing(annotation_connection()) as connection, connection:
        connection.execute('''INSERT INTO annotations (
            candidate_key,folder,image_name,x,y,width,height,center_x,center_y,
            global_x,global_y,global_center_x,global_center_y,score,review_label,query_json,updated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(candidate_key) DO UPDATE SET
            score=excluded.score,review_label=excluded.review_label,
            query_json=excluded.query_json,updated_at=excluded.updated_at''',values)
    return key


def open_folder(folder):
    path = Path(folder).expanduser().resolve()
    if not path.is_dir():
        raise ValueError('文件夹不存在，请填写本机的完整路径。')
    files = sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() == '.png'],
                   key=lambda p: p.name.lower())
    if not files:
        raise ValueError('该文件夹中没有 PNG 图片（只读取当前层）。')
    with lock:
        if job['status'] == 'running':
            raise ValueError('请先取消正在运行的任务，再切换文件夹。')
        dataset.update(folder=str(path), files=files, generation=secrets.token_hex(8))
        job.update(status='idle', phase=f'已打开 {len(files)} 张 PNG', done=0,
                   total=0, results=[], errors=[])
    return files


def file_at(index):
    with lock:
        if not 0 <= index < len(dataset['files']):
            raise ValueError('图片编号无效，请重新打开文件夹。')
        return dataset['files'][index]


def load_model():
    global model
    if model is None:
        with lock:
            job['phase'] = '正在加载 DINOv2 模型'
        repo = ROOT / 'models/hub/facebookresearch_dinov2_main'
        weights = ROOT / 'models/hub/checkpoints/dinov2_vits14_pretrain.pth'
        if not repo.is_dir() or not weights.is_file():
            raise ValueError('本地模型缺失，请重新解压完整交付包并运行环境检查。')
        m = torch.hub.load(str(repo), 'dinov2_vits14', source='local', pretrained=False)
        m.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
        model = m.eval().to(DEVICE)
    return model


def cache_path(path, scale):
    stat = path.stat()
    key = f'{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{MODEL_ID}|{scale}'
    return CACHE / (hashlib.sha256(key.encode()).hexdigest()+'.npz')


@torch.inference_mode()
def features(path, scale):
    target = cache_path(path, scale)
    if target.is_file():
        with np.load(target, allow_pickle=False) as z:
            return z['f'], int(z['width']), int(z['height'])
    m = load_model()
    with Image.open(path) as source:
        im = source.convert('RGB')
        w, h = im.size
        if min(w,h)<2 or max(w,h)*scale>1536:
            raise ValueError(f'{path.name}: 当前支持放大后最长边不超过 1536 像素。')
        if scale != 1:
            im = im.resize((w*scale,h*scale), Image.Resampling.BICUBIC)
        a = np.array(im, dtype=np.float32)/255
    t = torch.from_numpy(a).permute(2,0,1).unsqueeze(0).to(DEVICE)
    ph, pw = (-t.shape[2])%14, (-t.shape[3])%14
    # No crop: retain the entire image. Only bottom/right replicated padding.
    t = F.pad(t,(0,pw,0,ph),mode='replicate')
    t = (t-t.new_tensor([.485,.456,.406])[None,:,None,None])/t.new_tensor([.229,.224,.225])[None,:,None,None]
    with torch.autocast(device_type=DEVICE, enabled=DEVICE=='cuda', dtype=torch.float16):
        f = m.forward_features(t)['x_norm_patchtokens']
    f = f.reshape(t.shape[2]//14,t.shape[3]//14,384).float().cpu().numpy().astype(np.float16)
    temp = target.with_suffix('.tmp')
    with temp.open('wb') as stream:
        np.savez(stream, f=f, width=w, height=h)
    temp.replace(target)
    return f,w,h


def overlap_weights(starts, length, cells, stride, device=DEVICE):
    """Fractional ROI/token overlap; no claim of pixel-resolution features."""
    starts = torch.as_tensor(starts, dtype=torch.float32,device=device).reshape(-1,1)
    edges = torch.arange(cells,device=device,dtype=torch.float32)[None,:]*stride
    return (torch.minimum(starts+length,edges+stride)-torch.maximum(starts,edges)).clamp(min=0)/length


def descriptors(f, xs, ys, width, height, stride):
    wy = overlap_weights(ys,height,f.shape[0],stride, f.device)
    wx = overlap_weights(xs,width,f.shape[1],stride, f.device)
    a = torch.einsum('yh,hwc->ywc',wy,f)
    a = torch.einsum('xw,ywc->yxc',wx,a)
    return F.normalize(a,dim=-1,eps=1e-8)


def axis_starts(limit, extent, stride):
    values = list(np.arange(0,limit-extent+1e-5,stride,dtype=np.float32))
    if not values or abs(values[-1]-(limit-extent))>1e-4:
        values.append(float(limit-extent))
    return values


def separated(points, x, y, distance):
    return all(math.hypot(x-p[1],y-p[2])>=distance for p in points)


@torch.inference_mode()
def multiple_reference_matches(array, image_w, image_h, rw, rh, scale, queries,
                               threshold=-1., max_candidates=20, nms_distance=14.):
    """Find local maxima against a bank of normalized reference descriptors.

    The score at each location is the maximum cosine similarity to any
    reference.  The returned reference index identifies which exemplar won.
    """
    f=torch.as_tensor(array.astype(np.float32),device=DEVICE)
    queries=torch.as_tensor(queries,dtype=torch.float32,device=DEVICE)
    if queries.ndim==1:queries=queries[None,:]
    if queries.ndim!=2 or not queries.shape[0]:raise ValueError('至少需要一个参考点')
    queries=F.normalize(queries,dim=-1,eps=1e-8)
    stride=14/scale
    xs=axis_starts(image_w,rw,stride);ys=axis_starts(image_h,rh,stride)
    d=descriptors(f,xs,ys,rw,rh,stride)
    reference_scores=d@queries.T
    score,best_reference=reference_scores.max(dim=-1)
    local_max=F.max_pool2d(score[None,None],kernel_size=3,stride=1,padding=1)[0,0]
    mask=(score>=threshold)&(score>=local_max-1e-7)
    indices=mask.flatten().nonzero().flatten()
    if not indices.numel():return []
    order=indices[score.flatten()[indices].argsort(descending=True)].tolist()
    coarse=[]
    for i in order:
        iy,ix=divmod(i,len(xs))
        if separated(coarse,float(xs[ix]),float(ys[iy]),nms_distance):
            coarse.append((float(score[iy,ix]),float(xs[ix]),float(ys[iy]),i))
        if len(coarse)>=max_candidates*3:break
    refined=[]
    for _,_,_,i in coarse:
        iy,ix=divmod(i,len(xs))
        xx=np.arange(max(0,math.floor(xs[ix]-stride)),min(image_w-rw,math.ceil(xs[ix]+stride))+1)
        yy=np.arange(max(0,math.floor(ys[iy]-stride)),min(image_h-rh,math.ceil(ys[iy]+stride))+1)
        values=descriptors(f,xx,yy,rw,rh,stride)@queries.T
        location_scores,location_references=values.max(dim=-1)
        flat=int(location_scores.argmax());y,x=divmod(flat,len(xx))
        s=float(location_scores[y,x].clamp(-1,1));reference=int(location_references[y,x])
        if s>=threshold:refined.append((s,int(xx[x]),int(yy[y]),reference))
    kept=[]
    for candidate in sorted(refined,reverse=True):
        if separated(kept,candidate[1],candidate[2],nms_distance):
            kept.append(candidate)
        if len(kept)>=max_candidates:break
    return kept


@torch.inference_mode()
def multiple_matches(array, image_w, image_h, rw, rh, scale, query,
                     threshold=-1., max_candidates=20, nms_distance=14.):
    """Backward-compatible single-reference search."""
    return [(score,x,y) for score,x,y,_ in multiple_reference_matches(
        array,image_w,image_h,rw,rh,scale,query,threshold,max_candidates,nms_distance)]


@torch.inference_mode()
def best_match(array, image_w, image_h, rw, rh, scale, query):
    matches=multiple_matches(array,image_w,image_h,rw,rh,scale,query,
                             threshold=-1,max_candidates=1,nms_distance=1)
    return matches[0]


def update(**kwargs):
    with lock:job.update(kwargs)


def worker(config, files):
    started=time.time()
    results=[]; errors=[]
    try:
        scale=config['scale']
        if config['mode']=='search':
            rw,rh=config['width'],config['height'];query_values=[]
            for reference in config['references']:
                source=files[reference['source']]
                sf,w,h=features(source,scale);x=reference['x'];y=reference['y']
                if not (0<=x<=w-rw and 0<=y<=h-rh):
                    raise ValueError(f'参考点超出原图：{source.name}')
                query_values.append(descriptors(
                    torch.as_tensor(sf.astype(np.float32),device=DEVICE),
                    [x],[y],rw,rh,14/scale)[0,0])
            queries=torch.stack(query_values)
            reference_sources={reference['source'] for reference in config['references']}
        total=len(files)
        for i,path in enumerate(files):
            if cancel.is_set():
                update(status='cancelled',phase='已取消；已完成的特征缓存可复用',done=i,
                       results=sorted(results,key=lambda r:r['score'],reverse=True),errors=errors)
                return
            update(phase=('提取整图特征' if config['mode']=='index' else '比较区域特征')+f'：{path.name}',done=i,total=total)
            try:
                if config['mode']=='search' and i in reference_sources and config['exclude_source']:
                    continue
                a,w,h=features(path,scale)
                if config['mode']=='search':
                    if rw>w or rh>h:
                        raise ValueError('图片小于选框，已跳过')
                    matches=multiple_reference_matches(
                        a,w,h,rw,rh,scale,queries,config['candidate_threshold'],
                        config['max_candidates'],config['nms_distance'])
                    with Image.open(path) as source_image:
                        color_image=source_image.convert('RGB')
                    for score,bx,by,reference_index in matches:
                        reference=config['references'][reference_index]
                        row=dict(id=i,name=path.name,score=score,x=bx,y=by,width=rw,height=rh,image_width=w,image_height=h)
                        row.update(matched_reference_index=reference.get('reference_number',reference_index+1),
                                   matched_reference_id=reference.get('reference_id',str(reference_index+1)),
                                   matched_reference_image=files[reference['source']].name)
                        row['candidate_key']=annotation_key(dataset['folder'],path.name,bx,by,rw,rh)
                        row['center_x']=bx+rw/2;row['center_y']=by+rh/2
                        row['color']=summarize_rgb(np.asarray(color_image.crop((bx,by,bx+rw,by+rh))))
                        m=re.search(r'x(\d+)_y(\d+)',path.stem)
                        if m:
                            row.update(global_x=int(m[1])+bx,global_y=int(m[2])+by)
                            row.update(global_center_x=row['global_x']+rw/2,global_center_y=row['global_y']+rh/2)
                        results.append(row)
            except Exception as exc:
                errors.append({'name':path.name,'error':str(exc)})
                # A model failure must not silently become an all-empty success.
                if model is None:raise
        update(status='done',phase='完成',done=total,total=total,
               results=sorted(results,key=lambda r:r['score'],reverse=True),errors=errors,
               elapsed=round(time.time()-started,1),config=config)
    except Exception as exc:
        traceback.print_exc()
        update(status='error',phase=str(exc),errors=errors,elapsed=round(time.time()-started,1))


@app.before_request
def local_guard():
    if request.host.split(':')[0] not in {'127.0.0.1','localhost'}:
        return jsonify(error='仅允许本机访问'),403
    if request.method=='POST' and request.headers.get('X-Scope-Token')!=TOKEN:
        return jsonify(error='请刷新页面后重试'),403


@app.errorhandler(Exception)
def error(exc):
    return jsonify(error=str(exc)),400


@app.get('/')
def index():
    return (ROOT/'static/index.html').read_text(encoding='utf-8').replace('__TOKEN__',TOKEN)


@app.get('/app.js')
def js():return send_file(ROOT/'static/app.js',mimetype='application/javascript')


@app.get('/manual')
def manual():return send_file(ROOT/'00_使用说明.html',mimetype='text/html')


@app.get('/style.css')
def css():return send_file(ROOT/'static/style.css',mimetype='text/css')


@app.get('/review.css')
def review_css():return send_file(ROOT/'static/review.css',mimetype='text/css')


@app.get('/api/state')
def state():
    with lock:
        return jsonify(package_id=PACKAGE_ID,installation=str(ROOT),device_note=DEVICE_NOTE,folder=dataset['folder'],generation=dataset['generation'],
                       files=[dict(id=i,name=p.name) for i,p in enumerate(dataset['files'])],
                       model='DINOv2 ViT-S/14',device=DEVICE,
                       gpu=torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU')


@app.post('/api/folder')
def folder():
    open_folder(request.json.get('folder',''))
    return state()


@app.get('/api/image/<int:index>')
def image(index):
    path=file_at(index)
    if 'crop' not in request.args and 'thumb' not in request.args:
        return send_file(path,mimetype='image/png',conditional=True)
    with Image.open(path) as source:
        im=source.convert('RGB')
    if 'crop' in request.args:
        x,y,w,h=[int(request.args[k]) for k in ('x','y','w','h')]
        if not (0<=x<im.width and 0<=y<im.height and 1<=w<=im.width-x and 1<=h<=im.height-y):
            raise ValueError('裁剪范围无效')
        # Show surrounding context plus the matched box; marker is display-only.
        context=max(w,h,64)
        left=max(0,min(im.width-context,x+w//2-context//2));top=max(0,min(im.height-context,y+h//2-context//2))
        im=im.crop((left,top,min(im.width,left+context),min(im.height,top+context)))
        draw=ImageDraw.Draw(im);draw.rectangle((x-left,y-top,x-left+w-1,y-top+h-1),outline='#00ab8e',width=1)
        im=im.resize((256,256),Image.Resampling.NEAREST)
    else:im.thumbnail((160,160))
    stream=io.BytesIO();im.save(stream,format='PNG');stream.seek(0)
    return send_file(stream,mimetype='image/png')


@app.post('/api/start')
def start():
    data=request.json or {}
    with lock:
        if job['status']=='running':raise ValueError('已有任务运行中')
        if data.get('generation')!=dataset['generation']:raise ValueError('文件夹已经变化，请刷新')
        files=list(dataset['files'])
        if not files:raise ValueError('请先打开文件夹')
        mode=data.get('mode','search')
        raw_references=data.get('references')
        if raw_references is None and mode=='search':
            raw_references=[dict(source=data.get('source',0),x=data.get('x',0),y=data.get('y',0),
                                 reference_id='legacy-1')]
        references=[]
        for number,row in enumerate(raw_references or [],1):
            if not isinstance(row,dict):raise ValueError('参考点格式无效')
            references.append(dict(source=int(row.get('source',-1)),x=int(row.get('x',-1)),
                                   y=int(row.get('y',-1)),
                                   reference_id=str(row.get('reference_id',number))[:80],
                                   reference_number=int(row.get('reference_number',number))))
        config=dict(mode=mode,scale=int(data.get('scale',1)),
                    source=references[0]['source'] if references else int(data.get('source',0)),
                    x=references[0]['x'] if references else int(data.get('x',0)),
                    y=references[0]['y'] if references else int(data.get('y',0)),
                    width=int(data.get('width',14)),height=int(data.get('height',14)),
                    references=references,
                    exclude_source=bool(data.get('exclude_source',True)),
                    candidate_threshold=float(data.get('candidate_threshold',.85)),
                    max_candidates=int(data.get('max_candidates',20)),
                    nms_distance=float(data.get('nms_distance',14)))
        if config['mode'] not in {'index','search'} or config['scale'] not in (1,2):raise ValueError('配置无效')
        if not 0<=config['source']<len(files):raise ValueError('源图片无效')
        if not 1<=config['width']<=512 or not 1<=config['height']<=512:raise ValueError('宽高需在 1–512 之间')
        if config['mode']=='search':
            if config['width']!=14 or config['height']!=14:raise ValueError('多参考检索固定使用 14×14 参考框')
            if not 1<=len(references)<=5:raise ValueError('参考库需要 1–5 个已启用参考点')
            if any(not 0<=row['source']<len(files) or row['x']<0 or row['y']<0 for row in references):
                raise ValueError('参考点坐标或来源图片无效')
            if any(not 1<=row['reference_number']<=5 for row in references):raise ValueError('参考点编号无效')
        if not -1<=config['candidate_threshold']<=1:raise ValueError('候选相似度需在 -1 到 1 之间')
        if not 1<=config['max_candidates']<=100:raise ValueError('每图候选上限需在 1–100 之间')
        if not 1<=config['nms_distance']<=512:raise ValueError('候选去重距离需在 1–512 像素之间')
        cancel.clear()
        job.clear();job.update(status='running',phase='准备中',done=0,total=len(files),results=[],errors=[],config=config)
        threading.Thread(target=worker,args=(config,files),daemon=True).start()
    return jsonify(ok=True)


@app.post('/api/cancel')
def stop():cancel.set();return jsonify(ok=True)


@app.get('/api/job')
def status():
    with lock:
        d={k:v for k,v in job.items() if k!='results'}
        d['result_count']=len(job.get('results',[]))
        return jsonify(d)


@app.get('/api/results')
def results():
    with lock:
        folder=dataset['folder'];rows=[dict(row) for row in job.get('results',[])]
        config=dict(job.get('config',{}))
    labels={row['candidate_key']:row['review_label'] for row in annotations_for(folder)}
    for row in rows:row['review_label']=labels.get(row.get('candidate_key'),'')
    return jsonify(results=rows,config=config,folder=folder)


@app.get('/api/annotations')
def get_annotations():
    with lock:folder=dataset['folder']
    return jsonify(annotations=annotations_for(folder),folder=folder)


@app.get('/api/review-state')
def review_state():
    with lock:folder=dataset['folder']
    return jsonify(annotations=annotations_for(folder),manual_points=manual_points_for(folder),
                   patch_reviews=patch_reviews_for(folder),folder=folder)


@app.post('/api/annotation')
def save_annotation():
    data=request.json or {}
    label=data.get('review_label','')
    if label not in {'positive','negative',''}:raise ValueError('人工标签无效')
    with lock:
        if data.get('generation')!=dataset['generation']:raise ValueError('文件夹已经变化，请刷新')
        folder=dataset['folder'];config=dict(job.get('config',{}))
    path=file_at(int(data.get('id',-1)))
    x=int(data['x']);y=int(data['y']);width=int(data['width']);height=int(data['height'])
    with Image.open(path) as im:
        if not (0<=x<=im.width-width and 0<=y<=im.height-height and width>0 and height>0):
            raise ValueError('候选框坐标无效')
    key=annotation_key(folder,path.name,x,y,width,height)
    if not label:
        with closing(annotation_connection()) as connection, connection:
            connection.execute('DELETE FROM annotations WHERE candidate_key=?',(key,))
        return jsonify(ok=True,candidate_key=key,review_label='')
    write_candidate_annotation(folder,path,x,y,width,height,data.get('score',0),label,config)
    return jsonify(ok=True,candidate_key=key,review_label=label)


@app.post('/api/manual-point')
def manual_point():
    data=request.json or {}
    with lock:
        if data.get('generation')!=dataset['generation']:raise ValueError('文件夹已经变化，请刷新')
        folder=dataset['folder']
    action=data.get('action','add')
    if action=='delete':
        key=str(data.get('point_key',''))
        with closing(annotation_connection()) as connection, connection:
            connection.execute('DELETE FROM manual_points WHERE point_key=? AND folder=?',(key,folder))
        return jsonify(ok=True,point_key=key,deleted=True)
    path=file_at(int(data.get('id',-1)))
    width=int(data.get('width',14));height=int(data.get('height',14))
    center_x=float(data['center_x']);center_y=float(data['center_y'])
    with Image.open(path) as im:
        if not (1<=width<=im.width and 1<=height<=im.height):raise ValueError('人工框尺寸无效')
        if not (0<=center_x<im.width and 0<=center_y<im.height):raise ValueError('人工点坐标无效')
        x=max(0,min(im.width-width,int(round(center_x-width/2))))
        y=max(0,min(im.height-height,int(round(center_y-height/2))))
    center_x=x+width/2;center_y=y+height/2
    gx,gy=candidate_coordinates(path,x,y,width,height)
    key=hashlib.sha256(f'{folder}|manual|{path.name}|{center_x}|{center_y}'.encode('utf-8')).hexdigest()[:24]
    now=datetime.now(timezone.utc).isoformat()
    values=(key,folder,path.name,x,y,width,height,center_x,center_y,gx,gy,
            None if gx is None else gx+width/2,None if gy is None else gy+height/2,now)
    with closing(annotation_connection()) as connection, connection:
        connection.execute('''INSERT INTO manual_points (
            point_key,folder,image_name,x,y,width,height,center_x,center_y,
            global_x,global_y,global_center_x,global_center_y,updated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(point_key) DO NOTHING''',values)
    return jsonify(ok=True,point=dict(zip(
        ['point_key','folder','image_name','x','y','width','height','center_x','center_y',
         'global_x','global_y','global_center_x','global_center_y','updated_at'],values)))


@app.post('/api/patch-review')
def patch_review():
    data=request.json or {}
    status=data.get('review_status','')
    if status not in {'complete','deferred',''}:raise ValueError('Patch 审核状态无效')
    index=int(data.get('id',-1));path=file_at(index)
    with lock:
        if data.get('generation')!=dataset['generation']:raise ValueError('文件夹已经变化，请刷新')
        folder=dataset['folder'];config=dict(job.get('config',{}))
        candidates=[dict(row) for row in job.get('results',[]) if row.get('id')==index]
    if status=='complete':
        existing={row['candidate_key']:row['review_label'] for row in annotations_for(folder)}
        for row in candidates:
            if existing.get(row['candidate_key'])!='negative':
                write_candidate_annotation(folder,path,row['x'],row['y'],row['width'],row['height'],
                                           row['score'],'positive',config)
    now=datetime.now(timezone.utc).isoformat()
    with closing(annotation_connection()) as connection, connection:
        if status:
            connection.execute('''INSERT INTO patch_reviews(folder,image_name,review_status,updated_at)
                VALUES(?,?,?,?) ON CONFLICT(folder,image_name) DO UPDATE SET
                review_status=excluded.review_status,updated_at=excluded.updated_at''',
                (folder,path.name,status,now))
        else:connection.execute('DELETE FROM patch_reviews WHERE folder=? AND image_name=?',(folder,path.name))
    return jsonify(ok=True,image_name=path.name,review_status=status,updated_at=now)


@app.get('/api/review-export')
def review_export():
    with lock:folder=dataset['folder']
    annotations=annotations_for(folder);manual=manual_points_for(folder);reviews=patch_reviews_for(folder)
    by_image={}
    for row in annotations:
        values=by_image.setdefault(row['image_name'],{'positive':0,'negative':0})
        if row['review_label'] in values:values[row['review_label']]+=1
    for row in manual:by_image.setdefault(row['image_name'],{'positive':0,'negative':0})['positive']+=1
    patches=[]
    for row in reviews:
        counts=by_image.get(row['image_name'],{'positive':0,'negative':0})
        patches.append({**row,'gold_count':counts['positive'],'gold_presence':int(counts['positive']>0),
                        'rejected_count':counts['negative']})
    return jsonify(folder=folder,annotations=annotations,manual_points=manual,patches=patches)


if DEFAULT is not None and DEFAULT.is_dir():open_folder(DEFAULT)
if __name__=='__main__':
    app.run(host='127.0.0.1',port=PORT,threaded=True,debug=False,use_reloader=False)
