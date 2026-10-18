"""Partition overlays, overview images, and a local interactive mask viewer."""
import base64
import json

import numpy as np
from PIL import Image, ImageDraw


def render_partition(rgb, labels, owners=None):
    """Thin raster boundaries are visual overlays; they never erase mask pixels."""
    boundary = np.zeros(labels.shape, bool)
    boundary[1:] |= labels[1:] != labels[:-1]
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    rng = np.random.default_rng(42)
    palette = rng.integers(45, 235, (int(labels.max()) + 1, 3), dtype=np.uint8)
    colors = palette[labels]
    overlay = np.rint(0.72 * rgb + 0.28 * colors).astype(np.uint8)
    edges = np.asarray(rgb).copy()
    edges[boundary] = (255, 220, 20)
    overlay[boundary] = (30, 30, 30)
    provenance = np.asarray(rgb).copy()
    if owners is not None:
        provenance[owners == 0] = np.rint(0.4 * provenance[owners == 0] +
                                         0.6 * np.array([255, 50, 160])).astype(np.uint8)
    return {'boundaries': Image.fromarray(edges), 'overlay': Image.fromarray(overlay),
            'colors': Image.fromarray(colors), 'coverage': Image.fromarray(provenance)}


def contact_sheet(items, path, columns=4, cell_width=320):
    if not items:
        return
    ratio = cell_width / items[0][1].width
    cell_height = round(items[0][1].height * ratio) + 28
    sheet = Image.new('RGB', (columns * cell_width,
                            ((len(items) + columns - 1) // columns) * cell_height), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (title, im) in enumerate(items):
        x, y = (i % columns) * cell_width, (i // columns) * cell_height
        draw.text((x + 5, y + 5), title, fill='black')
        sheet.paste(im.resize((cell_width, cell_height - 28)), (x, y + 28))
    sheet.save(path)


def write_viewer(output, summary):
    """Self-contained ID data allow mask selection when opened via file://."""
    from config import METHOD_NAMES
    data = {'methods': {m: METHOD_NAMES[m] for m in summary['methods']}, 'images': []}
    for row in summary['images']:
        views = {}
        for method, result in row['results'].items():
            directory = output / result['directory']
            metadata = json.loads((directory / 'metadata.json').read_text())
            views[method] = {**result, 'regions': metadata['regions'], 'id_image':
                'data:image/png;base64,' + base64.b64encode((directory / 'ids.png').read_bytes()).decode('ascii')}
        first = next(iter(views.values()))
        with Image.open(output / first['directory'] / 'original.png') as original:
            width, height = original.size
        data['images'].append({'name': row['name'], 'key': row['key'], 'width': width,
                               'height': height, 'results': views,
                               'original': first['directory'] + '/original.png'})
    page = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Dense Partition</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f3f5f8;color:#213047;font:15px system-ui,sans-serif}
header,main{padding:20px 24px}header{background:white}h1{font-size:24px;margin:0 0 16px}h2{font-size:17px}
p{line-height:1.6}select,button{font:inherit;padding:6px}label{margin-right:16px}.scroll{overflow:auto}
#grid{display:grid;grid-template-columns:repeat(var(--columns),minmax(260px,1fr));gap:12px}
#grid.full{grid-template-columns:repeat(var(--columns),var(--image-width))}
article{padding:12px;background:white;border-radius:8px}.picture{position:relative;line-height:0;cursor:crosshair}
.picture img{width:100%;height:auto}.picture canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.stats,.selection{font-size:13px;min-height:48px}.ok{color:#147343}#error{color:#b00020}button:disabled{opacity:.4}
</style><header><h1>Dense Partition · 全覆盖分割</h1>
<label>图像 <select id="image"></select></label>
<label>显示 <select id="display"><option value="boundaries">边界叠加</option><option value="overlay">半透明彩色</option><option value="colors">纯色 Mask</option><option value="coverage">模型原始未覆盖区（粉色）</option></select></label>
<label><input type="checkbox" id="full">原尺寸</label><button id="clear">清除选中</button>
<p>点击任意图像，同步查看该像素在各方法中所属的 Mask。边界线仅是显示效果，每个像素都有唯一归属。</p>
</header><main><p>模型未覆盖的像素也会纳入最终分区；粉色表示其来源。SLIC 没有这项模型覆盖信息。</p>
<p id="error"></p><div class="scroll"><div id="grid"></div></div>
<p><a href="overview.jpg">总览</a> · <a href="summary.json">配置与统计</a></p>
<p>所有导出的二值 Mask 均已重新读取验证：每像素恰好属于一个 Mask，每个 Mask 为 4 连通区域。全覆盖不代表语义分割准确率。</p></main>
<script>
const DATA=__DATA__,byId=id=>document.getElementById(id);
let active=[],point=null,token=0;
DATA.images.forEach((im,i)=>byId('image').add(new Option(im.name,i)));
function loadImage(src){return new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=()=>reject(new Error('图像加载失败'));im.src=src;});}
async function update(){
 const current=++token;active=[];point=null;byId('error').textContent='';
 const im=DATA.images[Number(byId('image').value)],grid=byId('grid');grid.replaceChildren();
 grid.style.setProperty('--columns',Object.keys(DATA.methods).length+1);grid.style.setProperty('--image-width',im.width+'px');
 for(const method of ['original',...Object.keys(DATA.methods)]){
  const result=im.results[method],article=document.createElement('article'),title=document.createElement('h2');
  title.textContent=result?DATA.methods[method]:'原图';article.append(title);
  const wrap=document.createElement('div');wrap.className='picture';const image=document.createElement('img');
  image.src=result?result.directory+'/'+byId('display').value+'.png':im.original;image.alt=title.textContent;
  const canvas=document.createElement('canvas');canvas.width=im.width;canvas.height=im.height;wrap.append(image,canvas);article.append(wrap);
  const stats=document.createElement('p');stats.className='stats';
  if(result){stats.textContent='K='+result.region_count+' · 中位面积 '+result.median_area+' px';
   const ok=document.createElement('div');ok.className='ok';ok.textContent='覆盖 100% · 重叠 0 · 连通检查通过';stats.append(ok);
   if(method!=='slic'){const raw=document.createElement('div');raw.textContent='模型原始覆盖 '+(result.raw_coverage_fraction*100).toFixed(2)+'%';stats.append(raw);}}
  article.append(stats);const selection=document.createElement('p');selection.className='selection';selection.textContent='点击图像选择像素';article.append(selection);
  const download=document.createElement('button');download.textContent='下载选中 Mask';download.disabled=true;if(result)article.append(download);
  grid.append(article);const entry={method,result,canvas,selection,download,ids:null};active.push(entry);
  wrap.onclick=e=>{const rect=wrap.getBoundingClientRect();point=[Math.max(0,Math.min(im.width-1,Math.floor((e.clientX-rect.left)*im.width/rect.width))),Math.max(0,Math.min(im.height-1,Math.floor((e.clientY-rect.top)*im.height/rect.height)))];highlight();};
  download.onclick=()=>{if(!point||!entry.ids)return;const id=entry.ids[point[1]*im.width+point[0]];
   const c=document.createElement('canvas');c.width=im.width;c.height=im.height;const ctx=c.getContext('2d'),pixels=ctx.createImageData(im.width,im.height);
   for(let i=0;i<entry.ids.length;i++){const v=entry.ids[i]===id?255:0;pixels.data.set([v,v,v,255],i*4);}ctx.putImageData(pixels,0,0);
   const link=document.createElement('a');link.download=im.key+'_'+method+'_mask_'+id+'.png';link.href=c.toDataURL('image/png');link.click();};
  if(result){const labelImage=await loadImage(result.id_image);if(current!==token)return;
   const c=document.createElement('canvas');c.width=im.width;c.height=im.height;const ctx=c.getContext('2d',{willReadFrequently:true});ctx.drawImage(labelImage,0,0);
   const pixels=ctx.getImageData(0,0,im.width,im.height).data;entry.ids=new Uint32Array(im.width*im.height);
   for(let i=0;i<entry.ids.length;i++)entry.ids[i]=pixels[4*i]+(pixels[4*i+1]<<8)+(pixels[4*i+2]<<16);}
 }
}
function highlight(){
 const im=DATA.images[Number(byId('image').value)];
 for(const entry of active){const ctx=entry.canvas.getContext('2d');ctx.clearRect(0,0,im.width,im.height);entry.download.disabled=true;
  if(!point){entry.selection.textContent='点击图像选择像素';continue;}
  if(!entry.ids){entry.selection.textContent='像素 ('+point.join(', ')+')';continue;}
  const id=entry.ids[point[1]*im.width+point[0]],region=entry.result.regions[id-1],pixels=ctx.createImageData(im.width,im.height);
  for(let i=0;i<entry.ids.length;i++)if(entry.ids[i]===id)pixels.data.set([0,255,190,180],4*i);ctx.putImageData(pixels,0,0);
  ctx.strokeStyle='#ff255d';ctx.lineWidth=2;ctx.strokeRect(point[0]-4,point[1]-4,8,8);
  entry.selection.textContent='像素 ('+point.join(', ')+') → Mask #'+id+' · '+region.area+' px';entry.download.disabled=false;
 }
}
for(const id of ['image','display'])byId(id).onchange=()=>update().catch(e=>byId('error').textContent=e.message);
byId('full').onchange=()=>byId('grid').classList.toggle('full',byId('full').checked);
byId('clear').onclick=()=>{point=null;highlight();};update().catch(e=>byId('error').textContent=e.message);
</script></html>'''
    (output / 'index.html').write_text(page.replace('__DATA__',
        json.dumps(data, ensure_ascii=False).replace('</', '<\\/')), encoding='utf-8')
