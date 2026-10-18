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


def render_raw_masks(rgb, masks):
    """Layer all original masks for display; never resolve their pixel ownership."""
    rng = np.random.default_rng(42)
    palette = rng.integers(45, 235, (len(masks) + 1, 3), dtype=np.uint8)
    overlay = rgb.astype(float).copy()
    colors = np.zeros_like(rgb)
    boundary = np.zeros(rgb.shape[:2], bool)
    counts = masks.sum(axis=0, dtype=np.uint32)
    # Draw large masks first so nested small masks remain visible. Data stay untouched.
    for i in np.argsort(-masks.sum(axis=(1, 2), dtype=np.int64), kind='stable'):
        mask = masks[i]
        overlay[mask] = 0.65 * overlay[mask] + 0.35 * palette[i + 1]
        colors[mask] = palette[i + 1]
        padded = np.pad(mask, 1, constant_values=False)
        interior = (padded[:-2, 1:-1] & padded[2:, 1:-1] &
                    padded[1:-1, :-2] & padded[1:-1, 2:])
        boundary |= mask & ~interior
    edges = rgb.copy()
    edges[boundary] = (255, 220, 20)
    overlay[boundary] = (30, 30, 30)
    coverage = rgb.copy()
    for selected, color in ((counts == 0, [255, 50, 160]), (counts > 1, [255, 150, 0])):
        coverage[selected] = np.rint(0.4 * rgb[selected] + 0.6 * np.array(color)).astype(np.uint8)
    return {'boundaries': Image.fromarray(edges),
            'overlay': Image.fromarray(np.rint(overlay).astype(np.uint8)),
            'colors': Image.fromarray(colors), 'coverage': Image.fromarray(coverage)}


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
            views[method] = {**result, 'raw_output': metadata.get('raw_output'),
                'regions': metadata['regions'], 'id_image':
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
<label>显示 <select id="display"><option value="overlay">半透明彩色</option><option value="boundaries">边界叠加</option><option value="colors">纯色 Mask</option><option value="coverage">原始覆盖情况</option></select></label>
<label><input type="checkbox" id="full">原尺寸</label><button id="clear">清除选中</button>
<p>每行依次为原图、处理前、处理后。点击任意图像，在各方法的处理后列查看并下载该像素所属的 Mask。</p>
</header><main><p>模型处理前：自动生成器返回的全部候选（已包含其内部筛选），允许重叠和遗漏。处理后：本项目消除重叠、补齐遗漏并拆分连通域。SLIC 处理前为其直接输出。</p>
<p>原始覆盖情况：粉色为模型未覆盖像素，橙色为多个候选重叠；处理后列的粉色只标记补齐区域的来源。彩色叠加存在遮挡，完整原始候选保存在 raw_masks.npz 中。</p>
<p id="error"></p><div class="scroll"><div id="grid"></div></div>
<p><a href="overview.jpg">总览</a> · <a href="summary.json">配置与统计</a></p>
<p>最终分区的二值 Mask 均已重新读取验证：每像素恰好属于一个 Mask，每个 Mask 为 4 连通区域。原始候选按生成器输出保留。全覆盖不代表语义分割准确率。</p></main>
<script>
const DATA=__DATA__,byId=id=>document.getElementById(id);
let active=[],point=null,token=0;
DATA.images.forEach((im,i)=>byId('image').add(new Option(im.name,i)));
function loadImage(src){return new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=()=>reject(new Error('图像加载失败'));im.src=src;});}
async function update(){
 const current=++token;active=[];point=null;byId('error').textContent='';
 const im=DATA.images[Number(byId('image').value)],grid=byId('grid');grid.replaceChildren();
 grid.style.setProperty('--columns',3);grid.style.setProperty('--image-width',im.width+'px');
 for(const method of Object.keys(DATA.methods)){
 for(const stage of ['original','raw','final']){
  const result=im.results[method],article=document.createElement('article'),title=document.createElement('h2');
  title.textContent=DATA.methods[method]+' · '+({original:'原图',raw:'处理前',final:'处理后'}[stage]);article.append(title);
  const wrap=document.createElement('div');wrap.className='picture';const image=document.createElement('img');
  image.src=stage==='original'?im.original:result.directory+'/'+(stage==='raw'?'raw_':'')+byId('display').value+'.png';image.alt=title.textContent;
  const canvas=document.createElement('canvas');canvas.width=im.width;canvas.height=im.height;wrap.append(image,canvas);article.append(wrap);
  const stats=document.createElement('p');stats.className='stats';
  if(stage==='final'){stats.textContent='K='+result.region_count+' · 中位面积 '+result.median_area+' px';
   const ok=document.createElement('div');ok.className='ok';ok.textContent='覆盖 100% · 重叠 0 · 连通检查通过';stats.append(ok);
   if(method!=='slic'){const raw=document.createElement('div');raw.textContent='模型原始覆盖 '+(result.raw_coverage_fraction*100).toFixed(2)+'%';stats.append(raw);}}
  if(stage==='raw'&&result.raw_output){const raw=result.raw_output;stats.textContent='K='+raw.mask_count+' · 覆盖 '+(raw.coverage_fraction*100).toFixed(2)+'% · 重叠 '+(raw.overlap_fraction*100).toFixed(2)+'%';
   const link=document.createElement('a');link.href=result.directory+'/'+raw.file;link.textContent='下载原始分割数据';link.download=raw.file;article.append(link);}
  if(stage==='final'){const link=document.createElement('a');link.href=result.directory+'/comparison.jpg';link.textContent='打开前后对照图';article.append(link);}
  article.append(stats);const selection=document.createElement('p');selection.className='selection';selection.textContent='点击图像选择像素';article.append(selection);
  const download=document.createElement('button');download.textContent='下载选中 Mask';download.disabled=true;if(stage==='final')article.append(download);
  grid.append(article);const entry={method,stage,result,canvas,selection,download,ids:null};active.push(entry);
  wrap.onclick=e=>{const rect=wrap.getBoundingClientRect();point=[Math.max(0,Math.min(im.width-1,Math.floor((e.clientX-rect.left)*im.width/rect.width))),Math.max(0,Math.min(im.height-1,Math.floor((e.clientY-rect.top)*im.height/rect.height)))];highlight();};
  download.onclick=()=>{if(!point||!entry.ids)return;const id=entry.ids[point[1]*im.width+point[0]];
   const c=document.createElement('canvas');c.width=im.width;c.height=im.height;const ctx=c.getContext('2d'),pixels=ctx.createImageData(im.width,im.height);
   for(let i=0;i<entry.ids.length;i++){const v=entry.ids[i]===id?255:0;pixels.data.set([v,v,v,255],i*4);}ctx.putImageData(pixels,0,0);
   const link=document.createElement('a');link.download=im.key+'_'+method+'_mask_'+id+'.png';link.href=c.toDataURL('image/png');link.click();};
  if(stage==='final'){const labelImage=await loadImage(result.id_image);if(current!==token)return;
   const c=document.createElement('canvas');c.width=im.width;c.height=im.height;const ctx=c.getContext('2d',{willReadFrequently:true});ctx.drawImage(labelImage,0,0);
   const pixels=ctx.getImageData(0,0,im.width,im.height).data;entry.ids=new Uint32Array(im.width*im.height);
   for(let i=0;i<entry.ids.length;i++)entry.ids[i]=pixels[4*i]+(pixels[4*i+1]<<8)+(pixels[4*i+2]<<16);}
 }
 }
}
function highlight(){
 const im=DATA.images[Number(byId('image').value)];
 for(const entry of active){const ctx=entry.canvas.getContext('2d');ctx.clearRect(0,0,im.width,im.height);entry.download.disabled=true;
  if(!point){entry.selection.textContent='点击图像选择像素';continue;}
  if(!entry.ids){entry.selection.textContent='像素 ('+point.join(', ')+')'+(entry.stage==='raw'?' · 对应的最终分区见处理后列':'');
   ctx.strokeStyle='#ff255d';ctx.lineWidth=2;ctx.strokeRect(point[0]-4,point[1]-4,8,8);continue;}
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
