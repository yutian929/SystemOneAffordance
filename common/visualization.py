"""Partition overlays, overview images, and a local interactive mask viewer."""
import base64
import json

import numpy as np
from PIL import Image, ImageDraw


VIEW_COLUMNS = (
    ('original', 'Original'),
    ('raw_colors', 'Raw colors'),
    ('boundaries', 'Boundaries'),
    ('colors', 'Colors'),
    ('overlay', 'Overlay'),
)


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


def contact_sheet(items, path, columns=4, cell_width=320, row_titles=None):
    if not items:
        return
    rows = [items[start:start + columns] for start in range(0, len(items), columns)]
    header_height = 28 if row_titles else 0
    heights = [max(round(im.height * cell_width / im.width) for _, im in row) for row in rows]
    sheet = Image.new('RGB', (columns * cell_width,
                            sum(heights) + len(rows) * (header_height + 28)), 'white')
    draw = ImageDraw.Draw(sheet)
    y = 0
    for row_index, row in enumerate(rows):
        if row_titles:
            draw.text((5, y + 5), row_titles[row_index], fill='black')
            y += header_height
        for column, (title, im) in enumerate(row):
            x = column * cell_width
            draw.text((x + 5, y + 5), title, fill='black')
            height = round(im.height * cell_width / im.width)
            sheet.paste(im.resize((cell_width, height)), (x, y + 28))
        y += heights[row_index] + 28
    sheet.save(path)


def write_overview(output, summary):
    """One row per image/method, using the same five views as the HTML viewer."""
    from config import METHOD_NAMES
    tiles, titles = [], []
    for row in summary['images']:
        for method in summary['methods']:
            result = row['results'][method]
            raw_count = (result.get('raw_output') or {}).get('mask_count', '?')
            titles.append(f'{row["name"]} | {METHOD_NAMES[method]} | '
                          f'Raw: {raw_count} / Final: {result["region_count"]} masks')
            for name, title in VIEW_COLUMNS:
                with Image.open(output / result['directory'] / f'{name}.png') as image:
                    tiles.append((title, image.copy()))
    contact_sheet(tiles, output / 'overview.jpg', columns=len(VIEW_COLUMNS), row_titles=titles)


def write_viewer(output, summary):
    """Self-contained ID data allow mask selection when opened via file://."""
    from config import METHOD_NAMES
    data = {'methods': {m: METHOD_NAMES[m] for m in summary['methods']},
            'columns': VIEW_COLUMNS, 'images': []}
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
                               'height': height, 'results': views})
    page = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Dense Partition</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f5f7;color:#243044;font:14px system-ui,sans-serif}
header{display:flex;align-items:center;gap:16px;flex-wrap:wrap;padding:16px 20px;background:white;border-bottom:1px solid #e1e4e8}
h1{font-size:19px;margin:0}h2{font-size:15px;margin:0}select,button{font:inherit;padding:5px 8px}
select{max-width:min(400px,80vw)}label{display:flex;align-items:center;gap:6px}a{color:#45617c}
.links{margin-left:auto;display:flex;gap:14px}main{padding:0 20px 20px}.hint{color:#6b7280;font-size:13px}
.scroll{overflow:auto}#grid{min-width:960px}#grid.full{min-width:0;width:calc(5 * var(--image-width) + 58px)}
.method{padding:12px;background:white;border:1px solid #e1e4e8;border-radius:6px;margin-bottom:12px}
.method-header{display:flex;align-items:center;gap:16px;min-height:28px;margin-bottom:8px}
.counts,.selection{font-size:12px;color:#6b7280}.selection{margin-left:auto}
.views{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px}
figure{margin:0;min-width:0}figcaption{font-size:12px;color:#6b7280;margin-bottom:6px}
.picture{position:relative;line-height:0;cursor:crosshair}.picture img{display:block;width:100%;height:auto}
.picture canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
#error{color:#b00020}#error:empty{display:none}
</style></head><body><header><h1>Dense Partition</h1>
<label>图像 <select id="image"></select></label>
<label><input type="checkbox" id="full">原尺寸</label><button id="clear">清除选中</button>
<nav class="links"><a href="overview.jpg">Overview</a><a href="summary.json">配置与统计</a></nav>
</header><main><p class="hint">点击任意图像，比较各方法在该位置的最终 Mask。</p>
<p id="error" role="alert"></p><div class="scroll"><div id="grid"></div></div></main>
<script>
const DATA=__DATA__,byId=id=>document.getElementById(id);
let active=[],point=null,token=0;
DATA.images.forEach((im,i)=>byId('image').add(new Option(im.name,i)));
function loadImage(src){return new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=()=>reject(new Error('图像加载失败'));im.src=src;});}
async function update(){
 const current=++token;active=[];point=null;byId('error').textContent='';
 const im=DATA.images[Number(byId('image').value)],grid=byId('grid');grid.replaceChildren();
 grid.style.setProperty('--image-width',im.width+'px');
 const pending=[];
 for(const method of Object.keys(DATA.methods)){
  const result=im.results[method],section=document.createElement('section');section.className='method';
  const header=document.createElement('div');header.className='method-header';
  const title=document.createElement('h2');title.textContent=DATA.methods[method];
  const counts=document.createElement('span');counts.className='counts';
  counts.textContent='Raw: '+(result.raw_output?.mask_count??'—')+' → Final: '+result.region_count+' masks';
  const selection=document.createElement('span');selection.className='selection';
  const download=document.createElement('button');download.textContent='下载 Mask';download.hidden=true;
  header.append(title,counts,selection,download);section.append(header);
  const views=document.createElement('div');views.className='views';section.append(views);grid.append(section);
  const entry={method,result,canvases:[],selection,download,ids:null};active.push(entry);
  for(const [name,label] of DATA.columns){
   const figure=document.createElement('figure'),caption=document.createElement('figcaption');caption.textContent=label;
   const wrap=document.createElement('div');wrap.className='picture';const image=document.createElement('img');
   image.src=result.directory+'/'+name+'.png';image.alt=DATA.methods[method]+' · '+label;image.draggable=false;
   const canvas=document.createElement('canvas');canvas.width=im.width;canvas.height=im.height;
   wrap.append(image,canvas);figure.append(caption,wrap);views.append(figure);entry.canvases.push({name,canvas});
   wrap.onclick=e=>{const rect=wrap.getBoundingClientRect();point=[Math.max(0,Math.min(im.width-1,Math.floor((e.clientX-rect.left)*im.width/rect.width))),Math.max(0,Math.min(im.height-1,Math.floor((e.clientY-rect.top)*im.height/rect.height)))];highlight();};
  }
  download.onclick=()=>{if(!point||!entry.ids)return;const id=entry.ids[point[1]*im.width+point[0]];
   const c=document.createElement('canvas');c.width=im.width;c.height=im.height;const ctx=c.getContext('2d'),pixels=ctx.createImageData(im.width,im.height);
   for(let i=0;i<entry.ids.length;i++){const v=entry.ids[i]===id?255:0;pixels.data.set([v,v,v,255],i*4);}ctx.putImageData(pixels,0,0);
   const link=document.createElement('a');link.download=im.key+'_'+method+'_mask_'+id+'.png';link.href=c.toDataURL('image/png');link.click();};
  pending.push(loadImage(result.id_image).then(labelImage=>{
   if(current!==token)return;
   const c=document.createElement('canvas');c.width=im.width;c.height=im.height;const ctx=c.getContext('2d',{willReadFrequently:true});ctx.drawImage(labelImage,0,0);
   const pixels=ctx.getImageData(0,0,im.width,im.height).data;entry.ids=new Uint32Array(im.width*im.height);
   for(let i=0;i<entry.ids.length;i++)entry.ids[i]=pixels[4*i]+(pixels[4*i+1]<<8)+(pixels[4*i+2]<<16);
  }));
 }
 try{await Promise.all(pending);if(current===token)highlight();}
 catch(e){if(current===token)byId('error').textContent=e.message;}
}
function highlight(){
 const im=DATA.images[Number(byId('image').value)];
 for(const entry of active){
  entry.download.hidden=true;entry.selection.textContent='';let pixels=null;
  if(point&&entry.ids){
   const id=entry.ids[point[1]*im.width+point[0]],region=entry.result.regions[id-1];
   pixels=entry.canvases[0].canvas.getContext('2d').createImageData(im.width,im.height);
   for(let i=0;i<entry.ids.length;i++)if(entry.ids[i]===id)pixels.data.set([0,255,190,180],4*i);
   entry.selection.textContent='('+point.join(', ')+') · Mask #'+id+' · '+region.area+' px';entry.download.hidden=false;
  }
  for(const {name,canvas} of entry.canvases){
   const ctx=canvas.getContext('2d');ctx.clearRect(0,0,im.width,im.height);
   if(!point)continue;
   if(pixels&&name!=='original'&&name!=='raw_colors')ctx.putImageData(pixels,0,0);
   ctx.strokeStyle='#ff255d';ctx.lineWidth=2;ctx.strokeRect(point[0]-4,point[1]-4,8,8);
  }
 }
}
byId('image').onchange=()=>update().catch(e=>byId('error').textContent=e.message);
byId('full').onchange=()=>byId('grid').classList.toggle('full',byId('full').checked);
byId('clear').onclick=()=>{point=null;highlight();};update().catch(e=>byId('error').textContent=e.message);
</script></body></html>'''
    (output / 'index.html').write_text(page.replace('__DATA__',
        json.dumps(data, ensure_ascii=False).replace('</', '<\\/')), encoding='utf-8')
