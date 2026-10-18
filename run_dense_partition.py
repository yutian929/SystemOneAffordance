"""Compare SAM 2.1, UnSAMv2+ and SLIC as dense, disjoint image partitions."""
import argparse
import base64
import csv
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from dense_partition import (GENERATOR, SLIC_CONFIG, export_partition, merge_tiny_regions,
                             refine_partition, render_partition, resolve_masks, slic_partition)
from experiment_utils import checkpoint_info, contact_sheet, digest, resolve, runtime, write_json
from unsamv2_backend import (CODE_REVISION, CHECKPOINT_SHA256, default_code_dir,
                             generate_candidates)

METHOD_NAMES = {'sam2': 'SAM 2.1 small', 'unsamv2': 'UnSAMv2+', 'slic': 'SLIC (DIP)'}


class ProposalGenerator:
    def __init__(self, device, granularity):
        self.device, self.granularity = device, granularity
        self.sam = None
        self.metadata = {
            'sam2': checkpoint_info(resolve('models/SAM2.1-small')),
            'unsamv2': checkpoint_info(resolve('models/UnSAMv2-plus'))}
        if self.metadata['unsamv2']['revision'] != CHECKPOINT_SHA256:
            raise ValueError('UnSAMv2+ checkpoint must match the pinned release')

    def generate(self, method, image, rgb, directory, cache=None):
        sha = digest(image)
        expected = dict(method=method, image_sha256=sha, generator=GENERATOR,
                        checkpoint=self.metadata[method],
                        granularity=self.granularity if method == 'unsamv2' else None,
                        code_revision=CODE_REVISION if method == 'unsamv2' else None)
        directory.mkdir(parents=True, exist_ok=False)
        source = cache / 'proposals' / method / sha[:16] if cache is not None else None
        if source is not None and (source / 'metadata.json').is_file():
            metadata = json.loads((source / 'metadata.json').read_text())
            for field, value in expected.items():
                if metadata.get(field) != value:
                    raise ValueError(f'Cached proposal {field} differs: {source}')
            shutil.copy2(source / 'masks.npz', directory / 'masks.npz')
            metadata['cache_source'] = str(source)
            with np.load(directory / 'masks.npz', allow_pickle=False) as archive:
                masks = archive['masks']
        else:
            if method == 'unsamv2':
                proposals, timing = generate_candidates(
                    image, resolve('models/UnSAMv2-plus'), default_code_dir(), self.device,
                    self.granularity, GENERATOR, directory / 'backend.log')
            else:
                import torch
                from sam2.build_sam import build_sam2
                from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
                cuda = self.device.startswith('cuda')
                if cuda:
                    torch.cuda.set_device(self.device)
                torch.manual_seed(0)
                torch.set_float32_matmul_precision('highest')
                torch.backends.cudnn.allow_tf32 = False
                def sync():
                    if cuda:
                        torch.cuda.synchronize(self.device)
                load_seconds = 0.0
                if self.sam is None:
                    start = time.perf_counter()
                    self.sam = build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml',
                        str(resolve('models/SAM2.1-small/sam2.1_hiera_small.pt')),
                        device=self.device, apply_postprocessing=False)
                    sync()
                    load_seconds = time.perf_counter() - start
                generator = SAM2AutomaticMaskGenerator(self.sam, **GENERATOR)
                sync()
                start = time.perf_counter()
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=cuda):
                    proposals = generator.generate(rgb)
                sync()
                timing = dict(generate_seconds=time.perf_counter() - start, load_seconds=load_seconds)
                # AMG can leave a large CUDA allocator cache. Release it before
                # starting the independent UnSAM worker on the same GPU.
                del generator
                self.sam = None
                gc.collect()
                if cuda:
                    torch.cuda.empty_cache()
            masks = np.stack([p['segmentation'] for p in proposals]) if proposals else np.zeros(
                (0, *rgb.shape[:2]), dtype=bool)
            records = [{k: v for k, v in p.items() if k != 'segmentation'} for p in proposals]
            metadata = {**expected, 'image': str(image), 'records': records, 'timing': timing,
                        'apply_postprocessing': False, 'seed': 0, 'proposal_cap': None}
            np.savez_compressed(directory / 'masks.npz', masks=masks)
        if masks.dtype != bool or masks.shape != (len(metadata['records']), *rgb.shape[:2]):
            raise ValueError('Invalid proposal array')
        write_json(directory / 'metadata.json', metadata)
        return masks, metadata


def save_result(output, image_info, method, preset, rgb, labels, owners, extra):
    relative = Path('partitions') / image_info['key'] / method / preset
    directory = output / relative
    stats, regions = export_partition(directory, labels, owners)
    figures = render_partition(rgb, labels, owners)
    for name, figure in figures.items():
        figure.save(directory / f'{name}.png')
    record = dict(image_key=image_info['key'], image=image_info['image'], method=method,
                  preset=preset, path=str(relative), **stats, **extra)
    write_json(directory / 'metadata.json', {**record, 'regions': regions,
               'encoding': 'labels.npy: uint32 HxW, IDs 1..K. masks.npz: packed_masks uint8 '
                           'KxHxceil(W/8), numpy.packbits axis=2, bitorder=big; shape=H,W; ids=1..K.'})
    view = {**record, 'regions': regions, 'id_image': 'data:image/png;base64,' +
            base64.b64encode((directory / 'ids.png').read_bytes()).decode('ascii')}
    return record, view, figures['boundaries']


def dip_study(images, output):
    """A small fixed RGB-only comparison supporting the choice of SLIC."""
    from skimage.color import rgb2gray
    from skimage.filters import gaussian, sobel
    from skimage.segmentation import felzenszwalb, watershed
    from dense_partition import connected_labels, validate_partition
    directory = output / 'dip_study'
    directory.mkdir()
    records, panels = [], []
    for info in images:
        rgb = np.array(Image.open(resolve(info['image'])).convert('RGB'))
        panels.append((Path(info['image']).parent.name, Image.fromarray(rgb)))
        methods = [
            ('SLIC 600', lambda: slic_partition(rgb, 600), dict(n_segments=600, **SLIC_CONFIG)),
            ('SLIC 1200', lambda: slic_partition(rgb, 1200), dict(n_segments=1200, **SLIC_CONFIG)),
            ('Felzenszwalb', lambda: connected_labels(felzenszwalb(
                rgb, scale=40, sigma=0.5, min_size=25)), dict(scale=40, sigma=0.5, min_size=25)),
            ('Compact watershed', lambda: connected_labels(watershed(
                sobel(gaussian(rgb2gray(rgb), sigma=0.5)), markers=600,
                compactness=0.001, watershed_line=False)),
                dict(markers=600, compactness=0.001, sigma=0.5, watershed_line=False))]
        for name, function, config in methods:
            start = time.perf_counter()
            labels = function()
            seconds = time.perf_counter() - start
            stats = validate_partition(labels)
            record = dict(image=info['image'], method=name, config=config, seconds=seconds, **stats)
            records.append(record)
            panels.append((f'{name} K={stats["region_count"]}',
                           render_partition(rgb, labels)['boundaries']))
    contact_sheet(panels, directory / 'overview.jpg', columns=5, cell_width=640)
    write_json(directory / 'metrics.json', records)


def write_viewer(output, images, views, segments, granularity):
    data = dict(images=images, results=views, segments=segments, granularity=granularity,
                methods=METHOD_NAMES)
    page = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dense Partition · 三方法对照</title>
<style>
*{box-sizing:border-box}body{font:15px system-ui,sans-serif;color:#1c2735;background:#f3f5f8;margin:0}
header{padding:20px 24px;background:#fff;border-bottom:1px solid #dce1e8;position:sticky;top:0;z-index:3}
h1{font-size:22px;margin:0 0 12px}p{line-height:1.6;margin:8px 0}.controls{display:flex;flex-wrap:wrap;gap:14px;align-items:center}
select,button{font:inherit;padding:6px 9px}a{color:#175f9e}main{padding:20px 24px}.scroll{overflow:auto}
.grid{display:grid;grid-template-columns:repeat(4,minmax(260px,1fr));gap:14px}.grid.full{grid-template-columns:repeat(4,640px)}
article{background:white;border:1px solid #dce1e8;border-radius:7px;padding:10px}h2{font-size:17px;margin:0 0 8px}
.picture{position:relative;line-height:0;cursor:crosshair}.picture img{width:100%;height:auto}.picture canvas{position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none}
.stats,.selection{font-size:13px;line-height:1.65;min-height:45px}.ok{color:#147343}.note{padding:12px 16px;background:#e8eef7;margin-bottom:16px}
button:disabled{opacity:.4}#error{color:#b00020}.legend{font-size:13px}footer{margin-top:20px}
</style>
<header><h1>细粒度 Dense Partition：SAM 2.1 / UnSAMv2+ / SLIC</h1>
<div class="controls"><label>图像 <select id="image"></select></label>
<label>分区 <select id="preset"></select></label>
<label>显示 <select id="display"><option value="boundaries">边界叠加</option><option value="overlay">半透明彩色</option><option value="colors">纯色 Mask</option><option value="coverage">模型原始未覆盖区（粉色）</option></select></label>
<label><input type="checkbox" id="full">原尺寸查看</label><button id="clear">清除选中</button></div>
<p class="legend">点击任意图像：同步查看该像素在三个方案中所属的 Mask，并可下载。彩色和边界是显示效果，边界像素也有唯一归属。</p>
</header><main><div class="note" id="explanation"></div><p id="error"></p><div class="scroll"><div class="grid" id="grid"></div></div>
<footer><p><a href="summary.csv">指标 CSV</a> · <a href="summary.json">配置与验证 JSON</a> · <a href="dip_study/overview.jpg">传统算法选型对照</a></p>
<p>全部结果只使用 RGB。未调用 Valen，未使用任务指令或 GT。所有导出的 Mask 已重新读入并验证：每像素恰好属于一个 Mask；每个 Mask 为 4 连通区域。</p></footer></main>
<script>
const DATA=__DATA__;
const byId=id=>document.getElementById(id);
let active=[],token=0,point=null;
DATA.images.forEach((im,i)=>byId('image').add(new Option(im.name+' · '+im.tasks.join(', '),i)));
byId('preset').add(new Option('模型分区：解重叠 + 连通空白区','native'));
DATA.segments.forEach(n=>byId('preset').add(new Option('全图细分：SLIC 目标 '+n+' 区域','s'+n)));
byId('preset').value='native';
function imageLoad(src){return new Promise((resolve,reject)=>{let image=new Image();image.onload=()=>resolve(image);image.onerror=()=>reject(new Error('图片加载失败：'+src));image.src=src;});}
async function update(){
 const current=++token;point=null;active=[];byId('error').textContent='';
 const im=DATA.images[Number(byId('image').value)],preset=byId('preset').value,display=byId('display').value;
 byId('explanation').textContent=preset==='native'?
 '模型分区：较小候选优先占有重叠像素；未覆盖像素按连通域补齐，再合并小于 16 像素的区域。粉色区域并非模型直接预测。SLIC 列显示第一档超像素分区。':
 '全图细分：两个模型的互斥分区分别与同一份 SLIC 分区取交集，保留模型分区边界并细分大区域。模型两列是“模型 + SLIC”的组合结果；SLIC 列是纯传统算法。目标区域数是 SLIC 参数，三列最终 Mask 数量不强行相同。';
 const grid=byId('grid');grid.replaceChildren();
 for(const method of ['original','sam2','unsamv2','slic']){
  const result=method==='original'?null:DATA.results.find(r=>r.image_key===im.key&&r.method===method&&r.preset===preset);
  const article=document.createElement('article'),title=document.createElement('h2');
  title.textContent=method==='original'?'原图':DATA.methods[method]+(method==='unsamv2'?' · g='+DATA.granularity:'');
  article.append(title);
  const wrap=document.createElement('div');wrap.className='picture';const image=document.createElement('img');
  image.alt=title.textContent;image.src=method==='original'?im.original:result.path+'/'+display+'.png';
  const canvas=document.createElement('canvas');canvas.width=im.width;canvas.height=im.height;
  wrap.append(image,canvas);article.append(wrap);
  const stats=document.createElement('p');stats.className='stats';
  stats.textContent=result?'K='+result.region_count+' · 中位面积 '+result.median_area+' px · 最大 '+result.maximum_area+' px':'';
  if(result){const ok=document.createElement('div');ok.className='ok';ok.textContent='覆盖 100% · 重叠 0 · 连通检查通过';stats.append(ok);
   if(method!=='slic'){const raw=document.createElement('div');raw.textContent='模型原始覆盖 '+(result.raw_coverage_fraction*100).toFixed(2)+'%';stats.append(raw);}}
  article.append(stats);const selection=document.createElement('p');selection.className='selection';selection.textContent='点击图像选择像素';article.append(selection);
  const download=document.createElement('button');download.textContent='下载选中 Mask';download.disabled=true;if(result)article.append(download);
  grid.append(article);const entry={method,result,canvas,selection,download,ids:null};active.push(entry);
  wrap.onclick=e=>{const rect=wrap.getBoundingClientRect();point=[Math.min(im.width-1,Math.floor((e.clientX-rect.left)*im.width/rect.width)),Math.min(im.height-1,Math.floor((e.clientY-rect.top)*im.height/rect.height))];highlight();};
  download.onclick=()=>{if(!point||!entry.ids)return;const maskId=entry.ids[point[1]*im.width+point[0]];const c=document.createElement('canvas');c.width=im.width;c.height=im.height;const ctx=c.getContext('2d');const pixels=ctx.createImageData(im.width,im.height);for(let i=0;i<entry.ids.length;i++){let v=entry.ids[i]===maskId?255:0;pixels.data.set([v,v,v,255],4*i);}ctx.putImageData(pixels,0,0);const a=document.createElement('a');a.href=c.toDataURL('image/png');a.download=im.key+'_'+method+'_'+preset+'_mask_'+maskId+'.png';a.click();};
  if(result){const labelImage=await imageLoad(result.id_image);if(current!==token)return;
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
  const id=entry.ids[point[1]*im.width+point[0]],region=entry.result.regions[id-1];
  const pixels=ctx.createImageData(im.width,im.height);for(let i=0;i<entry.ids.length;i++)if(entry.ids[i]===id)pixels.data.set([0,255,190,180],4*i);
  ctx.putImageData(pixels,0,0);ctx.strokeStyle='#ff255d';ctx.lineWidth=2;ctx.strokeRect(point[0]-4,point[1]-4,8,8);
  entry.selection.textContent='像素 ('+point.join(', ')+') → Mask #'+id+' · '+region.area+' px'+(entry.method==='slic'?'':' · 原模型覆盖 '+region.model_covered_pixels+' px');entry.download.disabled=false;
 }
}
for(const id of ['image','preset','display'])byId(id).onchange=()=>update().catch(e=>byId('error').textContent=e.message);
byId('full').onchange=()=>byId('grid').classList.toggle('full',byId('full').checked);
byId('clear').onclick=()=>{point=null;highlight();};
update().catch(e=>byId('error').textContent=e.message);
</script></html>'''
    (output / 'index.html').write_text(page.replace('__DATA__', json.dumps(data, ensure_ascii=False).replace('</', '<\\/')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='configs/umd_debug.json')
    parser.add_argument('--output', default='outputs/dense_partition_debug')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--granularity', type=float, default=0.15)
    parser.add_argument('--segments', nargs='+', type=int, default=[600, 1200])
    parser.add_argument('--proposal-cache', help='Reuse compatible saved proposals; generate missing entries')
    args = parser.parse_args()
    if not np.isfinite(args.granularity) or not 0.1 <= args.granularity <= 1:
        parser.error('granularity must be finite and in [0.1,1]')
    if any(n < 1 for n in args.segments) or len(set(args.segments)) != len(args.segments):
        parser.error('segments must be distinct positive integers')
    output = resolve(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(resolve(args.manifest).read_text())
    images_by_sha = {}
    for sample in manifest['samples']:
        image = resolve(sample['image']).resolve()
        sha = digest(image)
        if sha not in images_by_sha:
            with Image.open(image) as im:
                width, height = im.size
            images_by_sha[sha] = dict(key=sha[:16], image=str(image), image_sha256=sha,
                                      name=image.parent.name, width=width, height=height, tasks=[])
        images_by_sha[sha]['tasks'].append(sample['id'])
    images = list(images_by_sha.values())
    if not images:
        parser.error('manifest must contain images')
    config = dict(completed=False, images=images, manifest=str(resolve(args.manifest)),
                  generator=GENERATOR, granularity=args.granularity,
                  slic_config=SLIC_CONFIG, slic_segments=args.segments, connectivity=4,
                  overlap_rule='smaller area first, then higher quality/stability, then source index',
                  residual_rule='uncovered pixels form connected residual regions',
                  minimum_native_region_area=16,
                  tiny_region_merge='smallest first into adjacent closest mean Lab region',
                  fine_rule='connected intersections of native and SLIC partitions; no post-merge',
                  gt_used=False, valen_used=False, runtime=runtime(args.device))
    write_json(output / 'run.json', config)
    generator = ProposalGenerator(args.device, args.granularity)
    records, views = [], []
    overview = {preset: [] for preset in ['native', *[f's{n}' for n in args.segments]]}
    cache = resolve(args.proposal_cache).resolve() if args.proposal_cache else None
    for info in images:
        image = Path(info['image'])
        rgb = np.array(Image.open(image).convert('RGB'))
        original = Path('images') / f'{info["key"]}.png'
        (output / original).parent.mkdir(exist_ok=True)
        Image.fromarray(rgb).save(output / original)
        info['original'] = str(original)
        for panels in overview.values():
            panels.append((info['name'], Image.fromarray(rgb)))
        slic_maps, slic_seconds = {}, {}
        for n in args.segments:
            start = time.perf_counter()
            slic_maps[n] = slic_partition(rgb, n)
            slic_seconds[n] = time.perf_counter() - start
        for method in ['sam2', 'unsamv2', 'slic']:
            print(f'{info["name"]} / {method}', flush=True)
            owners, raw = None, {}
            if method != 'slic':
                masks, metadata = generator.generate(method, image, rgb,
                    output / 'proposals' / method / info['key'], cache)
                start = time.perf_counter()
                native, owners, raw = resolve_masks(masks, metadata['records'], rgb.shape[:2])
                del masks
                raw['regions_before_cleanup'] = int(native.max())
                native, merged = merge_tiny_regions(native, rgb, minimum=16)
                raw.update(merged_tiny_regions=merged, native_partition_seconds=time.perf_counter()-start,
                           source_generate_seconds=metadata['timing']['generate_seconds'])
                np.save(output / 'proposals' / method / info['key'] / 'pixel_owners.npy', owners)
            for preset in overview:
                n = args.segments[0] if preset == 'native' else int(preset[1:])
                start = time.perf_counter()
                labels = slic_maps[n] if method == 'slic' else (
                    native if preset == 'native' else refine_partition(native, slic_maps[n]))
                extra = {**raw, 'slic_n_segments': n if method == 'slic' or preset != 'native' else None,
                         'slic_seconds': slic_seconds[n] if method == 'slic' or preset != 'native' else 0.0,
                         'refine_seconds': time.perf_counter() - start,
                         'hybrid_with_slic': method != 'slic' and preset != 'native'}
                record, view, figure = save_result(output, info, method, preset, rgb, labels, owners, extra)
                records.append(record); views.append(view)
                overview[preset].append((f'{METHOD_NAMES[method]} K={record["region_count"]}', figure))
                print(f'  {preset}: K={record["region_count"]}, coverage=1, overlap=0, exported masks verified', flush=True)
    for preset, panels in overview.items():
        contact_sheet(panels, output / f'overview_{preset}.jpg', columns=4, cell_width=640)
    dip_study(images, output)
    config.update(completed=True, images=images, checkpoints=generator.metadata)
    write_json(output / 'run.json', config)
    report = dict(config=config, samples=records, verified_partition_count=len(records),
                  note='Model native partitions are postprocessed proposals; fine model partitions '
                       'are explicitly model+SLIC hybrids. Region count is not segmentation accuracy.')
    write_json(output / 'summary.json', report)
    fields = list(dict.fromkeys(key for record in records for key in record))
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(records)
    write_viewer(output, images, views, args.segments, args.granularity)
    print(f'Completed {len(images)} images / {len(records)} verified partitions: {output}', flush=True)


if __name__ == '__main__':
    main()
