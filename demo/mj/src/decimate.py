# Panda 시각용 메시만 줄여 가벼운 사본을 만든다 (충돌 메시 *_c 와 STL 은 그대로 복사)
import os, shutil, trimesh, fast_simplification, numpy as np
SRC = 'menagerie/franka_emika_panda'; DST = 'panda_lite'
shutil.rmtree(DST, ignore_errors=True); os.makedirs(DST + '/assets')
for f in os.listdir(SRC):
    if f.endswith('.xml'): shutil.copy(f'{SRC}/{f}', DST)
tot0 = tot1 = 0
for f in sorted(os.listdir(SRC + '/assets')):
    p = f'{SRC}/assets/{f}'; q = f'{DST}/assets/{f}'
    if not f.endswith('.obj') or '_c' in f.split('.')[0][-3:]:
        shutil.copy(p, q); continue
    m = trimesh.load(p, force='mesh', process=False)
    m.merge_vertices(merge_tex=True, merge_norm=True)   # OBJ 는 꼭짓점이 면마다 쪼개져 있어 합치지 않으면 줄이기가 면을 통째로 지운다
    n0 = len(m.faces); tot0 += n0
    if n0 > 400:
        v, fc = fast_simplification.simplify(np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32), target_reduction=0.8)
        m = trimesh.Trimesh(v, fc, process=False)
    tot1 += len(m.faces)
    m.export(q)
print('visual faces', tot0, '->', tot1)
