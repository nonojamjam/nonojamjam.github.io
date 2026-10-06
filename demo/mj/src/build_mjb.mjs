// 장면 XML 을 컴파일해 .mjb 바이너리로 저장 (사이트 전송량 줄이기)
import loadMujoco from '@mujoco/mujoco';
import fs from 'fs'; import path from 'path';
const mj = await loadMujoco();
function copy(dir, dst) { mj.FS.mkdirTree(dst); for (const f of fs.readdirSync(dir)) { const p = path.join(dir, f); if (fs.statSync(p).isDirectory()) copy(p, dst + '/' + f); else mj.FS.writeFile(dst + '/' + f, fs.readFileSync(p)); } }
copy(process.argv[2] || 'menagerie/franka_emika_panda', '/panda');
const model = mj.MjModel.from_xml_path('/panda/pickplace.xml');
console.log('nq', model.nq, 'nu', model.nu, 'ngeom', model.ngeom, 'nmesh', model.nmesh, 'nmeshvert', model.nmeshvert);
mj.mj_saveModel(model, '/out.mjb', null);
const buf = mj.FS.readFile('/out.mjb'); fs.writeFileSync('pickplace.mjb', buf);
console.log('mjb bytes', buf.length);
