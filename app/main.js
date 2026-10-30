// Electron wrapper for the blind comparison player.
//
// The wrapper only provides filesystem access and a window: a pack is an
// ordinary folder (audio/*.flac + pack.json), ratings are written alongside
// in ratings.json. No network, no background processes.
const { app, BrowserWindow, ipcMain, dialog, protocol, net } = require('electron');
const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');

// Custom protocol for pack files: the page lives in app/ while the audio sits
// in the pack folder (outside the app directory), and Chromium will not hand
// file:// across the two directly.
protocol.registerSchemesAsPrivileged([
  { scheme: 'pack', privileges: { standard: true, stream: true, supportFetchAPI: true } },
]);
let currentPackDir = null;

const DATA_ROOT = process.env.RESTORE_DATA_ROOT || '/media/mydum/Data/Music/restore';
const LISTEN_ROOT = path.join(DATA_ROOT, 'dataset', 'listen');

let win = null;

function createWindow(packDir) {
  win = new BrowserWindow({
    width: 1100,
    height: 760,
    backgroundColor: '#12141a',
    title: 'restore-listen',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  win.loadFile(path.join(__dirname, 'renderer', 'index.html'));
  win.webContents.on('did-finish-load', () => {
    if (packDir) win.webContents.send('open-pack', packDir);
  });
}

function listPacks() {
  if (!fs.existsSync(LISTEN_ROOT)) return [];
  return fs.readdirSync(LISTEN_ROOT)
    .map((name) => {
      const dir = path.join(LISTEN_ROOT, name);
      const pj = path.join(dir, 'pack.json');
      if (!fs.statSync(dir).isDirectory() || !fs.existsSync(pj)) return null;
      const pack = JSON.parse(fs.readFileSync(pj, 'utf8'));
      return { dir, name, n_items: pack.n_items, created: pack.created, has_ours: pack.has_ours };
    })
    .filter(Boolean)
    .sort((a, b) => (b.created || '').localeCompare(a.created || ''));
}

ipcMain.handle('list-packs', () => listPacks());

ipcMain.handle('pick-pack', async () => {
  const res = await dialog.showOpenDialog(win, {
    title: 'Pick a pack to listen to',
    defaultPath: fs.existsSync(LISTEN_ROOT) ? LISTEN_ROOT : undefined,
    properties: ['openDirectory'],
  });
  if (res.canceled || !res.filePaths.length) return null;
  return res.filePaths[0];
});

ipcMain.handle('load-pack', (_e, dir) => {
  const pj = path.join(dir, 'pack.json');
  if (!fs.existsSync(pj)) throw new Error('no pack.json in ' + dir);
  const pack = JSON.parse(fs.readFileSync(pj, 'utf8'));
  pack.dir = dir;
  currentPackDir = dir;
  return pack;
});

ipcMain.handle('load-ratings', (_e, dir) => {
  const f = path.join(dir, 'ratings.json');
  if (!fs.existsSync(f)) return { ratings: {} };
  try {
    return JSON.parse(fs.readFileSync(f, 'utf8'));
  } catch (e) {
    return { ratings: {} };
  }
});

ipcMain.handle('save-rating', (_e, dir, itemId, payload) => {
  const f = path.join(dir, 'ratings.json');
  let doc = { ratings: {} };
  if (fs.existsSync(f)) {
    try { doc = JSON.parse(fs.readFileSync(f, 'utf8')); } catch (e) { /* overwrite */ }
  }
  doc.ratings[itemId] = { ...payload, at: new Date().toISOString() };
  doc.updated = new Date().toISOString();
  fs.writeFileSync(f, JSON.stringify(doc, null, 1), 'utf8');
  return true;
});

ipcMain.handle('export-csv', (_e, dir) => {
  const f = path.join(dir, 'ratings.json');
  const rows = ['item_id,labels,answer,choice,variant'];
  if (fs.existsSync(f)) {
    const doc = JSON.parse(fs.readFileSync(f, 'utf8'));
    for (const [id, r] of Object.entries(doc.ratings || {})) {
      rows.push([id, JSON.stringify(r.labels || {}), r.choice || '', r.variant || ''].join(','));
    }
  }
  const out = path.join(dir, 'ratings.csv');
  fs.writeFileSync(out, rows.join('\n'), 'utf8');
  return out;
});

app.whenReady().then(() => {
  protocol.handle('pack', (req) => {
    const u = new URL(req.url);
    const rel = decodeURIComponent(u.pathname).replace(/^\/+/, '');
    const abs = path.resolve(currentPackDir || '', rel);
    if (!currentPackDir || !abs.startsWith(path.resolve(currentPackDir))) {
      return new Response('forbidden', { status: 403 });
    }
    return net.fetch(pathToFileURL(abs).toString());
  });
  const argPack = process.argv.find((a) => a.startsWith('--pack='));
  const packs = listPacks();
  const initial = argPack ? argPack.split('=')[1] : (packs[0] ? packs[0].dir : null);
  createWindow(initial);
  app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(initial); });
});

app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });
