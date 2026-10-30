// Bridge between the renderer and the filesystem (only what the player needs).
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
  listPacks: () => ipcRenderer.invoke('list-packs'),
  pickPack: () => ipcRenderer.invoke('pick-pack'),
  loadPack: (dir) => ipcRenderer.invoke('load-pack', dir),
  loadRatings: (dir) => ipcRenderer.invoke('load-ratings', dir),
  saveRating: (dir, itemId, payload) => ipcRenderer.invoke('save-rating', dir, itemId, payload),
  exportCsv: (dir) => ipcRenderer.invoke('export-csv', dir),
  onOpenPack: (cb) => ipcRenderer.on('open-pack', (_e, dir) => cb(dir)),
});
