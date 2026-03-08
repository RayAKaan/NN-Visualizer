import React, { useMemo, useState } from "react";

interface MapItem {
  mapId: number;
  data: number[] | number[][];
  mean?: number;
  kernel?: number[];
}

interface LayerItem {
  id: string;
  name: string;
  maps: MapItem[];
}

interface Props {
  activations: {
    layers?: LayerItem[];
  };
}

const cache = new Map<string, string>();

function makeHeatmap(data: number[] | number[][], w = 28, h = 28) {
  const flat = Array.isArray(data[0]) ? (data as number[][]).flat() : (data as number[]);
  const key = `${w}x${h}:${flat.slice(0, 60).join(",")}:${flat.length}`;
  if (cache.has(key)) return cache.get(key)!;
  const cv = document.createElement("canvas");
  cv.width = w;
  cv.height = h;
  const ctx = cv.getContext("2d");
  if (!ctx) return "";
  const img = ctx.createImageData(w, h);
  for (let i = 0; i < w * h; i += 1) {
    const v = Math.max(0, Math.min(1, Number(flat[i] ?? 0)));
    const off = i * 4;
    img.data[off] = Math.round(20 + v * 235);
    img.data[off + 1] = Math.round(170 + v * 85);
    img.data[off + 2] = Math.round(180 + v * 75);
    img.data[off + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  const url = cv.toDataURL();
  cache.set(key, url);
  if (cache.size > 200) {
    const first = cache.keys().next().value;
    if (first) cache.delete(first);
  }
  return url;
}

export function CNNFeatureGallery({ activations }: Props) {
  const [selected, setSelected] = useState<{ layer: string; mapId: number } | null>(null);

  const layers = useMemo(() => (Array.isArray(activations.layers) ? activations.layers : []), [activations.layers]);
  if (layers.length === 0) return <div className="text-xs text-slate-400">No CNN feature maps available.</div>;

  return (
    <div className="space-y-4" style={{ transform: "perspective(900px) rotateX(5deg)" }}>
      {layers.map((layer) => {
        const top = [...layer.maps].sort((a, b) => Number(b.mean ?? 0) - Number(a.mean ?? 0)).slice(0, 3).map((m) => m.mapId);
        const maps = layer.maps.slice(0, 64);
        return (
          <section key={layer.id} className="space-y-2">
            <h3 className="text-xs font-semibold text-cyan-200">{layer.name}</h3>
            <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 gap-2">
              {maps.map((map) => {
                const selectedThis = selected?.layer === layer.id && selected.mapId === map.mapId;
                const url = makeHeatmap(map.data);
                return (
                  <button
                    key={`${layer.id}-${map.mapId}`}
                    className="relative w-14 h-14 rounded-sm overflow-hidden border border-white/10 bg-black/30 transition-transform duration-150 hover:scale-110"
                    style={{
                      transform: selectedThis ? "scale(1.12) rotateY(6deg)" : "none",
                      boxShadow: top.includes(map.mapId) ? "0 0 12px rgba(34,211,238,0.45)" : "none",
                      borderColor: selectedThis ? "rgba(34,211,238,0.7)" : undefined,
                    }}
                    onClick={() => setSelected({ layer: layer.id, mapId: map.mapId })}
                    title={`Map ${map.mapId}`}
                  >
                    <img src={url} alt={`Feature map ${map.mapId} from ${layer.name}`} className="w-full h-full object-cover" />
                    {top.includes(map.mapId) && <span className="absolute top-0 right-0 text-[9px] bg-cyan-500/80 px-1">top</span>}
                  </button>
                );
              })}
            </div>
            {selected?.layer === layer.id && (
              <div className="flex flex-wrap items-center gap-3 rounded-md border border-white/10 bg-black/25 p-2">
                <img
                  src={makeHeatmap(layer.maps.find((m) => m.mapId === selected.mapId)?.data ?? [])}
                  alt="Selected map preview"
                  className="w-24 h-24 rounded bg-black/40 border border-white/10"
                />
                <div>
                  <div className="text-xs text-slate-300">Map {selected.mapId}</div>
                  <div className="text-[11px] text-slate-400 mb-1">Kernel preview</div>
                  <div className="grid grid-cols-3 gap-1">
                    {(layer.maps.find((m) => m.mapId === selected.mapId)?.kernel ?? []).slice(0, 9).map((k, i) => (
                      <div key={i} className="w-4 h-4 rounded-sm" style={{ background: `rgba(244,114,182,${Math.max(0.1, Math.min(1, Math.abs(Number(k))))})` }} />
                    ))}
                  </div>
                </div>
              </div>
            )}
          </section>
        );
      })}
    </div>
  );
}
