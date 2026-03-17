import type { FlowSnapshot } from "../../../types/flow";

interface Props {
  snapshot: FlowSnapshot;
  isActive: boolean;
  onClick: () => void;
}

export function FlowThumbnail({ snapshot, isActive, onClick }: Props) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="relative overflow-hidden rounded-lg transition-transform hover:scale-105"
      style={{
        width: isActive ? 52 : 40,
        height: isActive ? 52 : 40,
        border: isActive ? "2px solid var(--fwd)" : "1px solid var(--glass-border)",
      }}
      title={`${snapshot.stageId} (${snapshot.shape.join("x")})`}
    >
      <img
        src={snapshot.thumbnail.url}
        alt=""
        className="h-full w-full object-cover"
        style={{ imageRendering: snapshot.thumbnail.width < 32 ? "pixelated" : "auto" }}
      />
      <div
        className="absolute bottom-0 left-0 right-0 text-center text-[8px]"
        style={{ background: "rgba(0,0,0,0.55)", color: "var(--text-4)" }}
      >
        {snapshot.dimensionality === "3d" ? "3D" : snapshot.dimensionality === "2d" ? "2D" : "1D"}
      </div>
    </button>
  );
}
