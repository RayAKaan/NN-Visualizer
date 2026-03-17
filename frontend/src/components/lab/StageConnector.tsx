import type { StageStatus } from "../../types/pipeline";

interface Props {
  fromStatus?: StageStatus;
  toStatus?: StageStatus;
}

export function StageConnector({ fromStatus, toStatus }: Props) {
  const live = fromStatus === "completed" || fromStatus === "active";
  const processing = toStatus === "processing";

  return (
    <div className="flex relative h-12 flex-col items-center justify-center overflow-hidden">
      {/* Container Track */}
      <div className="absolute h-full w-[3px] rounded-full bg-[rgba(36,40,54,0.6)] shadow-inner" />
      
      {/* Glowing Filled Track */}
      <div
        className="absolute top-0 h-full w-[3px] rounded-full transition-all duration-500 ease-out"
        style={{
          background: live ? "linear-gradient(to bottom, #06B6D4, rgba(6,182,212,0.3))" : "transparent",
          boxShadow: live ? "0 0 12px rgba(6,182,212,0.6), 0 0 24px rgba(6,182,212,0.3)" : "none",
          opacity: processing ? 0.8 : 0.4,
          height: live ? "100%" : "0%"
        }}
      />
      
      {/* Animated Flow Dot */}
      {live && processing && (
        <div className="absolute w-[9px] h-[9px] rounded-full bg-[#efedfc] shadow-[0_0_12px_#00f0ff,0_0_24px_#3b82f6] animate-pipelineFlow" />
      )}
      
      {/* Arrow Head */}
      <div 
        className="absolute bottom-0 z-10 h-0 w-0 border-l-[7px] border-r-[7px] border-t-[9px] border-l-transparent border-r-transparent transition-colors duration-300" 
        style={{ borderTopColor: live ? "#06B6D4" : "rgba(100,116,139,0.3)" }}
      />
    </div>
  );
}
