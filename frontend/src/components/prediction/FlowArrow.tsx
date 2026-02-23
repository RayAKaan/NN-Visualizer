import React from "react";

interface Props {
  label: string;
}

export default function FlowArrow({ label }: Props) {
  return (
    <div className="flow-arrow">
      <span>{label}</span>
    </div>
  );
}
