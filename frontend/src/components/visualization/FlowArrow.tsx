import React from "react";

interface Props {
  label: string;
}

const FlowArrow = React.memo(function FlowArrow({ label }: Props) {
  return (
    <div className="flow-arrow">
      <span>{label}</span>
    </div>
  );
});

export default FlowArrow;
