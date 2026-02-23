import React, { useEffect, useRef, useState } from "react";

interface Props {
  content: string | React.ReactNode;
  position?: "top" | "bottom" | "left" | "right";
  children: React.ReactNode;
  delay?: number;
}

export default function Tooltip({ content, position = "top", children, delay = 300 }: Props) {
  const [visible, setVisible] = useState(false);
  const timeoutRef = useRef<number | null>(null);

  const handleMouseEnter = () => {
    timeoutRef.current = window.setTimeout(() => setVisible(true), delay);
  };

  const handleMouseLeave = () => {
    if (timeoutRef.current !== null) {
      window.clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    setVisible(false);
  };

  useEffect(() => {
    return () => {
      if (timeoutRef.current !== null) {
        window.clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return (
    <div className="tooltip-wrapper" onMouseEnter={handleMouseEnter} onMouseLeave={handleMouseLeave}>
      {children}
      {visible && <div className={`tooltip-content ${position}`}>{content}</div>}
    </div>
  );
}
