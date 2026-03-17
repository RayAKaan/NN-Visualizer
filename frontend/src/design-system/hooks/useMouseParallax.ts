import { useEffect, useState } from 'react';

export function useMouseParallax(depth = 2) {
  const [offset, setOffset] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const onMove = (event: MouseEvent) => {
      const x = (event.clientX / window.innerWidth - 0.5) * depth * 2;
      const y = (event.clientY / window.innerHeight - 0.5) * depth * 2;
      setOffset({ x: -x, y: -y });
    };
    window.addEventListener('mousemove', onMove);
    return () => window.removeEventListener('mousemove', onMove);
  }, [depth]);

  return offset;
}
