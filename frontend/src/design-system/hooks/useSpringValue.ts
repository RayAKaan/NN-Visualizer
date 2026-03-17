import { useEffect, useRef, useState } from 'react';

export function useSpringValue(target: number, stiffness = 260, damping = 26) {
  const [value, setValue] = useState(target);
  const valueRef = useRef(target);
  const velocity = useRef(0);

  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const delta = target - valueRef.current;
      const accel = delta * (stiffness / 1000) - velocity.current * (damping / 1000);
      velocity.current += accel;
      valueRef.current += velocity.current;
      setValue(valueRef.current);
      if (Math.abs(delta) > 0.001 || Math.abs(velocity.current) > 0.001) {
        raf = requestAnimationFrame(tick);
      }
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, stiffness, damping]);

  return value;
}
