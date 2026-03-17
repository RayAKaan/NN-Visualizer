import { Renderer, Program, Mesh, Triangle } from 'ogl';
import React, { useEffect, useRef, useState } from 'react';
import { useReducedMotion } from '@/design-system/hooks/useReducedMotion';
import { useSimulatorStore } from '@/store/simulatorStore';
import { useTrainingSimStore } from '@/store/trainingSimStore';
import { ambientFragment, ambientVertex } from './ambientShaders';

interface NeuralAmbientProps {
  className?: string;
}

export function NeuralAmbient({ className }: NeuralAmbientProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const reducedMotion = useReducedMotion();
  const forwardPassState = useSimulatorStore((s) => s.forwardPassState);
  const autoPlay = useSimulatorStore((s) => s.autoPlay);
  const isTraining = useTrainingSimStore((s) => s.isTraining);
  const [webglOk, setWebglOk] = useState(true);

  useEffect(() => {
    if (reducedMotion) return;
    const container = containerRef.current;
    if (!container) return;

    let renderer: Renderer;
    try {
      renderer = new Renderer({ dpr: Math.min(2, window.devicePixelRatio || 1) });
    } catch {
      setWebglOk(false);
      return;
    }
    const gl = renderer.gl;
    container.appendChild(gl.canvas);
    gl.clearColor(0, 0, 0, 1);

    const geometry = new Triangle(gl);
    const program = new Program(gl, {
      vertex: ambientVertex,
      fragment: ambientFragment,
      uniforms: {
        uTime: { value: 0 },
        uResolution: { value: [gl.canvas.width, gl.canvas.height] },
        uState: { value: 0 },
        uMouse: { value: [0.5, 0.5] },
        uBrightness: { value: 1 },
      },
    });
    const mesh = new Mesh(gl, { geometry, program });

    let running = true;
    let lastTime = 0;

    const resize = () => {
      const { clientWidth, clientHeight } = container;
      renderer.setSize(clientWidth, clientHeight);
      program.uniforms.uResolution.value = [clientWidth, clientHeight];
    };

    const onMouse = (event: MouseEvent) => {
      const rect = container.getBoundingClientRect();
      const x = (event.clientX - rect.left) / rect.width;
      const y = 1 - (event.clientY - rect.top) / rect.height;
      program.uniforms.uMouse.value = [x, y];
    };

    const onVisibility = () => {
      running = !document.hidden;
      if (running) requestAnimationFrame(render);
    };

    const render = (time: number) => {
      if (!running) return;
      const delta = time - lastTime;
      if (delta < 16) {
        requestAnimationFrame(render);
        return;
      }
      lastTime = time;
      program.uniforms.uTime.value = time * 0.001;
      program.uniforms.uState.value = isTraining ? 4 : forwardPassState === 'idle' && !autoPlay ? 0 : 2;
      program.uniforms.uBrightness.value = 0.6;
      renderer.render({ scene: mesh });
      requestAnimationFrame(render);
    };

    resize();
    window.addEventListener('resize', resize);
    window.addEventListener('mousemove', onMouse);
    document.addEventListener('visibilitychange', onVisibility);
    requestAnimationFrame(render);

    return () => {
      running = false;
      window.removeEventListener('resize', resize);
      window.removeEventListener('mousemove', onMouse);
      document.removeEventListener('visibilitychange', onVisibility);
      container.removeChild(gl.canvas);
      renderer.gl.getExtension('WEBGL_lose_context')?.loseContext();
    };
  }, [reducedMotion, forwardPassState, autoPlay, isTraining]);

  if (reducedMotion || !webglOk) {
    return <div className={`neural-ambient-fallback ${className ?? ''}`} />;
  }

  return <div ref={containerRef} className={`neural-ambient ${className ?? ''}`} />;
}
