/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        neural: {
          void: 'var(--neural-void)',
          abyss: 'var(--neural-abyss)',
          obsidian: 'var(--neural-obsidian)',
          slate: 'var(--neural-slate)',
          graphite: 'var(--neural-graphite)',
          steel: 'var(--neural-steel)',
          silver: 'var(--neural-silver)',
          cloud: 'var(--neural-cloud)',
          pearl: 'var(--neural-pearl)',
          white: 'var(--neural-white)',
          ash: 'var(--neural-ash)',
          synapse: 'var(--neural-synapse)',
          axon: 'var(--neural-axon)',
          dendrite: 'var(--neural-dendrite)',
          soma: 'var(--neural-soma)',
          cortex: 'var(--neural-cortex)',
          lesion: 'var(--neural-lesion)',
          myelin: 'var(--neural-myelin)',
        },
      },
      fontFamily: {
        ui: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'SF Mono', 'monospace'],
      },
    },
  },
  plugins: [],
}
