import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Served by the FastAPI app at /app/ (src/main.py mounts web/dist there) —
// base must match so built asset URLs resolve correctly from that path.
export default defineConfig({
  plugins: [react()],
  base: '/app/',
  build: {
    outDir: 'dist',
  },
})
