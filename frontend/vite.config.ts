import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// The dev server runs on localhost:5173, which the backend's CORS list allows by default.
export default defineConfig({
  plugins: [react()],
  server: { host: 'localhost', port: 5173, strictPort: true },
  test: { environment: 'node', include: ['src/**/*.test.ts'] },
})
