import { Navigate, Route, Routes } from 'react-router-dom'
import { AppShell } from './components/AppShell'
import { AnalysisProvider } from './state/AnalysisContext'
import { HealthProvider } from './state/HealthContext'
import { PrefsProvider } from './state/PrefsContext'
import { About } from './pages/About'
import { Analytics } from './pages/Analytics'
import { Dashboard } from './pages/Dashboard'
import { DamageMapPage } from './pages/DamageMapPage'
import { Reports } from './pages/Reports'
import { Results } from './pages/Results'
import { SettingsPage } from './pages/SettingsPage'
import { Upload } from './pages/Upload'

export function App() {
  return (
    <PrefsProvider>
      <HealthProvider>
        <AnalysisProvider>
          <Routes>
            <Route element={<AppShell />}>
              <Route index element={<Dashboard />} />
              <Route path="upload" element={<Upload />} />
              <Route path="results" element={<Results />} />
              <Route path="map" element={<DamageMapPage />} />
              <Route path="analytics" element={<Analytics />} />
              <Route path="reports" element={<Reports />} />
              <Route path="settings" element={<SettingsPage />} />
              <Route path="about" element={<About />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Route>
          </Routes>
        </AnalysisProvider>
      </HealthProvider>
    </PrefsProvider>
  )
}
