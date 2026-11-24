import { useEffect, useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { BarChart3, FileText, Home, ImagePlus, Info, Map as MapIcon, Menu, ScanSearch, Settings, X } from 'lucide-react'
import { useHealth } from '../state/HealthContext'
import { fmtDateTime } from '../lib/derive'

const SIDE = [
  { to: '/', label: 'Dashboard', icon: Home, end: true },
  { to: '/upload', label: 'Upload', sub: 'Satellite Images', icon: ImagePlus },
  { to: '/results', label: 'View Results', icon: ScanSearch },
  { to: '/map', label: 'Damage Map', icon: MapIcon },
  { to: '/reports', label: 'Reports', icon: FileText },
  { to: '/about', label: 'About', icon: Info },
]
const TOP = [
  { to: '/', label: 'Dashboard', icon: Home, end: true },
  { to: '/analytics', label: 'Analytics', icon: BarChart3 },
  { to: '/reports', label: 'Reports', icon: FileText },
  { to: '/settings', label: 'Settings', icon: Settings },
]

export function AppShell() {
  const [open, setOpen] = useState(false)
  const loc = useLocation()
  const { checkedAt } = useHealth()
  useEffect(() => setOpen(false), [loc.pathname])
  useEffect(() => { window.scrollTo(0, 0) }, [loc.pathname])

  return (
    <div className="shell">
      <header className="topbar">
        <button className="menu-btn" onClick={() => setOpen((o) => !o)} aria-label={open ? 'Close navigation' : 'Open navigation'} aria-expanded={open}>
          {open ? <X size={22} /> : <Menu size={22} />}
        </button>
        <div className="brand">
          <img className="emblem" src="/logo.png" alt="EAGLE-DAR logo" width={56} height={56} />
          <div className="brand-sep" />
          <div>
            <div className="brand-name">EAGLE-DAR</div>
            <div className="brand-sub">Earthquake &amp; Disaster Assessment using AI for Rapid Response</div>
          </div>
        </div>
        <nav className="topnav" aria-label="Primary">
          {TOP.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.end} className={({ isActive }) => `toplink ${isActive ? 'active' : ''}`}>
              <n.icon size={18} /><span>{n.label}</span>
            </NavLink>
          ))}
          <div className="avatar" title="Local user — there is no sign-in in this prototype">U</div>
        </nav>
      </header>

      <div className={`scrim ${open ? 'show' : ''}`} onClick={() => setOpen(false)} />
      <aside className={`sidebar ${open ? 'open' : ''}`} aria-label="Sections">
        <nav>
          {SIDE.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.end} className={({ isActive }) => `sidelink ${isActive ? 'active' : ''}`}>
              <n.icon size={20} />
              <span>{n.label}{n.sub && <small>{n.sub}</small>}</span>
            </NavLink>
          ))}
          <div className="drawer-only">
            {TOP.slice(1).filter((t) => t.to !== '/reports').map((n) => (
              <NavLink key={n.to} to={n.to} className={({ isActive }) => `sidelink ${isActive ? 'active' : ''}`}><n.icon size={20} /><span>{n.label}</span></NavLink>
            ))}
          </div>
        </nav>
      </aside>

      <main className="workspace"><Outlet /></main>

      <footer className="footer">
        <span>EAGLE-DAR · research prototype · not an official government system</span>
        <span>Building a disaster-resilient workflow · Model outputs are estimates; verify before acting</span>
        <span>Backend last checked: {checkedAt ? fmtDateTime(checkedAt.toISOString()) : '—'}</span>
      </footer>
    </div>
  )
}
