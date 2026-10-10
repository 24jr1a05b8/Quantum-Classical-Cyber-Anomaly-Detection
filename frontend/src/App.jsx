import { Routes, Route, NavLink } from 'react-router-dom'
import Detect from './pages/Detect'
import Compare from './pages/Compare'
import About from './pages/About'

export default function App() {
  const linkClass = ({ isActive }) =>
    `px-4 py-2 rounded font-medium text-sm transition-colors ` +
    (isActive
      ? 'bg-violet-700 text-white'
      : 'text-gray-600 hover:bg-gray-200')

  return (
    <div className="min-h-screen flex flex-col">
      {/* ── Nav ─────────────────────────────────────────────────────────── */}
      <nav className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-6xl mx-auto px-4 py-3 flex items-center gap-6">
          <span className="font-bold text-violet-700 text-lg tracking-tight">
            QCAD
          </span>
          <NavLink to="/"        className={linkClass}>Detect</NavLink>
          <NavLink to="/compare" className={linkClass}>Compare</NavLink>
          <NavLink to="/about"   className={linkClass}>About</NavLink>
        </div>
      </nav>

      {/* ── Content ─────────────────────────────────────────────────────── */}
      <main className="flex-1 max-w-6xl w-full mx-auto px-4 py-8">
        <Routes>
          <Route path="/"        element={<Detect  />} />
          <Route path="/compare" element={<Compare />} />
          <Route path="/about"   element={<About   />} />
        </Routes>
      </main>

      <footer className="text-center text-xs text-gray-400 border-t py-3">
        Qiskit Fall Fest 2026 · IBM Quantum Hackathon ·{' '}
        <span className="italic">Quantum model runs on a simulator — no quantum advantage is claimed.</span>
      </footer>
    </div>
  )
}
