import { useEffect, useState } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, ErrorBar
} from 'recharts'
import { API_BASE } from '../config'

const FPR_LABELS = { Recall_1: '1% target FPR', Recall_5: '5% target FPR' }

export default function Compare() {
  const [data,    setData]    = useState(null)
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState(null)
  const [tab,     setTab]     = useState('v2')

  useEffect(() => {
    axios.get(`${API_BASE}/results`)
      .then(r => { setData(r.data); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <Spinner />
  if (error)   return <ErrorBox msg={error} />

  const runData = data?.[tab]?.results || []
  const sumData = data?.[tab]?.summary  || []
  const perData = data?.[tab]?.per_attack_recall || []

  const recallChartData = buildRecallChart(sumData)
  const perAttackData   = buildPerAttackChart(perData)

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Model Comparison</h1>
        <p className="text-sm text-gray-500 mt-1">
          Classical (RBF) vs Quantum (ZZFeatureMap fidelity kernel) — Qiskit Statevector simulator.
        </p>
      </div>

      {/* Simulator banner */}
      <div className="bg-amber-50 border border-amber-300 rounded-lg px-4 py-3 text-sm text-amber-800">
        ⚠️ <strong>Quantum model runs on a Qiskit Statevector simulator.</strong>{' '}
        No quantum advantage or speedup is claimed. Hyperparameters were tuned on a small labelled
        sample of DoS and PortScan attacks from the same dataset families as the test set.
      </div>

      {/* Version tabs */}
      <div className="flex gap-2">
        {['v1_quick', 'v2'].map(v => (
          <button
            key={v}
            onClick={() => setTab(v)}
            className={`px-4 py-1.5 rounded text-sm font-medium border transition-colors ` +
              (tab === v ? 'bg-violet-700 text-white border-violet-700' : 'bg-white text-gray-600 border-gray-300 hover:bg-gray-50')}
          >
            {v === 'v1_quick' ? 'v1 Quick' : 'v2 Full'}
          </button>
        ))}
      </div>

      {/* Recall grouped bar chart */}
      {recallChartData.r1.length > 0 && (
        <Section title="Recall vs. PCA Components (mean ± std over seeds)">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {[['r1', 'Recall @ 1% target FPR'], ['r5', 'Recall @ 5% target FPR']].map(([key, title]) => (
              <div key={key}>
                <p className="text-xs text-gray-500 mb-2">{title}</p>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={recallChartData[key]} margin={{ top: 5, right: 5, bottom: 5, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="k" tickFormatter={v => `k=${v}`} tick={{ fontSize: 11 }} />
                    <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                    <Tooltip formatter={v => v.toFixed(3)} />
                    <Legend />
                    <Bar dataKey="classical" fill="#2563eb" name="Classical">
                      <ErrorBar dataKey="classical_err" width={4} strokeWidth={2} stroke="#1e40af" />
                    </Bar>
                    <Bar dataKey="quantum" fill="#7c3aed" name="Quantum (simulator)">
                      <ErrorBar dataKey="quantum_err" width={4} strokeWidth={2} stroke="#5b21b6" />
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* Measured FPR table */}
      {sumData.length > 0 && (
        <Section title="Measured FPR (mean over seeds) — target vs. actual">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 text-xs text-gray-500 uppercase">
                <tr>
                  <th className="px-3 py-2 text-left">k</th>
                  <th className="px-3 py-2 text-left">Model</th>
                  <th className="px-3 py-2 text-right">Recall @1%</th>
                  <th className="px-3 py-2 text-right">Meas. FPR @1%</th>
                  <th className="px-3 py-2 text-right">Recall @5%</th>
                  <th className="px-3 py-2 text-right">Meas. FPR @5%</th>
                  <th className="px-3 py-2 text-right">PR-AUC</th>
                  <th className="px-3 py-2 text-right">ROC-AUC</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {sumData.map((row, i) => (
                  <tr key={i}>
                    <td className="px-3 py-1.5">{row.k ?? '—'}</td>
                    <td className="px-3 py-1.5 capitalize">{row.model ?? '—'}</td>
                    <td className="px-3 py-1.5 text-right">{fmt(row['Recall_1'] ?? row['Recall_1_mean'])}</td>
                    <td className="px-3 py-1.5 text-right text-orange-600">{fmt(row['Test_FPR_1'] ?? row['Test_FPR_1_mean'])}</td>
                    <td className="px-3 py-1.5 text-right">{fmt(row['Recall_5'] ?? row['Recall_5_mean'])}</td>
                    <td className="px-3 py-1.5 text-right text-orange-600">{fmt(row['Test_FPR_5'] ?? row['Test_FPR_5_mean'])}</td>
                    <td className="px-3 py-1.5 text-right">{fmt(row['PRAUC'] ?? row['PRAUC_mean'])}</td>
                    <td className="px-3 py-1.5 text-right">{fmt(row['ROCAUC'] ?? row['ROCAUC_mean'])}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-gray-400 mt-2">
            Orange = measured FPR on test set. The threshold is derived from Monday benign flows;
            test benign flows come from Wednesday/Friday, so measured FPR may differ from target.
          </p>
        </Section>
      )}

      {/* Per-attack recall */}
      {perAttackData.length > 0 && (
        <Section title="Per-Attack Recall @ 5% target FPR (mean over seeds, highest k)">
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={perAttackData} margin={{ top: 5, right: 5, bottom: 40, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="attack" tick={{ fontSize: 10, angle: -20, textAnchor: 'end' }} interval={0} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
              <Tooltip formatter={v => v.toFixed(3)} />
              <Legend />
              <Bar dataKey="classical" fill="#2563eb" name="Classical" />
              <Bar dataKey="quantum"   fill="#7c3aed" name="Quantum (simulator)" />
            </BarChart>
          </ResponsiveContainer>
          {perAttackData.some(d => d.low_n) && (
            <p className="text-xs text-amber-700 mt-2">
              * low-n: attack type has fewer than 20 test samples — treat result with caution.
            </p>
          )}
        </Section>
      )}

      {(!runData.length && !sumData.length) && (
        <div className="text-gray-500 text-sm">
          No results found for <strong>{tab}</strong>. Run the experiment to generate results.
        </div>
      )}
    </div>
  )
}

// ── helpers ───────────────────────────────────────────────────────────────────

function fmt(v) {
  if (v == null || v === '') return '—'
  const n = parseFloat(v)
  return isNaN(n) ? '—' : n.toFixed(3)
}

function Section({ title, children }) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5 space-y-4">
      <h2 className="font-semibold text-gray-800">{title}</h2>
      {children}
    </div>
  )
}

function Spinner() {
  return <div className="text-center text-gray-500 py-20 text-sm">Loading results…</div>
}

function ErrorBox({ msg }) {
  return <div className="bg-red-50 border border-red-300 rounded-lg px-4 py-3 text-sm text-red-700">{msg}</div>
}

function buildRecallChart(summary) {
  if (!summary.length) return { r1: [], r5: [] }

  // summary rows: each row has k, model, and metric columns
  const byKModel = {}
  for (const row of summary) {
    const k = row.k
    const m = row.model
    if (!byKModel[k]) byKModel[k] = {}
    byKModel[k][m] = row
  }

  const r1 = [], r5 = []
  for (const [k, models] of Object.entries(byKModel)) {
    const c = models['classical'] || {}
    const q = models['quantum']   || {}
    r1.push({
      k: parseInt(k),
      classical: parseFloat(c['Recall_1'] ?? c['Recall_1_mean'] ?? 0),
      classical_err: parseFloat(c['Recall_1_std'] ?? 0),
      quantum:   parseFloat(q['Recall_1'] ?? q['Recall_1_mean'] ?? 0),
      quantum_err:   parseFloat(q['Recall_1_std'] ?? 0),
    })
    r5.push({
      k: parseInt(k),
      classical: parseFloat(c['Recall_5'] ?? c['Recall_5_mean'] ?? 0),
      classical_err: parseFloat(c['Recall_5_std'] ?? 0),
      quantum:   parseFloat(q['Recall_5'] ?? q['Recall_5_mean'] ?? 0),
      quantum_err:   parseFloat(q['Recall_5_std'] ?? 0),
    })
  }
  return { r1: r1.sort((a, b) => a.k - b.k), r5: r5.sort((a, b) => a.k - b.k) }
}

function buildPerAttackChart(perRows) {
  if (!perRows.length) return []
  const maxK = Math.max(...perRows.map(r => r.k))
  const subset = perRows.filter(r => r.k === maxK)

  const byAttack = {}
  for (const row of subset) {
    if (!byAttack[row.attack]) byAttack[row.attack] = { low_n: row.low_n || row.low_n === 'True' }
    const mean = perRows
      .filter(r => r.attack === row.attack && r.model === row.model)
      .reduce((s, r) => s + parseFloat(r.recall_5), 0) /
      perRows.filter(r => r.attack === row.attack && r.model === row.model).length
    byAttack[row.attack][row.model] = mean
  }

  return Object.entries(byAttack).map(([attack, vals]) => ({
    attack: vals.low_n ? `${attack}*` : attack,
    classical: vals.classical ?? 0,
    quantum:   vals.quantum   ?? 0,
    low_n: vals.low_n,
  }))
}
