import { useState, useRef } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Cell
} from 'recharts'
import { API_BASE } from '../config'

const MODELS = ['classical', 'quantum', 'both']
const FPRS   = [{ label: '1% target FPR', value: 0.01 }, { label: '5% target FPR', value: 0.05 }]

export default function Detect() {
  const [model,      setModel]      = useState('both')
  const [targetFpr,  setTargetFpr]  = useState(0.05)
  const [loading,    setLoading]    = useState(false)
  const [error,      setError]      = useState(null)
  const [result,     setResult]     = useState(null)
  const fileRef = useRef(null)

  const loadDemo = async () => {
    setError(null)
    setLoading(true)
    try {
      const resp = await axios.get(`${API_BASE}/demo-sample`)
      runPredict(null, resp.data)
    } catch (e) {
      setError(e.response?.data?.detail || e.message)
      setLoading(false)
    }
  }

  const runPredict = async (csvFile, demoData) => {
    setError(null)
    setLoading(true)
    setResult(null)
    try {
      let resp
      if (demoData) {
        const flows = demoData.rows.map(r => ({
          flow_duration:                 r['Flow Duration']                 ?? r['flow_duration'],
          total_fwd_packets:             r['Total Fwd Packets']             ?? r['total_fwd_packets'],
          total_backward_packets:        r['Total Backward Packets']        ?? r['total_backward_packets'],
          total_length_of_fwd_packets:   r['Total Length of Fwd Packets']   ?? r['total_length_of_fwd_packets'],
          total_length_of_bwd_packets:   r['Total Length of Bwd Packets']   ?? r['total_length_of_bwd_packets'],
          flow_bytes_s:                  r['Flow Bytes/s']                  ?? r['flow_bytes_s'],
          flow_packets_s:                r['Flow Packets/s']                ?? r['flow_packets_s'],
          packet_length_mean:            r['Packet Length Mean']             ?? r['packet_length_mean'],
          packet_length_std:             r['Packet Length Std']              ?? r['packet_length_std'],
          flow_iat_mean:                 r['Flow IAT Mean']                  ?? r['flow_iat_mean'],
          flow_iat_std:                  r['Flow IAT Std']                   ?? r['flow_iat_std'],
        }))
        resp = await axios.post(`${API_BASE}/predict`, {
          flows, model, target_fpr: targetFpr
        })
      } else {
        const fd = new FormData()
        fd.append('file', csvFile)
        resp = await axios.post(
          `${API_BASE}/predict/csv?model=${model}&target_fpr=${targetFpr}`,
          fd, { headers: { 'Content-Type': 'multipart/form-data' } }
        )
      }
      setResult(resp.data)
    } catch (e) {
      setError(e.response?.data?.detail || e.message)
    } finally {
      setLoading(false)
    }
  }

  const onFileChange = (e) => {
    const f = e.target.files[0]
    if (f) runPredict(f, null)
  }

  // Build histogram data per model from results
  const histData = result
    ? buildHistogram(result.results)
    : null

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Anomaly Detection</h1>
        <p className="text-sm text-gray-500 mt-1">
          Upload a CSV of network flows or use the demo sample to detect anomalies.
        </p>
      </div>

      {/* Simulator banner */}
      <div className="bg-amber-50 border border-amber-300 rounded-lg px-4 py-3 text-sm text-amber-800">
        ⚠️ <strong>Quantum model runs on a Qiskit Statevector simulator.</strong>{' '}
        No quantum advantage or speedup is claimed.
      </div>

      {/* Controls */}
      <div className="bg-white rounded-xl border border-gray-200 p-5 flex flex-wrap gap-4 items-end">
        <div>
          <label className="block text-xs font-medium text-gray-600 mb-1">Model</label>
          <select
            value={model}
            onChange={e => setModel(e.target.value)}
            className="border border-gray-300 rounded px-3 py-1.5 text-sm"
          >
            {MODELS.map(m => <option key={m} value={m}>{m}</option>)}
          </select>
        </div>
        <div>
          <label className="block text-xs font-medium text-gray-600 mb-1">Target FPR</label>
          <select
            value={targetFpr}
            onChange={e => setTargetFpr(parseFloat(e.target.value))}
            className="border border-gray-300 rounded px-3 py-1.5 text-sm"
          >
            {FPRS.map(f => <option key={f.value} value={f.value}>{f.label}</option>)}
          </select>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => fileRef.current?.click()}
            disabled={loading}
            className="px-4 py-2 bg-violet-700 text-white rounded text-sm font-medium hover:bg-violet-800 disabled:opacity-50"
          >
            Upload CSV
          </button>
          <button
            onClick={loadDemo}
            disabled={loading}
            className="px-4 py-2 bg-gray-100 text-gray-700 rounded text-sm font-medium hover:bg-gray-200 disabled:opacity-50"
          >
            Use Demo Sample
          </button>
          <input ref={fileRef} type="file" accept=".csv" className="hidden" onChange={onFileChange} />
        </div>
      </div>

      {loading && (
        <div className="text-center text-gray-500 py-10 text-sm">
          Running inference… (quantum model may take a moment)
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-300 rounded-lg px-4 py-3 text-sm text-red-700">
          {typeof error === 'string' ? error : JSON.stringify(error)}
        </div>
      )}

      {result && !loading && (
        <div className="space-y-6">
          {/* Summary cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <Card label="Rows Scored"   value={result.summary.rows_scored} />
            <Card label="Flagged"       value={result.summary.flagged_count} color="red" />
            <Card label="Flagged Rate"  value={`${(result.summary.flagged_share * 100).toFixed(1)}%`} />
            <Card label="Rows Dropped"  value={result.rows_dropped} />
          </div>

          {result.rows_dropped > 0 && (
            <div className="text-xs text-gray-500">
              {result.drop_reasons.join(' | ')}
            </div>
          )}

          {/* Histogram */}
          {histData && histData.length > 0 && (
            <div className="bg-white rounded-xl border border-gray-200 p-5">
              <h2 className="font-semibold text-gray-700 mb-3 text-sm">Score Distribution</h2>
              <div className="flex gap-6 flex-wrap">
                {Object.entries(histData).map(([mdl, { bins, threshold }]) => (
                  <div key={mdl} className="flex-1 min-w-[260px]">
                    <p className="text-xs text-gray-500 mb-2 capitalize">{mdl} model</p>
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart data={bins}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="x" tick={{ fontSize: 10 }} />
                        <YAxis tick={{ fontSize: 10 }} />
                        <Tooltip />
                        <Bar dataKey="count" fill={mdl === 'quantum' ? '#7c3aed' : '#2563eb'} />
                        <ReferenceLine x={threshold} stroke="#dc2626" strokeDasharray="4 2" label={{ value: 'threshold', fontSize: 9 }} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Results table */}
          <div className="bg-white rounded-xl border border-gray-200 overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 text-xs text-gray-500 uppercase tracking-wide">
                <tr>
                  <th className="px-4 py-2 text-left">Row</th>
                  <th className="px-4 py-2 text-left">Model</th>
                  <th className="px-4 py-2 text-right">Score</th>
                  <th className="px-4 py-2 text-right">Threshold</th>
                  <th className="px-4 py-2 text-center">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {result.results.slice(0, 200).map((r, i) => (
                  <tr key={i} className={r.flagged ? 'bg-red-50' : ''}>
                    <td className="px-4 py-1.5 text-gray-600">{r.row_index}</td>
                    <td className="px-4 py-1.5 capitalize">{r.model}</td>
                    <td className="px-4 py-1.5 text-right font-mono">{r.score.toFixed(4)}</td>
                    <td className="px-4 py-1.5 text-right font-mono text-gray-400">{r.threshold.toFixed(4)}</td>
                    <td className="px-4 py-1.5 text-center">
                      {r.flagged
                        ? <span className="px-2 py-0.5 rounded-full bg-red-100 text-red-700 text-xs font-medium">Flagged</span>
                        : <span className="px-2 py-0.5 rounded-full bg-green-100 text-green-700 text-xs font-medium">Normal</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {result.results.length > 200 && (
              <p className="text-xs text-gray-400 px-4 py-2">
                Showing first 200 of {result.results.length} results.
              </p>
            )}
          </div>

          <p className="text-xs text-gray-400 italic">{result.disclaimer}</p>
        </div>
      )}
    </div>
  )
}

function Card({ label, value, color }) {
  const valueClass = color === 'red' && Number(value) > 0
    ? 'text-red-600 font-bold'
    : 'text-gray-900 font-bold'
  return (
    <div className="bg-white rounded-lg border border-gray-200 px-4 py-3">
      <p className="text-xs text-gray-500">{label}</p>
      <p className={`text-xl ${valueClass}`}>{value}</p>
    </div>
  )
}

function buildHistogram(results) {
  const byModel = {}
  for (const r of results) {
    if (!byModel[r.model]) byModel[r.model] = { scores: [], threshold: r.threshold }
    byModel[r.model].scores.push(r.score)
  }
  const out = {}
  for (const [mdl, { scores, threshold }] of Object.entries(byModel)) {
    const min = Math.min(...scores)
    const max = Math.max(...scores)
    const BINS = 20
    const step = (max - min) / BINS || 1
    const bins = Array.from({ length: BINS }, (_, i) => ({
      x: (min + i * step).toFixed(2),
      count: 0,
    }))
    for (const s of scores) {
      const idx = Math.min(Math.floor((s - min) / step), BINS - 1)
      bins[idx].count++
    }
    out[mdl] = { bins, threshold }
  }
  return out
}
