export default function About() {
  return (
    <div className="max-w-3xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">About QCAD</h1>
        <p className="text-sm text-gray-500 mt-1">
          Quantum-Classical Cyber Anomaly Detection · Qiskit Fall Fest 2026 · IBM Quantum Hackathon
        </p>
      </div>

      {/* Simulator notice */}
      <div className="bg-amber-50 border border-amber-300 rounded-lg px-4 py-3 text-sm text-amber-800">
        ⚠️ <strong>The quantum model runs on a Qiskit Statevector simulator, not real quantum hardware.</strong>{' '}
        No quantum speedup or advantage is claimed. Results are reported honestly as measured.
      </div>

      <Section title="Problem Statement">
        <p>
          Traditional intrusion detection systems rely on known attack signatures,
          missing novel or zero-day threats. This project explores a <em>behavioural
          anomaly detection</em> approach: learn what normal (benign) network traffic
          looks like, then flag deviations — without ever seeing labelled attack traffic
          during training.
        </p>
      </Section>

      <Section title="Pipeline">
        <ol className="list-decimal list-inside space-y-1 text-sm text-gray-700">
          <li>Load CICIDS2017 flow CSVs; drop Inf/NaN/duplicates; strip header whitespace.</li>
          <li>Apply <code>log1p</code> (negatives clipped to 0).</li>
          <li>Fit a <code>StandardScaler</code> on benign training data only.</li>
          <li>Drop one feature from any highly-correlated pair (|r| &gt; 0.95).</li>
          <li>Fit PCA to k components (k = 4, 6, or 8).</li>
          <li>Fit a <code>MinMaxScaler</code> mapping to [0, π] (with clipping at inference).</li>
          <li>Fit the kernel model (see below) on the transformed benign training set.</li>
          <li>Threshold = (1 – target FPR) quantile of benign validation scores.</li>
        </ol>
        <p className="text-xs text-gray-500 mt-2">
          All preprocessing steps are fitted exclusively on benign Monday flows.
          No attack data is ever seen during training.
        </p>
      </Section>

      <Section title="Models">
        <div className="space-y-4">
          <div>
            <h3 className="font-semibold text-blue-700 text-sm">Classical Model</h3>
            <p className="text-sm text-gray-700">
              One-Class SVM with an RBF kernel. The kernel is parameterised by
              γ, tuned by PR-AUC on a small labelled tuning set.
            </p>
          </div>
          <div>
            <h3 className="font-semibold text-violet-700 text-sm">Quantum Model (Simulator)</h3>
            <p className="text-sm text-gray-700">
              One-Class SVM with a <em>fidelity kernel</em>{' '}
              k(x,y) = |⟨φ(x)|φ(y)⟩|², where φ encodes features into the quantum
              state using Qiskit's <code>ZZFeatureMap</code> (linear entanglement,
              reps = 2). Statevectors are computed with <code>qiskit.quantum_info.Statevector</code>{' '}
              (exact Statevector simulation). A scale factor multiplied before encoding
              is the tunable hyperparameter, selected by PR-AUC on the tuning set.
            </p>
          </div>
        </div>
      </Section>

      <Section title="Dataset">
        <p className="text-sm text-gray-700">
          <strong>CICIDS2017</strong> (Canadian Institute for Cybersecurity).
        </p>
        <ul className="list-disc list-inside text-sm text-gray-700 mt-1 space-y-1">
          <li><strong>Monday:</strong> benign-only — used for training and validation.</li>
          <li><strong>Wednesday:</strong> benign + DoS attacks (GoldenEye, Hulk, Slowhttptest, Slowloris, Heartbleed) — test and tuning pool.</li>
          <li><strong>Friday afternoon:</strong> benign + PortScan — test and tuning pool.</li>
        </ul>
        <p className="text-xs text-gray-500 mt-2">
          The data files are not included in this repository.
          See <code>data/README.md</code> for download instructions.
          Citation: Sharafaldin, Lashkari, Ghorbani (ICISSP 2018).
        </p>
        <div className="mt-3 bg-yellow-50 border border-yellow-200 rounded p-3 text-xs text-yellow-800">
          <strong>Important:</strong> Hyperparameters (scale/gamma) were tuned on a small
          labelled sample drawn from the same attack families as the test set (different flows).
          Models were never trained on attack data. Performance may not generalise to
          unseen attack families or real-world traffic distributions.
        </div>
      </Section>

      <Section title="Limitations">
        <ul className="list-disc list-inside text-sm text-gray-700 space-y-1">
          <li>Quantum model: simulated, not run on real hardware. No quantum advantage observed.</li>
          <li>Training size is small (300–1 000 benign flows). Real deployments need much more data.</li>
          <li>Threshold drift: thresholds are set from Monday benign flows; test benign flows come from Wednesday/Friday — measured FPR may differ from target.</li>
          <li>Only DoS and PortScan attack families have been evaluated. Heartbleed has very few test samples (low-n).</li>
          <li>No real-time or streaming inference capability.</li>
        </ul>
      </Section>

      <Section title="References">
        <ul className="list-disc list-inside text-sm text-gray-700 space-y-1">
          <li>Sharafaldin, I., Lashkari, A.H., Ghorbani, A.A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. ICISSP 2018.</li>
          <li>Qiskit Contributors (2023). Qiskit: An Open-source Framework for Quantum Computing. <a href="https://doi.org/10.5281/zenodo.2573505" className="text-blue-600 underline" target="_blank" rel="noreferrer">DOI</a>.</li>
          <li>Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, pp. 2825–2830.</li>
        </ul>
      </Section>
    </div>
  )
}

function Section({ title, children }) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5 space-y-3">
      <h2 className="font-semibold text-gray-900">{title}</h2>
      {children}
    </div>
  )
}
