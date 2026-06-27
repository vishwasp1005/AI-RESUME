import { useState, useRef } from 'react';
import './App.css';

const API_URL         = 'http://localhost:8000/api/v1/analyze';
const COMPARE_API_URL = 'http://localhost:8000/api/v1/analyze-multiple';
const EXPORT_API_URL  = 'http://localhost:8000/api/v1/export-report'; // ← NEW

// ── Helpers ──────────────────────────────────────────────────────────────────
function getScoreColor(score) {
  if (score >= 80) return 'var(--green)';
  if (score >= 60) return 'var(--green)';
  if (score >= 40) return 'var(--orange)';
  return 'var(--red)';
}

function getMatchClass(level) {
  const map = { Excellent: 'match-excellent', Good: 'match-good', Average: 'match-average', Poor: 'match-poor' };
  return map[level] || 'match-poor';
}

function getMatchEmoji(level) {
  const map = { Excellent: '🏆', Good: '✅', Average: '⚡', Poor: '🔴' };
  return map[level] || '🔴';
}

function getProgressColor(score) {
  if (score >= 70) return 'var(--green)';
  if (score >= 40) return 'var(--orange)';
  return 'var(--red)';
}

// ── NEW: Recruiter decision helpers ──────────────────────────────────────────
function getHireDecision(score) {
  if (score > 75)  return { label: 'Strong Hire', cls: 'hire-strong',   emoji: '🟢' };
  if (score >= 40) return { label: 'Consider',    cls: 'hire-consider', emoji: '🟡' };
  return               { label: 'Reject',         cls: 'hire-reject',   emoji: '🔴' };
}

// ── Sub-Components (ALL UNCHANGED) ───────────────────────────────────────────
function ProgressBar({ label, icon, value }) {
  const pct = Math.round(value);
  return (
    <div className="progress-item">
      <div className="progress-header">
        <span className="progress-label">{icon} {label}</span>
        <span className="progress-value" style={{ color: getProgressColor(pct) }}>{pct}%</span>
      </div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${pct}%`, background: getProgressColor(pct) }} />
      </div>
    </div>
  );
}

function ScoreRing({ score }) {
  const pct = Math.round(score);
  const color = getScoreColor(score);
  const circumference = 440;
  const offset = circumference - (circumference * pct) / 100;
  return (
    <div className="score-ring">
      <svg viewBox="0 0 160 160">
        <circle className="score-ring-bg" cx="80" cy="80" r="70" />
        <circle
          className="score-ring-fill"
          cx="80" cy="80" r="70"
          stroke={color}
          style={{ strokeDasharray: circumference, strokeDashoffset: offset, transition: 'stroke-dashoffset 1s ease' }}
        />
      </svg>
      <div className="score-ring-value">
        <span className="score-number" style={{ color }}>{pct}</span>
        <span className="score-max">/100</span>
      </div>
    </div>
  );
}

function BadgeList({ items, variant }) {
  if (!items || items.length === 0) return <p className="empty-state">None identified</p>;
  return (
    <div className="badge-list">
      {items.map((skill, i) => (
        <span key={i} className={`badge badge-${variant}`}>
          {variant === 'green' ? '✓' : '✗'} {skill}
        </span>
      ))}
    </div>
  );
}

function SuggestionList({ suggestions }) {
  if (!suggestions || suggestions.length === 0) return <p className="empty-state">No suggestions available.</p>;
  return (
    <ul className="suggestion-list">
      {suggestions.map((s, i) => (
        <li key={i} className="suggestion-item">
          <span className="suggestion-icon">💡</span>
          <span>{s}</span>
        </li>
      ))}
    </ul>
  );
}

// ── NEW: AI Comparison Insights card ─────────────────────────────────────────
function ComparisonInsights({ insight }) {
  if (!insight || !Object.values(insight).some(v => v)) return null;
  const sections = [
    { emoji: '🥇', label: 'Why Top Candidate Wins',  key: 'best_candidate_reason' },
    { emoji: '⚖️', label: 'Strengths Comparison',    key: 'strengths_comparison'  },
    { emoji: '⚠️', label: 'Weaknesses Comparison',   key: 'weaknesses_comparison' },
    { emoji: '🎯', label: 'Final Recommendation',    key: 'final_recommendation'  },
  ];
  return (
    <div className="card insight-card">
      <div className="card-title">🤖 AI Comparison Insights</div>
      <div className="insight-grid">
        {sections.map(({ emoji, label, key }) =>
          insight[key] ? (
            <div key={key} className="insight-block">
              <div className="insight-block-label">{emoji} {label}</div>
              <div className="insight-block-text">{insight[key]}</div>
            </div>
          ) : null
        )}
      </div>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  // ── Existing single-resume state (UNCHANGED) ─────────────────────────────
  const [jd, setJd]             = useState('');
  const [file, setFile]         = useState(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading]   = useState(false);
  const [result, setResult]     = useState(null);
  const [error, setError]       = useState(null);
  const fileInputRef            = useRef(null);

  // ── Compare state (from previous update) ─────────────────────────────────
  const [mode, setMode]                     = useState('single');
  const [files, setFiles]                   = useState([]);
  const [compareResult, setCompareResult]   = useState(null);
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareError, setCompareError]     = useState(null);
  const multiFileInputRef                   = useRef(null);

  // ── Comparison insight + export state ────────────────────────────────────
  const [comparisonInsight, setComparisonInsight] = useState(null);
  const [exportLoading, setExportLoading]         = useState(false);
  const [exportError, setExportError]             = useState(null);

  // ── JD expansion state ────────────────────────────────────────────────────
  const [interpretedRole, setInterpretedRole] = useState(null);
  const [jdExpanded, setJdExpanded]           = useState(false);

  // ── REQUEST ISOLATION — abort controller + monotonic request ID ───────────
  // Prevents stale responses from a previous upload overwriting fresh results.
  const singleAbortRef  = useRef(null);   // AbortController for /analyze
  const compareAbortRef = useRef(null);   // AbortController for /analyze-multiple
  const singleReqId     = useRef(0);      // increments each single-mode submit
  const compareReqId    = useRef(0);      // increments each compare-mode submit

  // ── Helper: wipe ALL analysis state immediately ───────────────────────────
  // Called the instant the user selects a new file so old cards never linger.
  const _clearSingleResults = () => {
    setResult(null);
    setError(null);
    setInterpretedRole(null);
    setJdExpanded(false);
  };
  const _clearCompareResults = () => {
    setCompareResult(null);
    setCompareError(null);
    setComparisonInsight(null);
    setExportError(null);
    setInterpretedRole(null);
    setJdExpanded(false);
  };

  // ── Existing single-resume handlers (state-isolated) ─────────────────────
  const handleFileChange = (e) => {
    const f = e.target.files?.[0];
    if (f && f.type === 'application/pdf') {
      // Abort any in-flight request immediately
      if (singleAbortRef.current) singleAbortRef.current.abort();
      setFile(f);
      _clearSingleResults();   // ← wipe stale results the moment new file chosen
    } else if (f) {
      setError('Only PDF files are supported.');
    }
  };

  const handleDrop = (e) => {
    e.preventDefault(); setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f && f.type === 'application/pdf') {
      if (singleAbortRef.current) singleAbortRef.current.abort();
      setFile(f);
      _clearSingleResults();   // ← wipe stale results on drag-and-drop too
    } else if (f) {
      setError('Only PDF files are supported.');
    }
  };

  const handleAnalyze = async () => {
    if (!file)       return setError('Please upload a PDF resume.');
    if (!jd.trim())  return setError('Please enter a job description.');

    // Cancel any previous in-flight request so its response can never land
    if (singleAbortRef.current) singleAbortRef.current.abort();
    const controller = new AbortController();
    singleAbortRef.current = controller;

    // Monotonic ID — only the response matching the LATEST submit is accepted
    const myReqId = ++singleReqId.current;

    setLoading(true); setError(null); setResult(null);
    setInterpretedRole(null); setJdExpanded(false);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('jd', jd.trim());

      const res  = await fetch(API_URL, {
        method: 'POST',
        body:   formData,
        signal: controller.signal,   // ← abort signal wired in
      });

      // Guard: if a newer request was fired while this one was in-flight, drop result
      if (singleReqId.current !== myReqId) return;

      const json = await res.json();
      if (json.status === 'success' && json.data) {
        setResult(json.data);
        setInterpretedRole(json.interpreted_role || null);
        setJdExpanded(json.jd_expanded || false);
      } else {
        setError(json.error || 'Analysis failed. Please try again.');
      }
    } catch (err) {
      if (err.name === 'AbortError') return;   // silently discard aborted request
      setError('Could not connect to the backend. Make sure the server is running on port 8000.');
    } finally {
      // Only clear loading spinner if THIS request is still the latest
      if (singleReqId.current === myReqId) setLoading(false);
    }
  };

  // ── Compare handlers (from previous update, extended) ────────────────────
  const handleModeSwitch = (newMode) => {
    setMode(newMode);
    setResult(null); setError(null);
    setCompareResult(null); setCompareError(null);
    setComparisonInsight(null); setExportError(null);
    setInterpretedRole(null);  setJdExpanded(false);   // ← also clear expansion state
  };

  const handleMultiFileChange = (e) => {
    const selected = Array.from(e.target.files || []).filter(f => f.type === 'application/pdf');
    if (selected.length === 0) { setCompareError('Only PDF files are supported.'); return; }
    // Abort any in-flight compare request and wipe stale results immediately
    if (compareAbortRef.current) compareAbortRef.current.abort();
    setFiles(selected);
    _clearCompareResults();   // ← wipe stale compare cards the moment new files chosen
  };

  const handleCompareAnalyze = async () => {
    if (files.length < 2) return setCompareError('Please upload at least 2 PDF resumes to compare.');
    if (!jd.trim())        return setCompareError('Please enter a job description.');

    // Cancel any previous in-flight compare request
    if (compareAbortRef.current) compareAbortRef.current.abort();
    const controller = new AbortController();
    compareAbortRef.current = controller;

    // Monotonic ID — only latest compare response is accepted
    const myReqId = ++compareReqId.current;

    setCompareLoading(true);
    _clearCompareResults();   // ← full wipe before every new compare run

    try {
      const formData = new FormData();
      files.forEach(f => formData.append('files', f));
      formData.append('jd', jd.trim());

      const res  = await fetch(COMPARE_API_URL, {
        method: 'POST',
        body:   formData,
        signal: controller.signal,   // ← abort signal wired in
      });

      // Guard: if a newer compare was fired while this was in-flight, drop result
      if (compareReqId.current !== myReqId) return;

      const json = await res.json();
      if (json.status === 'success') {
        setCompareResult(json.data);
        setComparisonInsight(json.comparison_insight || null);
        setInterpretedRole(json.interpreted_role || null);
        setJdExpanded(json.jd_expanded || false);
      } else {
        setCompareError(json.error || 'Comparison failed. Please try again.');
      }
    } catch (err) {
      if (err.name === 'AbortError') return;   // silently discard aborted request
      setCompareError('Could not connect to the backend. Make sure the server is running on port 8000.');
    } finally {
      if (compareReqId.current === myReqId) setCompareLoading(false);
    }
  };

  // ── NEW: Export PDF handler ───────────────────────────────────────────────
  const handleExportReport = async () => {
    if (!compareResult || compareResult.length === 0) return;
    setExportLoading(true); setExportError(null);
    try {
      const res = await fetch(EXPORT_API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          results:            compareResult,
          jd:                 jd.trim(),
          comparison_insight: comparisonInsight || {},
        }),
      });
      if (!res.ok) {
        const json = await res.json().catch(() => ({}));
        setExportError(json.error || 'Export failed. Please try again.');
        return;
      }
      const blob = await res.blob();
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement('a');
      a.href     = url;
      a.download = 'resume_comparison_report.pdf';
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      setExportError('Could not connect to the backend for export.');
    } finally {
      setExportLoading(false);
    }
  };

  // ── Derived ───────────────────────────────────────────────────────────────
  const canAnalyze = file && jd.trim() && !loading;
  const canCompare = files.length >= 2 && jd.trim() && !compareLoading;

  return (
    <div className="app">
      {/* ── Header (UNCHANGED) ─────────────────────────────────────────── */}
      <header className="header">
        <div className="header-brand">
          <div className="header-icon">🎯</div>
          <span className="header-title">ResumeAI Scanner</span>
        </div>
        <span className="header-badge">AI Powered</span>
      </header>

      <main className="main">
        {/* ── Hero (UNCHANGED) ───────────────────────────────────────────── */}
        <section className="hero">
          <h1>AI Resume Screening System</h1>
          <p>Upload a resume and paste a job description to get an ATS score, skill analysis, and AI-powered insights instantly.</p>
        </section>

        {/* ── Mode Toggle ───────────────────────────────────────────────── */}
        <div className="mode-toggle">
          <button className={`mode-btn ${mode === 'single'  ? 'active' : ''}`} onClick={() => handleModeSwitch('single')}>
            📄 Single Resume
          </button>
          <button className={`mode-btn ${mode === 'compare' ? 'active' : ''}`} onClick={() => handleModeSwitch('compare')}>
            📊 Compare Resumes
          </button>
        </div>

        {/* ── Upload Section ─────────────────────────────────────────────── */}
        <section className="upload-section">
          <div className="upload-form">
            {/* JD textarea — shared (UNCHANGED) */}
            <div className="form-group">
              <label className="form-label">📋 Job Description</label>
              <textarea
                id="jd-input"
                className="form-textarea"
                placeholder="Paste the full job description here..."
                value={jd}
                onChange={(e) => { setJd(e.target.value); setInterpretedRole(null); setJdExpanded(false); }}
              />
              {/* ── NEW: auto-expansion label ───────────────────────────── */}
              {jdExpanded && interpretedRole && (
                <div className="jd-expanded-label">
                  ✨ Interpreted Role: <strong>{interpretedRole}</strong>
                  <span className="jd-expanded-tag">auto-expanded</span>
                </div>
              )}
            </div>

            {mode === 'single' ? (
              /* Existing single-file dropzone (UNCHANGED) */
              <div className="form-group">
                <label className="form-label">📄 Resume (PDF)</label>
                <div
                  id="resume-dropzone"
                  className={`dropzone ${dragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
                  onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <input ref={fileInputRef} type="file" accept=".pdf" onChange={handleFileChange} style={{ display: 'none' }} id="resume-file-input" />
                  {file ? (
                    <><span className="dropzone-icon">✅</span><span className="dropzone-filename">{file.name}</span><span className="dropzone-text" style={{ color: 'var(--green)' }}>PDF ready • Click to replace</span></>
                  ) : (
                    <><span className="dropzone-icon">☁️</span><span className="dropzone-text">Drag & drop PDF here</span><span className="dropzone-text" style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>or click to browse</span></>
                  )}
                </div>
              </div>
            ) : (
              /* Multi-file upload zone */
              <div className="form-group">
                <label className="form-label">📄 Resumes (PDF · select multiple)</label>
                <div className={`dropzone multi-dropzone ${files.length > 0 ? 'has-file' : ''}`} onClick={() => multiFileInputRef.current?.click()}>
                  <input ref={multiFileInputRef} type="file" accept=".pdf" multiple onChange={handleMultiFileChange} style={{ display: 'none' }} id="multi-resume-file-input" />
                  {files.length > 0 ? (
                    <>
                      <span className="dropzone-icon">✅</span>
                      <span className="dropzone-text" style={{ color: 'var(--green)', fontWeight: 600 }}>{files.length} PDF{files.length !== 1 ? 's' : ''} selected</span>
                      <div className="multi-file-list">
                        {files.map((f, i) => <span key={i} className="multi-file-chip">📄 {f.name}</span>)}
                      </div>
                      <span className="dropzone-text" style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Click to change selection</span>
                    </>
                  ) : (
                    <><span className="dropzone-icon">☁️</span><span className="dropzone-text">Click to select multiple PDFs</span><span className="dropzone-text" style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Minimum 2 resumes required</span></>
                  )}
                </div>
              </div>
            )}

            {/* Submit buttons */}
            {mode === 'single' ? (
              <button id="analyze-btn" className="btn-analyze" onClick={handleAnalyze} disabled={!canAnalyze}>
                {loading ? <><div className="spinner" /> Analyzing Resume...</> : <>🚀 Analyze Resume</>}
              </button>
            ) : (
              <button id="compare-btn" className="btn-analyze" onClick={handleCompareAnalyze} disabled={!canCompare}>
                {compareLoading ? <><div className="spinner" /> Comparing Resumes...</> : <>📊 Compare &amp; Rank Resumes</>}
              </button>
            )}
          </div>
        </section>

        {/* ── Errors ─────────────────────────────────────────────────────── */}
        {mode === 'single'  && error        && <div className="error-banner" id="error-banner">⚠️ {error}</div>}
        {mode === 'compare' && compareError  && <div className="error-banner" id="compare-error-banner">⚠️ {compareError}</div>}
        {mode === 'compare' && exportError   && <div className="error-banner">⚠️ Export error: {exportError}</div>}

        {/* ── Single-resume results (UNCHANGED) ─────────────────────────── */}
        {mode === 'single' && result && (
          <section className="results-grid" id="results-section">
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div className="card">
                <div className="card-title">🎯 Skill Analysis</div>
                <div className="skills-grid">
                  <div>
                    <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--green)', marginBottom: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                      ✅ Matched Skills ({result.matched_skills.length})
                    </div>
                    <BadgeList items={result.matched_skills} variant="green" />
                  </div>
                  <div>
                    <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--red)', marginBottom: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                      ✗ Missing Skills ({result.missing_skills.length})
                    </div>
                    <BadgeList items={result.missing_skills} variant="red" />
                  </div>
                </div>
              </div>
              <div className="card">
                <div className="card-title">💡 Improvement Suggestions</div>
                <SuggestionList suggestions={result.suggestions} />
              </div>
              <div className="card">
                <div className="card-title">
                  🤖 AI Insight
                  {result.domain && (
                    <span className="domain-chip" style={{ marginLeft: 'auto' }}>
                      🏷️ {result.domain.replace('_', ' / ')}
                    </span>
                  )}
                </div>
                <div className="ai-insight-box" id="ai-insight-box">{result.ai_insight || 'No AI insight available.'}</div>
              </div>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div className="score-card" style={{ '--score-glow': `${getScoreColor(result.ats_score)}40` }}>
                <ScoreRing score={result.ats_score} />
                <div className={`match-badge ${getMatchClass(result.match_level)}`}>{getMatchEmoji(result.match_level)} {result.match_level} Match</div>
                <div className="score-label" style={{ color: getScoreColor(result.ats_score) }}>ATS Score</div>
                <div className="score-filename">📄 {result.filename}</div>
                <div className="score-breakdown">
                  <div className="breakdown-title">Score Breakdown</div>
                  <ProgressBar label="Semantic Match" icon="🔍" value={result.section_scores.semantic} />
                  <ProgressBar label="Skills Match"   icon="🎯" value={result.section_scores.skills} />
                  <ProgressBar label="Experience"     icon="💼" value={result.section_scores.experience} />
                  <ProgressBar label="Education"      icon="🎓" value={result.section_scores.education} />
                </div>
              </div>
              <div className="card">
                <div className="card-title">📊 Quick Stats</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
                  {[
                    { label: 'ATS Score',     value: `${Math.round(result.ats_score)}%`, color: getScoreColor(result.ats_score) },
                    { label: 'Match Level',   value: result.match_level,                 color: getScoreColor(result.ats_score) },
                    { label: 'Matched Skills',value: result.matched_skills.length,       color: 'var(--green)' },
                    { label: 'Missing Skills',value: result.missing_skills.length,       color: result.missing_skills.length > 0 ? 'var(--red)' : 'var(--green)' },
                  ].map((stat, i) => (
                    <div key={i} style={{ background: 'var(--bg-secondary)', borderRadius: '10px', padding: '0.75rem', textAlign: 'center' }}>
                      <div style={{ fontSize: '1.3rem', fontWeight: 800, color: stat.color }}>{stat.value}</div>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.2rem', textTransform: 'uppercase', letterSpacing: '0.3px' }}>{stat.label}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </section>
        )}

        {/* ── Compare results ────────────────────────────────────────────── */}
        {mode === 'compare' && compareResult && compareResult.length > 0 && (
          <section className="compare-results" id="compare-results-section">

            {/* Ranking table card */}
            <div className="card">
              <div className="card-title">
                🏆 Ranking Results
                <span className="compare-count-chip">{compareResult.length} resumes</span>

                {/* ── NEW: Export button ─────────────────────────────────── */}
                <button
                  className="btn-export"
                  onClick={handleExportReport}
                  disabled={exportLoading}
                  title="Download Full PDF Report"
                >
                  {exportLoading
                    ? <><div className="spinner spinner-sm" /> Exporting...</>
                    : <>📥 Download Full Report</>}
                </button>
              </div>

              <table className="compare-table">
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Resume Name</th>
                    <th>ATS Score</th>
                    <th>Match Level</th>
                    <th>Decision</th>{/* ← NEW column */}
                  </tr>
                </thead>
                <tbody>
                  {compareResult.map((r) => {
                    const decision = getHireDecision(r.ats_score); // ← NEW
                    return (
                      <tr key={r.rank} className={`compare-row ${r.rank === 1 ? 'top-candidate' : ''}`}>
                        <td className="rank-cell">
                          {r.rank === 1 ? '🥇' : r.rank === 2 ? '🥈' : r.rank === 3 ? '🥉' : `#${r.rank}`}
                        </td>
                        <td className="filename-cell">
                          {r.rank === 1 && <span className="top-pick-tag">TOP PICK</span>}
                          <span className="compare-filename">{r.filename}</span>
                        </td>
                        <td className="score-cell">
                          <span className="score-pill" style={{
                            color: getScoreColor(r.ats_score),
                            borderColor: getScoreColor(r.ats_score),
                            background: `${getScoreColor(r.ats_score)}18`,
                          }}>
                            {r.ats_score}%
                          </span>
                        </td>
                        <td className="level-cell">
                          <span className={`match-badge ${getMatchClass(r.match_level)}`}>
                            {getMatchEmoji(r.match_level)} {r.match_level}
                          </span>
                        </td>
                        {/* ── NEW: Recruiter Decision badge ──────────────── */}
                        <td className="decision-cell">
                          <span className={`hire-badge ${decision.cls}`}>
                            {decision.emoji} {decision.label}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* ── NEW: AI Comparison Insights card ─────────────────────── */}
            <ComparisonInsights insight={comparisonInsight} />

          </section>
        )}
      </main>

      {/* ── Footer (UNCHANGED) ─────────────────────────────────────────── */}
      <footer className="footer">
        ResumeAI Scanner · Powered by FastAPI + Sentence Transformers + Ollama
      </footer>
    </div>
  );
}