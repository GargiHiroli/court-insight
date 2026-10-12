import React, { useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Scale, FileText, UploadCloud, BrainCircuit, ShieldCheck, Sparkles, AlertCircle, ChevronRight, Search, BarChart3, Fingerprint, Download } from 'lucide-react';
import './style.css';
const API = import.meta.env.VITE_API_BASE || 'http://localhost:8000';
const SAMPLE = `IN THE HIGH COURT — ORDER\nThe petitioner submitted documents in support of the application. The court considered the record and relevant evidence. The application is partly allowed. The parties are directed to file additional submissions. The matter remains pending for a further hearing. The court granted interim relief to the applicant pending final consideration.`;
const NAV = [['Overview',BarChart3],['Document Analysis',FileText],['AI Models',BrainCircuit]];
const TRAINING_DEFAULTS = {
  Good: ['The court granted the application and awarded relief to the applicant.', 'The appeal was allowed after the court considered the evidence.'].join('\n'),
  Better: ['The matter was adjourned for further hearing and the parties were directed to file submissions.', 'The court partly allowed the claim and requested additional evidence.'].join('\n'),
  Worst: ['The petition was dismissed and the application was refused for lack of evidence.', 'The appeal was rejected and the court imposed penalties.'].join('\n')
};
function App() {
  const [text,setText]=useState(''), [file,setFile]=useState(null), [embeddings,setEmbeddings]=useState(false), [busy,setBusy]=useState(false), [result,setResult]=useState(null), [error,setError]=useState(''), [tab,setTab]=useState('upload'), [page,setPage]=useState('Document Analysis'), [trainingTexts,setTrainingTexts]=useState(TRAINING_DEFAULTS), [trainingStatus,setTrainingStatus]=useState({type:'', message:''});
  async function run() {
    setBusy(true);setError('');setResult(null);
    try {
      let response;
      if(tab==='upload') {
        if(!file) throw new Error('Please select a PDF, DOCX or TXT document.');
        const body=new FormData();body.append('file',file);body.append('use_embeddings',String(embeddings));
        response=await fetch(`${API}/api/analyze/file`,{method:'POST',body});
      } else {
        if(text.trim().length<40) throw new Error('Paste at least 40 characters.');
        response=await fetch(`${API}/api/analyze/text`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,use_embeddings:embeddings})});
      }
      const json=await response.json();if(!response.ok) throw new Error(typeof json.detail==='string'?json.detail:'Analysis failed.');setResult(json);
    } catch(e) {setError(e.message || 'Could not contact API. Is the backend running?');} finally {setBusy(false);}
  }
  const scoreColor={Good:'#24b78b',Better:'#e5ad4a',Worst:'#ee6d71'};
  function download(){const blob=new Blob([JSON.stringify(result,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='court-insight-report.json';a.click();URL.revokeObjectURL(a.href);}
  async function trainModel() {
    const examples = Object.entries(trainingTexts)
      .flatMap(([label, value]) => value.split(/\n+/)
        .map(text => text.trim())
        .filter(Boolean)
        .map(text => ({ text, label })))
      .filter(({ text }) => text.length > 0);

    if (examples.length < 9 || Object.keys(trainingTexts).some(label => trainingTexts[label].split(/\n+/).map(text => text.trim()).filter(Boolean).length < 3)) {
      setTrainingStatus({ type: 'error', message: 'Provide at least 3 examples in each category for a total of 9 training examples.' });
      return;
    }

    try {
      setTrainingStatus({ type: '', message: 'Training model…' });
      const response = await fetch(`${API}/api/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ examples })
      });
      const json = await response.json();
      if (!response.ok) {
        throw new Error(typeof json.detail === 'string' ? json.detail : JSON.stringify(json.detail));
      }
      setTrainingStatus({ type: 'success', message: `Model trained with ${json.examples} examples.` });
    } catch (e) {
      setTrainingStatus({ type: 'error', message: e.message || 'Could not train the model.' });
    }
  }
  return <div className="shell"><aside className="sidebar"><div className="brand"><div className="brandicon"><Scale size={22}/></div><div><strong>CourtInsight</strong><span>LEGAL DOCUMENT AI</span></div></div><div className="section-label">WORKSPACE</div><nav>{NAV.map(([name,Icon])=><button key={name} onClick={()=>{setPage(name); if(name==='Overview')window.scrollTo({top:0,behavior:'smooth'});if(name==='AI Models')document.getElementById('models').scrollIntoView({behavior:'smooth'});}} className={page===name?'active':''}><Icon size={18}/>{name}{page===name&&<ChevronRight className="chev" size={14}/>}</button>)}</nav><div className="sidebottom"><div className="security"><ShieldCheck size={17}/><span>Private by design</span></div><small>Files are analyzed in memory and not stored by this prototype. Do not upload confidential case material to an untrusted server.</small></div></aside>
  <main><header className="topbar"><div><span className="breadcrumb">Workspace</span><ChevronRight size={14}/><span>Document Intelligence</span></div><span className="version"><span className="dot"/> Prototype v0.1</span></header><div className="content"><div className="intro"><div className="eyebrow"><Sparkles size={15}/> AI-POWERED DOCUMENT INTELLIGENCE</div><h1>Understand the language<br/><em>behind every case.</em></h1><p>Analyze court documents with NLP, TF-IDF and optional semantic embeddings. Explore linguistic signals through a transparent, explainable dashboard.</p></div>
  <div className="metrics"><div className="metric"><div className="metric-icon violet"><FileText size={21}/></div><span>DOCUMENT FORMATS</span><strong>PDF, DOCX, TXT</strong><small>Up to 10 MB per upload</small></div><div className="metric"><div className="metric-icon teal"><BrainCircuit size={21}/></div><span>ANALYSIS ENGINES</span><strong>3 approaches</strong><small>Lexical · TF-IDF ML · Embeddings</small></div><div className="metric"><div className="metric-icon gold"><Fingerprint size={21}/></div><span>EXPLAINABILITY</span><strong>Key signals</strong><small>Keywords & sentence excerpts</small></div></div>
  <section className="panel"><div className="panel-head"><div><div className="pill">01 / ANALYZE</div><h2>New document analysis</h2><p>Upload a judgment, order, or petition, or paste its text.</p></div><div className="outline-icon"><Search size={21}/></div></div><div className="tabs"><button className={tab==='upload'?'selected':''} onClick={()=>setTab('upload')}><UploadCloud size={17}/> Upload document</button><button className={tab==='paste'?'selected':''} onClick={()=>setTab('paste')}><FileText size={17}/> Paste text</button></div>{tab==='upload'?<label className="drop"><input type="file" accept=".pdf,.docx,.txt" onChange={e=>setFile(e.target.files?.[0]||null)}/><div className="upload-circle"><UploadCloud size={28}/></div><strong>{file?file.name:'Drop your court document here'}</strong><span>{file?'Click to select a different file':'or click to browse your files'}</span><small>PDF, DOCX, TXT · Maximum 10 MB</small></label>:<div className="pastebox"><textarea placeholder="Paste the text of a court order, judgment or petition…" value={text} onChange={e=>setText(e.target.value)}/><div className="example"><span>{text.length} characters</span><button onClick={()=>setText(SAMPLE)}>Insert sample order →</button></div></div>}<div className="controlrow"><label className="toggle"><input type="checkbox" checked={embeddings} onChange={e=>setEmbeddings(e.target.checked)}/><span className="switch"/><span><strong>Deep-learning semantic embeddings</strong><small>MiniLM · model downloaded on first use</small></span></label><button className="analyze" disabled={busy} onClick={run}>{busy?'Analyzing…':'Analyze document'}<ChevronRight size={18}/></button></div>{error&&<p className="error"><AlertCircle size={17}/>{error}</p>}</section>
  {result&&<section className="results"><div className="results-header"><div><div className="pill">02 / RESULTS</div><h2>Analysis report</h2><p>{result.filename||'Pasted text'} · {result.words.toLocaleString()} words · {result.methods.join(' + ')}</p></div><button className="export" onClick={download}><Download size={16}/> Export JSON</button></div><div className="results-grid"><div className="panel summary"><div className="muted">DOMINANT TEXTUAL SIGNAL</div><strong style={{color:scoreColor[result.label]}}>{result.label}</strong><p>Relative model score: {result.relative_score}%</p><span className="subtle">Uncertainty: {result.uncertainty}</span><div className="disclaimer"><AlertCircle size={16}/><span>{result.disclaimer}</span></div></div><div className="panel bars"><div className="muted">RELATIVE CATEGORY SCORES</div>{result.ranges.map(i=><div className="barrow" key={i.label}><div><strong>{i.label}</strong><span>{i.score}%</span></div><div className="track"><div style={{width:`${i.score}%`,background:scoreColor[i.label]}}/></div></div>)}<small>Scores are normalized comparison values, not calibrated probabilities.</small></div></div><div className="results-grid second"><div className="panel keywords"><h3>TF-IDF keywords</h3><p>Terms that stand out in this document.</p><div className="chips">{result.keywords.map(k=><span title={`TF-IDF weight: ${k.weight}`} key={k.term}>{k.term}</span>)}</div></div><div className="panel keywords"><h3>Evidence excerpts</h3><p>Sentences containing indicative legal language.</p>{result.evidence.length?result.evidence.map((e,i)=><div key={i} className="quote">“{e.excerpt}”<div>{e.signals.map(s=><span key={s}>{s}</span>)}</div></div>):<p>No matching indicator terms found.</p>}</div></div>{result.warning&&<p className="error"><AlertCircle size={17}/>{result.warning}</p>}</section>}
  <section className="method" id="models"><div className="pill">HOW IT WORKS</div><h2>Built with interpretable NLP</h2><div className="steps"><div><span>01</span><h3>Extract</h3><p>Read text from court PDF, DOCX or TXT documents.</p></div><div><span>02</span><h3>Represent</h3><p>Compute TF-IDF terms and optional MiniLM embeddings.</p></div><div><span>03</span><h3>Classify</h3><p>Combine lexical cues, a trained ML model if supplied, and semantic signals.</p></div><div><span>04</span><h3>Review</h3><p>Inspect relative scores, keywords, and evidence sentences.</p></div></div><div className="training-panel"><div className="training-header"><div><div className="pill">03 / AI MODELS</div><h3>Train the supervised classifier</h3></div></div><div className="training-grid"><div className="training-card"><label>Good examples<textarea value={trainingTexts.Good} onChange={e=>setTrainingTexts(current=>({...current, Good:e.target.value}))} placeholder="One example per line" /></label></div><div className="training-card"><label>Better examples<textarea value={trainingTexts.Better} onChange={e=>setTrainingTexts(current=>({...current, Better:e.target.value}))} placeholder="One example per line" /></label></div><div className="training-card"><label>Worst examples<textarea value={trainingTexts.Worst} onChange={e=>setTrainingTexts(current=>({...current, Worst:e.target.value}))} placeholder="One example per line" /></label></div></div><div className="controlrow training-actions"><button className="analyze" onClick={trainModel}>Train model<ChevronRight size={18}/></button></div>{trainingStatus.message && <p className={`training-status ${trainingStatus.type}`}>{trainingStatus.message}</p>}</div></section><footer><Scale size={16}/> CourtInsight is a research demonstration — not legal advice or an outcome prediction.</footer></div></main></div>
}

createRoot(document.getElementById('root')).render(<App/>);
