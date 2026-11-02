from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def build_metrics():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="diagnosis")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    models = {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)),
        "Support vector machine": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
        "Random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=42, class_weight="balanced"),
    }
    scores, fitted, diagnostics = [], {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        malignant_recall = recall_score(y_test, pred, pos_label=0)
        scores.append({"model": name, "accuracy": accuracy_score(y_test, pred),
                       "malignant_recall": malignant_recall,
                       "precision": precision_score(y_test, pred, pos_label=0),
                       "f1": f1_score(y_test, pred, pos_label=0)})
        fitted[name] = model
        proba = model.predict_proba(X_test)[:, list(model.classes_).index(0)]
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        fpr, tpr, _ = roc_curve(y_test.eq(0), proba)
        if name == "Random forest":
            importance = model.feature_importances_
            importance_label = "Forest importance"
        elif name == "Support vector machine":
            importance = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=5, random_state=42, n_jobs=-1).importances_mean
            importance = np.maximum(importance, 0)
            importance_label = "Held-out ROC AUC permutation importance"
        else:
            importance = np.abs(model[-1].coef_[0])
            importance_label = "Absolute standardized coefficient" if name == "Logistic regression" else "Model importance"
        features = sorted(zip(data.feature_names, importance), key=lambda pair: pair[1], reverse=True)[:8]
        diagnostics[name] = {
            "cm": cm.tolist(), "fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "auc": float(roc_auc_score(y_test.eq(0), proba)),
            "features": [[str(n), float(v)] for n, v in features],
            "importance_label": importance_label,
        }
    winner = max(scores, key=lambda item: (item["malignant_recall"], item["f1"]))
    result = {"scores": scores, "winner": winner["model"], "diagnostics": diagnostics,
              "n": int(len(y)), "test_n": int(len(y_test)), "malignant": int((y == 0).sum()), "benign": int((y == 1).sum())}
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame(scores).to_csv(OUT / "model_comparison.csv", index=False)
    return result


def dashboard(d):
    scores_json = json.dumps(d["scores"])
    diagnostics_json = json.dumps(d["diagnostics"])
    html = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Breast cancer model evaluation</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script><style>
:root{color-scheme:dark;--bg:#151918;--surface:#1d2321;--line:#343d38;--text:#e5eae6;--muted:#98a39c;--lime:#b7dc55;--green:#668e58;--grid:#303934}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:13px Inter,Arial,sans-serif}.wrap{max-width:1500px;margin:auto;padding:18px 22px}.header{display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line);padding:2px 0 13px;margin-bottom:13px}.eyebrow,.meta{font-size:11px;color:var(--muted)}.title{font-size:19px;font-weight:600;margin:5px 0 0}.tag{border:1px solid #4a5836;color:var(--lime);padding:6px 9px;font-size:10px;white-space:nowrap}.layout{display:grid;grid-template-columns:202px minmax(0,1fr);gap:14px}.rail{position:sticky;top:12px;align-self:start;background:var(--surface);border:1px solid var(--line);padding:14px}.rail h2,.panel h2{font-size:13px;font-weight:600;margin:0 0 10px}.fact{border-top:1px solid var(--line);padding:9px 0}.fact-label,.label{font-size:10px;color:var(--muted);margin-bottom:5px}.fact-value{font-size:12px;line-height:1.4}.rail-note{font-size:10px;color:var(--muted);line-height:1.5;margin-top:9px}.main{min-width:0}.meta{margin:0 0 9px}.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px;margin-bottom:10px}.card,.panel{background:var(--surface);border:1px solid var(--line)}.card{padding:10px 12px;min-height:70px}.value{font-size:20px;font-weight:600;color:var(--lime);line-height:1.2}.hint{font-size:10px;color:var(--muted);margin-top:4px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}.panel{padding:11px 12px;min-width:0;min-height:283px}.panel p{font-size:10px;color:var(--muted);margin:-5px 0 0}.chart{height:235px}.selectrow{display:flex;align-items:center;gap:8px;margin:0 0 9px}.selectrow label{font-size:10px;color:var(--muted)}select{background:#252d29;color:var(--text);border:1px solid #4b574f;padding:6px 9px;font:11px Arial;min-width:180px}select:focus{outline:1px solid var(--lime)}.foot{border-top:1px solid var(--line);margin-top:11px;padding-top:8px;color:var(--muted);font-size:10px;line-height:1.55}.foot strong{color:var(--text)}@media(max-width:850px){.wrap{padding:12px}.layout{grid-template-columns:1fr}.rail{position:static;display:grid;grid-template-columns:repeat(2,1fr);gap:0 12px}.rail h2,.rail-note{grid-column:1/-1}.cards{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}}@media(max-width:500px){.header{align-items:flex-start;gap:8px;flex-direction:column}.cards{grid-template-columns:1fr 1fr}.panel{padding:9px}.selectrow{align-items:flex-start;flex-direction:column}}
</style></head><body><main class="wrap"><header class="header"><div><div class="eyebrow">MODEL EVALUATION&nbsp; / &nbsp;WISCONSIN DIAGNOSTIC DATASET</div><h1 class="title">Breast cancer classification</h1></div><div class="tag">Research benchmark · not clinical use</div></header><div class="layout"><aside class="rail"><h2>Dataset &amp; method</h2><div class="fact"><div class="fact-label">Source</div><div class="fact-value">Wisconsin Diagnostic Breast Cancer</div></div><div class="fact"><div class="fact-label">Observations</div><div class="fact-value">'''+f'''{d['n']:,} total · {d['malignant']} malignant · {d['benign']} benign'''+r'''</div></div><div class="fact"><div class="fact-label">Measurements</div><div class="fact-value">30 numeric nucleus features</div></div><div class="fact"><div class="fact-label">Evaluation</div><div class="fact-value">Stratified 80/20 holdout</div></div><div class="fact"><div class="fact-label">Selection rule</div><div class="fact-value">Malignant recall, then F1</div></div><div class="rail-note">Features derive from digitized images; original medical images are not included.</div></aside><section class="main"><p class="meta">Three baseline classifiers · fixed random seed · test set: '''+f'''{d['test_n']} observations'''+r'''</p><div class="selectrow"><label for="modelSelect">Evaluated model</label><select id="modelSelect" aria-label="Select evaluated model"></select></div><section class="cards"><div class="card"><div class="label">Test accuracy</div><div class="value" id="kAccuracy"></div></div><div class="card"><div class="label">Malignant recall</div><div class="value" id="kRecall"></div><div class="hint">Positive class: malignant</div></div><div class="card"><div class="label">Malignant F1</div><div class="value" id="kF1"></div></div><div class="card"><div class="label">ROC AUC</div><div class="value" id="kAuc"></div></div></section><section class="grid"><div class="panel"><h2>Accuracy and malignant recall</h2><div id="models" class="chart"></div></div><div class="panel"><h2>Confusion matrix</h2><div id="cm" class="chart"></div></div><div class="panel"><h2>ROC curve</h2><div id="roc" class="chart"></div></div><div class="panel"><h2>Top eight feature measures</h2><p id="featureNote">Relative model-specific importance; not causal.</p><div id="features" class="chart"></div></div></section><div class="foot"><strong>Source:</strong> Breast Cancer Wisconsin (Diagnostic), distributed with scikit-learn. <strong>Method:</strong> one stratified 80/20 split with fixed seed. <strong>Important:</strong> Educational model comparison only; not a clinical diagnostic tool and must not guide patient care.</div></section></div></main><script>
const S='''+scores_json+r''';const D='''+diagnostics_json+r''';const chosen='''+json.dumps(d['winner'])+r''';const select=document.getElementById('modelSelect');S.forEach(s=>{let o=document.createElement('option');o.value=s.model;o.textContent=s.model+(s.model===chosen?' · selected':'');select.appendChild(o)});select.value=chosen;
const base={paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{family:'Arial,sans-serif',color:'#aab4ad',size:10},margin:{l:48,r:10,t:8,b:42},xaxis:{gridcolor:'#303934',zerolinecolor:'#414b44'},yaxis:{gridcolor:'#303934',zerolinecolor:'#414b44'},transition:{duration:380,easing:'cubic-in-out'}};const cfg={displayModeBar:false,responsive:true};
Plotly.newPlot('models',[{type:'bar',name:'Accuracy',x:S.map(x=>x.model),y:S.map(x=>x.accuracy),marker:{color:'#668e58'},hovertemplate:'%{x}<br>Accuracy %{y:.1%}<extra></extra>'},{type:'bar',name:'Malignant recall',x:S.map(x=>x.model),y:S.map(x=>x.malignant_recall),marker:{color:'#b7dc55'},hovertemplate:'%{x}<br>Malignant recall %{y:.1%}<extra></extra>'}],{...base,barmode:'group',yaxis:{...base.yaxis,tickformat:'.0%',range:[0,1],title:'Score'},legend:{orientation:'h',y:1.12},xaxis:{...base.xaxis,tickangle:0}},cfg);
function animateValue(el,value){const end=Number(value),start=Number(el.dataset.value||0),t0=performance.now();el.dataset.value=end;function step(now){let p=Math.min((now-t0)/360,1);el.textContent=(start+(end-start)*(1-Math.pow(1-p,3))).toFixed(3);if(p<1)requestAnimationFrame(step)}requestAnimationFrame(step)}
function update(){const name=select.value,d=D[name],s=S.find(x=>x.model===name);animateValue(document.getElementById('kAccuracy'),s.accuracy);animateValue(document.getElementById('kRecall'),s.malignant_recall);animateValue(document.getElementById('kF1'),s.f1);animateValue(document.getElementById('kAuc'),d.auc);
const cmTrace={type:'heatmap',z:d.cm,x:['Predicted malignant','Predicted benign'],y:['Actual malignant','Actual benign'],colorscale:[[0,'#27302b'],[1,'#789e52']],text:d.cm.map(r=>r.map(String)),texttemplate:'%{text}',textfont:{size:18,color:'#f0f4ef'},showscale:false,hovertemplate:'%{y}<br>%{x}: %{z}<extra></extra>'};const cmLayout={...base,margin:{l:112,r:8,t:8,b:49},xaxis:{side:'bottom',showgrid:false,tickfont:{size:9}},yaxis:{autorange:'reversed',showgrid:false,tickfont:{size:9}}};if(document.getElementById('cm').data)Plotly.react('cm',[cmTrace],cmLayout,cfg);else Plotly.newPlot('cm',[cmTrace],cmLayout,cfg);
const roc=[{x:d.fpr,y:d.tpr,type:'scatter',mode:'lines',name:name,line:{color:'#b7dc55',width:2},hovertemplate:'False positive rate %{x:.2f}<br>True positive rate %{y:.2f}<extra></extra>'},{x:[0,1],y:[0,1],type:'scatter',mode:'lines',name:'Chance',line:{color:'#68736c',dash:'dash'}}];const rocLayout={...base,xaxis:{...base.xaxis,title:'False positive rate',range:[0,1]},yaxis:{...base.yaxis,title:'True positive rate',range:[0,1]},legend:{orientation:'h',y:1.12}};if(document.getElementById('roc').data)Plotly.react('roc',roc,rocLayout,cfg);else Plotly.newPlot('roc',roc,rocLayout,cfg);
const items=d.features;const feat=[{type:'bar',orientation:'h',x:items.map(x=>x[1]).reverse(),y:items.map(x=>x[0]).reverse(),marker:{color:'#789e52'},hovertemplate:'%{y}<br>Importance %{x:.3f}<extra></extra>'}];const featLayout={...base,margin:{l:165,r:8,t:6,b:36},xaxis:{...base.xaxis,title:'Relative importance'},yaxis:{...base.yaxis,gridcolor:'rgba(0,0,0,0)',automargin:true,tickfont:{size:9}}};document.getElementById('featureNote').textContent=d.importance_label+'; not causal.';if(document.getElementById('features').data)Plotly.react('features',feat,featLayout,cfg);else Plotly.newPlot('features',feat,featLayout,cfg)}select.addEventListener('change',update);update();
</script></body></html>'''
    (OUT / "dashboard.html").write_text(html, encoding="utf-8")
    return html


if __name__ == "__main__":
    dashboard(build_metrics())
    print(f"Dashboard written to {OUT / 'dashboard.html'}")
