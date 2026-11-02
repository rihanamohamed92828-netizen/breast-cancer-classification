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
    scores, fitted = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        malignant_recall = recall_score(y_test, pred, pos_label=0)
        scores.append({"model": name, "accuracy": accuracy_score(y_test, pred),
                       "malignant_recall": malignant_recall,
                       "precision": precision_score(y_test, pred, pos_label=0),
                       "f1": f1_score(y_test, pred, pos_label=0)})
        fitted[name] = model
    winner = max(scores, key=lambda item: (item["malignant_recall"], item["f1"]))
    model = fitted[winner["model"]]
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, list(model.classes_).index(0)]
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    fpr, tpr, _ = roc_curve(y_test.eq(0), proba)
    auc = roc_auc_score(y_test.eq(0), proba)
    if winner["model"] == "Random forest":
        importance = model.feature_importances_
    else:
        classifier = model[-1]
        importance = np.abs(classifier.coef_[0])
    features = sorted(zip(data.feature_names, importance), key=lambda pair: pair[1], reverse=True)[:8]
    result = {"scores": scores, "winner": winner["model"], "cm": cm.tolist(), "fpr": fpr.tolist(),
              "tpr": tpr.tolist(), "auc": auc, "features": [[str(n), float(v)] for n, v in features],
              "n": int(len(y)), "test_n": int(len(y_test)), "malignant": int((y == 0).sum()), "benign": int((y == 1).sum())}
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame(scores).to_csv(OUT / "model_comparison.csv", index=False)
    return result


def dashboard(d):
    scores_json = json.dumps(d["scores"])
    cm_json = json.dumps(d["cm"])
    fpr_json, tpr_json = json.dumps(d["fpr"]), json.dumps(d["tpr"])
    feat_json = json.dumps(d["features"])
    html = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Breast Cancer Model Evaluation</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script><style>
*{box-sizing:border-box}body{margin:0;background:#e9ecef;color:#252d35;font:13px Arial,Helvetica,sans-serif}.wrap{max-width:1440px;margin:0 auto;padding:18px 22px 20px}.header{background:#17365d;color:white;padding:13px 18px;margin-bottom:13px;border:1px solid #122b4a}.eyebrow{font-size:11px;color:#d6deea;margin-bottom:4px}.title{font-size:21px;font-weight:600;margin:0}.layout{display:grid;grid-template-columns:205px minmax(0,1fr);gap:13px}.rail{background:#fff;border:1px solid #c8cfd6;padding:13px 14px;align-self:stretch}.rail h2{font-size:14px;font-weight:600;margin:0 0 12px;color:#293b4f}.fact{border-top:1px solid #e0e4e8;padding:9px 0}.fact:first-of-type{border-top:0}.fact-label{font-size:10px;color:#69737d;margin-bottom:4px}.fact-value{font-size:13px;line-height:1.35;color:#293b4f;font-weight:600}.rail-note{font-size:11px;color:#616b75;line-height:1.45;margin-top:11px}.main{min-width:0}.meta{font-size:11px;color:#59636e;margin:0 0 9px}.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px;margin-bottom:10px}.card,.panel{background:#fff;border:1px solid #c8cfd6;border-radius:2px}.card{padding:10px 12px;min-height:68px}.label{font-size:10px;color:#626d77;margin-bottom:6px}.value{font-size:20px;font-weight:600;color:#203954;line-height:1.15}.hint{font-size:10px;color:#69737d;margin-top:4px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}.panel{padding:10px 12px;min-height:285px}.panel h2{font-size:14px;font-weight:600;color:#293b4f;margin:0 0 2px}.panel p{font-size:10px;color:#69737d;margin:0 0 2px}.chart{height:238px}.foot{border-top:1px solid #c8cfd6;margin-top:11px;padding-top:8px;color:#59636e;font-size:10px;line-height:1.45}.foot strong{color:#34495e}.disclaimer{margin-top:4px;color:#69737d;font-size:10px}@media(max-width:850px){.wrap{padding:12px}.layout{grid-template-columns:1fr}.rail{display:grid;grid-template-columns:repeat(2,1fr);gap:0 12px}.rail h2,.rail-note{grid-column:1/-1}.cards{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}}@media(max-width:520px){.cards{grid-template-columns:1fr 1fr}.panel{padding:8px}.chart{height:245px}}
</style></head><body><main class="wrap"><header class="header"><div class="eyebrow">Model evaluation · Wisconsin diagnostic dataset</div><h1 class="title">Breast cancer classification</h1></header><div class="layout"><aside class="rail"><h2>Dataset &amp; method</h2><div class="fact"><div class="fact-label">Source</div><div class="fact-value">Wisconsin Diagnostic<br>Breast Cancer</div></div><div class="fact"><div class="fact-label">Observations</div><div class="fact-value">'''+f'''{d['n']:,} total · {d['malignant']} malignant · {d['benign']} benign'''+r'''</div></div><div class="fact"><div class="fact-label">Measurements</div><div class="fact-value">30 numeric nucleus features</div></div><div class="fact"><div class="fact-label">Evaluation</div><div class="fact-value">Stratified 80/20 holdout</div></div><div class="fact"><div class="fact-label">Model selection</div><div class="fact-value">Malignant recall, then F1</div></div><div class="rail-note">Features are derived from digitized images; original medical images are not included.</div></aside><section class="main"><p class="meta">Three baseline classifiers · fixed random seed · test set: '''+f'''{d['test_n']} observations'''+r'''</p><section class="cards"><div class="card"><div class="label">Dataset observations</div><div class="value">'''+f'''{d['n']:,}'''+r'''</div></div><div class="card"><div class="label">Test observations</div><div class="value">'''+f'''{d['test_n']}'''+r'''</div><div class="hint">20% stratified split</div></div><div class="card"><div class="label">Selected model</div><div class="value" style="font-size:16px">'''+d['winner']+r'''</div><div class="hint">Malignant recall, then F1</div></div><div class="card"><div class="label">ROC AUC</div><div class="value">'''+f'''{d['auc']:.3f}'''+r'''</div><div class="hint">Malignant class positive</div></div></section><section class="grid"><div class="panel"><h2>Accuracy and malignant recall by model</h2><div id="models" class="chart"></div></div><div class="panel"><h2>Confusion matrix — selected model</h2><div id="cm" class="chart"></div></div><div class="panel"><h2>ROC curve — selected model</h2><div id="roc" class="chart"></div></div><div class="panel"><h2>Top eight feature importance measures</h2><p>Absolute coefficients or forest importance; not causal.</p><div id="features" class="chart"></div></div></section><div class="foot"><strong>Source:</strong> Breast Cancer Wisconsin (Diagnostic), distributed with scikit-learn. <strong>Method:</strong> one stratified 80/20 split with fixed seed. Educational comparison only; not a clinical diagnostic tool.</div><div class="disclaimer">Metric estimates depend on the split and model settings. Do not use this benchmark to guide patient care.</div></section></div></main><script>const S='''+scores_json+r''';const C='''+cm_json+r''';const F='''+fpr_json+r''';const T='''+tpr_json+r''';const I='''+feat_json+r''';const base={paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{family:'Arial, sans-serif',color:'#4f5b66',size:10},margin:{l:46,r:10,t:8,b:43},xaxis:{gridcolor:'#e1e5e9',zerolinecolor:'#cbd2d9'},yaxis:{gridcolor:'#e1e5e9',zerolinecolor:'#cbd2d9'}};Plotly.newPlot('models',[{type:'bar',name:'Accuracy',x:S.map(x=>x.model),y:S.map(x=>x.accuracy),marker:{color:'#4472c4'},hovertemplate:'%{x}<br>Accuracy %{y:.1%}<extra></extra>'},{type:'bar',name:'Malignant recall',x:S.map(x=>x.model),y:S.map(x=>x.malignant_recall),marker:{color:'#70ad47'},hovertemplate:'%{x}<br>Malignant recall %{y:.1%}<extra></extra>'}],{...base,barmode:'group',yaxis:{...base.yaxis,tickformat:'.0%',range:[0,1],title:'Score'},legend:{orientation:'h',y:1.12},xaxis:{...base.xaxis,tickangle:0}},{displayModeBar:false,responsive:true});Plotly.newPlot('cm',[{type:'heatmap',z:C,x:['Predicted malignant','Predicted benign'],y:['Actual malignant','Actual benign'],colorscale:[[0,'#f1f4f8'],[1,'#4472c4']],text:C.map(r=>r.map(v=>String(v))),texttemplate:'%{text}',textfont:{size:18,color:'#202a35'},showscale:false,hovertemplate:'%{y}<br>%{x}: %{z}<extra></extra>'}],{...base,margin:{l:110,r:8,t:8,b:48},xaxis:{side:'bottom',showgrid:false,tickfont:{size:9}},yaxis:{autorange:'reversed',showgrid:false,tickfont:{size:9}}},{displayModeBar:false,responsive:true});Plotly.newPlot('roc',[{x:F,y:T,type:'scatter',mode:'lines',name:'Selected model',line:{color:'#4472c4',width:2},hovertemplate:'FPR %{x:.2f}<br>TPR %{y:.2f}<extra></extra>'},{x:[0,1],y:[0,1],type:'scatter',mode:'lines',name:'Chance',line:{color:'#929ba3',dash:'dash'}}],{...base,xaxis:{...base.xaxis,title:'False positive rate',range:[0,1]},yaxis:{...base.yaxis,title:'True positive rate',range:[0,1]},legend:{orientation:'h',y:1.12}},{displayModeBar:false,responsive:true});Plotly.newPlot('features',[{type:'bar',orientation:'h',x:I.map(x=>x[1]).reverse(),y:I.map(x=>x[0]).reverse(),marker:{color:'#70ad47'},hovertemplate:'%{y}<br>Importance %{x:.3f}<extra></extra>'}],{...base,margin:{l:165,r:8,t:6,b:36},xaxis:{...base.xaxis,title:'Relative importance'},yaxis:{...base.yaxis,gridcolor:'rgba(0,0,0,0)',automargin:true,tickfont:{size:9}}},{displayModeBar:false,responsive:true});</script></body></html>'''
    (OUT / "dashboard.html").write_text(html, encoding="utf-8")
    return html


if __name__ == "__main__":
    dashboard(build_metrics())
    print(f"Dashboard written to {OUT / 'dashboard.html'}")
