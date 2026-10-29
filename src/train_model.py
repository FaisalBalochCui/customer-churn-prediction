from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,ConfusionMatrixDisplay,classification_report
from sklearn.inspection import permutation_importance
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data"/"customer_churn.csv"; OUT=ROOT/"outputs"; OUT.mkdir(exist_ok=True)
def main():
    df=pd.read_csv(DATA)
    if "churn" not in df: raise ValueError("Missing churn target")
    df=df.dropna(subset=["churn"]).copy(); y=df.churn.astype(int); X=df.drop(columns=["churn","customer_id"],errors="ignore")
    num=X.select_dtypes(include="number").columns.tolist(); cat=X.select_dtypes(exclude="number").columns.tolist()
    prep=ColumnTransformer([("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
      ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    candidates={"Logistic Regression":LogisticRegression(max_iter=1000,class_weight="balanced"),
      "Random Forest":RandomForestClassifier(n_estimators=300,random_state=42,class_weight="balanced",min_samples_leaf=3)}
    pipes={}; rows=[]
    for name,clf in candidates.items():
        pipe=Pipeline([("prep",prep),("model",clf)]); pipe.fit(Xtr,ytr); pipes[name]=pipe
        pred=pipe.predict(Xte); proba=pipe.predict_proba(Xte)[:,1]
        rows.append({"model":name,"accuracy":accuracy_score(yte,pred),"precision":precision_score(yte,pred,zero_division=0),
          "recall":recall_score(yte,pred,zero_division=0),"f1":f1_score(yte,pred,zero_division=0),"roc_auc":roc_auc_score(yte,proba)})
    metrics=pd.DataFrame(rows).sort_values("roc_auc",ascending=False); metrics.to_csv(OUT/"model_metrics.csv",index=False)
    best_name=metrics.iloc[0]["model"]; best=pipes[best_name]; pred=best.predict(Xte); proba=best.predict_proba(Xte)[:,1]
    out=Xte.copy(); out["actual_churn"]=yte.values; out["predicted_churn"]=pred; out["churn_probability"]=proba; out.to_csv(OUT/"test_predictions.csv",index=False)
    (OUT/"classification_report.txt").write_text("SYNTHETIC DEMO ONLY — not a real business finding.\\nSelected by holdout ROC-AUC: "+best_name+"\\n\\n"+classification_report(yte,pred,zero_division=0),encoding="utf-8")
    ConfusionMatrixDisplay.from_predictions(yte,pred); plt.title(f"Synthetic Churn Demo — {best_name}"); plt.tight_layout(); plt.savefig(OUT/"confusion_matrix.png",dpi=160); plt.close()
    perm=permutation_importance(best,Xte,yte,n_repeats=5,random_state=42,scoring="roc_auc")
    imp=pd.Series(perm.importances_mean,index=Xte.columns).sort_values().tail(12); imp.plot(kind="barh",figsize=(8,5))
    plt.title("Permutation Importance — Synthetic Demo"); plt.xlabel("Mean ROC-AUC decrease when shuffled"); plt.tight_layout(); plt.savefig(OUT/"feature_importance.png",dpi=160); plt.close()
    print(metrics.to_string(index=False)); print("Selected model:",best_name); print("Outputs:",OUT)
if __name__=="__main__": main()
