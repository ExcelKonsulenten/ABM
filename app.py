# app.py - Streamlit-app for enkel filtrering og grafer (survival + dropouts)
# Kjør: streamlit run app.py
from __future__ import annotations
import glob, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

COL_MAP = {
    "id":"ID","status":"Status","startdato":"Startdato","opprettet den":"Opprettet den",
    "stoppdato":"Stoppdato","sist betalt dato":"Sist betalt dato","betalt til og med dato":"Betalt til og med dato",
    "neste giro dato":"Neste giro dato","betalingsmåte":"Betalingsmåte","totalt betalt":"Totalt betalt",
    "selger":"Selger","dør til dør sellerid":"Dør til dør sellerId","dør til dør sellerId":"Dør til dør sellerId",
}
def robust_read_csv(path: Path) -> pd.DataFrame:
    seps, encs = [";", ","], ["utf-8-sig","latin-1","cp1252"]
    for sep in seps:
        for enc in encs:
            try: return pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
            except Exception: continue
    st.error(f"Kunne ikke lese {path.name}"); return pd.DataFrame()
def normalize_columns(df): return df.rename(columns={c: COL_MAP.get(c.strip().lower(), c.strip()) for c in df.columns})
def coerce_dates(df):
    for c in ["Opprettet den","Startdato","Stoppdato","Sist betalt dato","Betalt til og med dato","Neste giro dato"]:
        if c in df.columns: df[c]=pd.to_datetime(df[c],errors="coerce",dayfirst=True)
    return df
def _extract_first_int(v):
    if pd.isna(v): return np.nan
    m=re.search(r"-?\d+",str(v)); return int(m.group(0)) if m else np.nan


def derive_avdeling(row: pd.Series) -> str:
    """
    Streng 4-sifret mapping basert på Dør til dør sellerId -> Selger.
    Regler:
      Oslo: 0–1999 (EKSKLUDERT 1007, 1025, 1030)
      Bergen: 2000–2999
      Stavanger: 3000–3999
      Trondheim: 4000–4999
      Kristiansand: 5000–5999 (INKLUDERT 1007, 1025, 1030)
      Tromsø: 7000–7999
    """
    def _extract_first_int(val):
        if pd.isna(val): return np.nan
        m = re.search(r"-?\d+", str(val))
        if not m: return np.nan
        try: return int(m.group(0))
        except: return np.nan

    # Prioritet: Dør til dør sellerId, ellers Selger
    n = _extract_first_int(row.get("Dør til dør sellerId", np.nan))
    if pd.isna(n):
        n = _extract_first_int(row.get("Selger", np.nan))
    if pd.isna(n):
        return "Ukjent"

    # Unntak -> Kristiansand
    if n in {1007, 1025, 1030}:
        return "Kristiansand"

    if   0    <= n <= 1999: return "Oslo"
    if 2000   <= n <= 2999: return "Bergen"
    if 3000   <= n <= 3999: return "Stavanger"
    if 4000   <= n <= 4999: return "Trondheim"
    if 5000   <= n <= 5999: return "Kristiansand"
    if 7000   <= n <= 7999: return "Tromsø"
    return "Ukjent"

def payments_from_total(total):
    if pd.isna(total): return np.nan,True
    s=str(total).strip().replace(" ","").replace("kr","").replace("KR","").replace(",",".")
    try: t=float(s)
    except: return np.nan,True
    tol=2.0
    for base in [300,325]:
        n=round(t/base)
        if abs(t-n*base)<=tol and n>=0: return int(n),False
    return np.nan,True
def load_data():
    parts=[]; files=[Path(p) for p in glob.glob("dennedata*.csv")]+[Path(p) for p in glob.glob("DENNEDATA*.CSV")]
    if not files: st.warning("Fant ingen filer som matcher 'dennedata*.csv'"); return pd.DataFrame()
    for p in files:
        df=robust_read_csv(p)
        if df.empty: continue
        df=normalize_columns(df); df["__kildefil"]=p.name; parts.append(df)
    if not parts: return pd.DataFrame()
    df=pd.concat(parts,ignore_index=True); df=coerce_dates(df); df["Avdeling"]=df.apply(derive_avdeling,axis=1)
    res=df.get("Totalt betalt",pd.Series([],dtype=float)).apply(payments_from_total)
    if len(res): df["Antall betalinger"]=[a for a,_ in res]; df["Avvik"]=[b for _,b in res]
    else: df["Antall betalinger"]=np.nan; df["Avvik"]=True
    df["Kansellert"]=df.get("Status","").astype(str).str.lower().str.contains("inaktiv|stopp|sagt opp|opphevet")
    if "Betalingsmåte" in df.columns:
        df["Betalingsmåte"]=df["Betalingsmåte"].astype(str).str.strip()
        # Normaliser noen vanlige varianter
        _map = {
            "vipps": "Vipps fast betaling",
            "vipps fast betaling": "Vipps fast betaling",
            "bank": "Bank",
            "avtalegiro": "Avtalegiro",
            "avtale giro": "Avtalegiro",
        }
        df["Betalingsmåte"] = df["Betalingsmåte"].str.lower().map(_map).fillna(df["Betalingsmåte"])
    return df

def apply_filters(df: pd.DataFrame):
    st.sidebar.header("Filtre")
    # Skjul avvik
    if st.sidebar.checkbox("Skjul avvik (beløp ≠ 300/325-multiplum)", value=True):
        df = df[df["Avvik"] == False]

    # --- Kun Startdato + År/Måned ---
    if "Startdato" in df.columns:
        sd = pd.to_datetime(df["Startdato"])
        years = sorted(sd.dt.year.dropna().unique().astype(int).tolist())
        months = list(range(1,13))
        month_names = ["jan","feb","mar","apr","mai","jun","jul","aug","sep","okt","nov","des"]

        # Velg år
        year = st.sidebar.selectbox("År (Startdato)", options=["Alle"] + [str(y) for y in years], index=(0 if len(years)==0 else len(years)))
        # Velg måneder (flervalg)
        month_opts = [f"{m:02d} - {month_names[m-1]}" for m in months]
        month_sel = st.sidebar.multiselect("Måneder (Startdato)", options=month_opts, default=month_opts)

        if year != "Alle":
            df = df[sd.dt.year == int(year)]
            sd = pd.to_datetime(df["Startdato"])  # refresh after filtering
        # Filtrer på valgt(e) måneder
        sel_month_nums = [int(x.split(" - ")[0]) for x in month_sel] if month_sel else months
        df = df[pd.to_datetime(df["Startdato"]).dt.month.isin(sel_month_nums)]

    # Avdeling
    avd_vals = sorted(df["Avdeling"].dropna().unique().tolist())
    avd_sel = st.sidebar.multiselect("Avdeling", avd_vals, default=None)
    if avd_sel:
        df = df[df["Avdeling"].isin(avd_sel)]

    # Betalingsmåte
    if "Betalingsmåte" in df.columns:
        pay_vals = sorted(df["Betalingsmåte"].dropna().unique().tolist())
        pay_sel = st.sidebar.multiselect("Betalingsmåte", pay_vals, default=None)
        if pay_sel:
            df = df[df["Betalingsmåte"].isin(pay_sel)]

    # Antall betalinger
    if df["Antall betalinger"].notna().any():
        min_b = int(np.nanmin(df["Antall betalinger"]))
        max_b = int(np.nanmax(df["Antall betalinger"]))
        if max_b <= min_b:
            st.sidebar.write(f"Antall betalinger i utvalg: **{min_b}**")
            sel_min, sel_max = min_b, max_b
        else:
            sel_min, sel_max = st.sidebar.slider("Antall betalinger", min_value=min_b, max_value=max_b, value=(min_b, max_b))
        df = df[df["Antall betalinger"].fillna(-1).between(sel_min, sel_max)]
    return df

def compute_survival_and_dropouts(df):
    if df.empty: return pd.DataFrame(),pd.DataFrame(),0
    agg=df.groupby("ID").agg(Antall_betalinger=("Antall betalinger","max"),Kansellert=("Kansellert","max")).reset_index()
    base=len(agg); max_n=int(np.nanmax(agg["Antall_betalinger"])) if agg["Antall_betalinger"].notna().any() else 0
    xs,ys=[],[]
    for n in range(1,max_n+1): xs.append(n); ys.append(int((agg["Antall_betalinger"].fillna(0)>=n).sum()))
    surv=pd.DataFrame({"Antall betalinger":xs,"Antall aktive eller fullførte":ys})
    drops=agg[(agg["Kansellert"]==True)&(agg["Antall_betalinger"].notna())].groupby("Antall_betalinger")["ID"].count()
    drop=pd.DataFrame({"Antall betalinger":range(1,max_n+1),"Antall som meldte seg ut":[drops.get(i,0) for i in range(1,max_n+1)]})
    return surv,drop,base
st.set_page_config(page_title="Abonnement – survival & dropout",layout="wide")
st.title("Abonnement – survival & dropout")
df=load_data()
if df.empty: st.stop()
with st.expander("Se rådata (filtrerbar)",expanded=False): st.dataframe(df)
fdf=apply_filters(df)
colA,colB=st.columns(2)
surv_df,drop_df,base=compute_survival_and_dropouts(fdf)
vis_prosent = st.sidebar.checkbox('Vis i prosent (%)', value=True)
with colA:
    st.subheader("Survival / Retensjon")
    if surv_df.empty: st.info("Ingen data etter filtrering.")
    else:
        y = surv_df["Antall aktive eller fullførte"].astype(float)
        if vis_prosent and base>0:
            y = y / float(base) * 100.0
            ylabel = "Andel abonnenter (>= n) [%]"
        else:
            ylabel = "Antall abonnenter (>= n)"
        fig=plt.figure()
        xs = surv_df["Antall betalinger"].astype(int)
        plt.plot(xs, y, marker="o")
        plt.xlabel("Antall betalinger"); plt.ylabel(ylabel); plt.grid(True, linestyle="--", alpha=0.4)
        try:
            import matplotlib.pyplot as _plt
            from matplotlib.ticker import MaxNLocator
            _ax = _plt.gca()
            _ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            # Sett diskrete ticks og marginer
            max_n = int(xs.max()) if len(xs) else 0
            if max_n > 0:
                if max_n <= 25:
                    _ax.set_xticks(list(range(1, max_n+1)))
                _ax.set_xlim(0.5, max_n + 0.5)
            if vis_prosent:
                _ax.set_ylim(0, 100)
        except Exception:
            pass
        st.pyplot(fig); st.caption(f"Totalt unike abonnement i utvalget: {base}")
with colB:
    st.subheader("Når meldte de seg ut – per betalingsmåte")
    if "Betalingsmåte" not in fdf.columns:
        st.info("Ingen kolonne for Betalingsmåte i data.")
    else:
        # Aggreger per ID først
        agg = fdf.groupby("ID").agg(
            Antall_betalinger=("Antall betalinger","max"),
            Kansellert=("Kansellert","max"),
            Betalingsmåte=("Betalingsmåte","first"),
        ).reset_index()
        if agg.empty:
            st.info("Ingen data etter filtrering.")
        else:
            # Basestørrelse per betalingsmåte (antall unike ID)
            bases = agg.groupby("Betalingsmåte")["ID"].count()
            max_n = int(agg["Antall_betalinger"].max()) if agg["Antall_betalinger"].notna().any() else 0
            import matplotlib.pyplot as plt
            from matplotlib.ticker import MaxNLocator
            fig2 = plt.figure()
            ax2 = plt.gca()
            # Tegn en linje per betalingsmåte
            for meth, base in bases.items():
                sub = agg[agg["Betalingsmåte"]==meth]
                # hvor mange som meldte seg ut etter n betalinger (akkurat n)
                drops = sub[(sub["Kansellert"]==True) & (sub["Antall_betalinger"].notna())].groupby("Antall_betalinger")["ID"].count()
                xs = list(range(1, max_n+1))
                ys = [drops.get(n, 0) for n in xs]
                if vis_prosent and base>0:
                    ys = [y / base * 100.0 for y in ys]
                ax2.plot(xs, ys, marker="o", label=str(meth))
            ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
            if max_n > 0:
                ax2.set_xlim(0.5, max_n + 0.5)
            ax2.grid(True, linestyle="--", alpha=0.4)
            ax2.set_xlabel("Antall betalinger")
            ax2.set_ylabel("Andel som meldte seg ut [%]" if vis_prosent else "Antall som meldte seg ut")
            if vis_prosent:
                ax2.set_ylim(0, 100)
            ax2.legend(title="Betalingsmåte")
            st.pyplot(fig2)
st.download_button("Last ned tabell (CSV)",data=fdf.to_csv(index=False).encode("utf-8-sig"),file_name="filtrert_rådata.csv",mime="text/csv")
