import sys, subprocess
from pathlib import Path
REQUIRED=["pandas","numpy","matplotlib","streamlit"]
def ensure_packages():
    import importlib
    for pkg in REQUIRED:
        try: importlib.import_module(pkg)
        except ImportError:
            print(f"Installerer {pkg} ...")
            subprocess.check_call([sys.executable,"-m","pip","install",pkg])
def main():
    ensure_packages()
    app=Path("app.py")
    if not app.exists():
        print("❌ Fant ikke app.py. Legg denne filen i samme mappe som app.py og CSV-filene.")
        return
    print("\n✅ Starter programmet – nettleseren åpnes automatisk...\n")
    subprocess.run([sys.executable,"-m","streamlit","run",str(app)],check=False)
if __name__=="__main__": main()
