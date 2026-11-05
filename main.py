"""Punto de entrada del entregable final CONAUTI."""
import argparse
from src.pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="CONAUTI | 6 modelos de IA")
    parser.add_argument("--solo-entrenar", action="store_true")
    args = parser.parse_args()
    print("CONAUTI | ENTREGABLE FINAL")
    print("1. Árbol de Decisión | 2. K-Means | 3. DBSCAN")
    print("4. MLP | 5. Autoencoder | 6. CNN")
    report = run_pipeline()
    print(f"Registros: {report['registros']} | normales: {report['normales']} | anomalías: {report['anomalias']}")
    for name, data in report["modelos"].items():
        if "f1_score" in data:
            print(f"{name}: accuracy={data['accuracy']:.3f} | F1={data['f1_score']:.3f}")
        else:
            print(f"{name}: {data}")
    if not args.solo_entrenar:
        from app import app
        app.run(host="127.0.0.1", port=5000, debug=False)

if __name__ == "__main__":
    main()
