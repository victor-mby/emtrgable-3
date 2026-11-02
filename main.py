"""Punto de entrada: entrena los modelos y levanta el panel web."""
import argparse
from src.pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Sistema CONAUTI")
    parser.add_argument("--solo-entrenar", action="store_true", help="Entrena y termina sin iniciar el panel")
    args = parser.parse_args()
    print("CONAUTI | DETECCIÓN DE ANOMALÍAS EN AGENTES DE IA")
    report = run_pipeline()
    print(f"Datos: {report['registros']} | normales: {report['normales']} | anomalías: {report['anomalias']}")
    print(f"MLP accuracy: {report['mlp']['accuracy']:.3f} | F1: {report['mlp']['f1_score']:.3f}")
    if not args.solo_entrenar:
        from app import app
        app.run(host="127.0.0.1", port=5000, debug=False)

if __name__ == "__main__":
    main()
