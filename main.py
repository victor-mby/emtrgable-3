"""Punto de entrada del proyecto CONAUTI."""

import json

from src.pipeline import run_pipeline


def main():
    print("=" * 66)
    print("CONAUTI S.A.C. | DETECCIÓN DE ANOMALÍAS EN AGENTES DE IA")
    print("=" * 66)
    print("Ejecutando generación de datos y entrenamiento de modelos...")

    report = run_pipeline()

    print("\nProceso finalizado.")
    print(f"Registros: {report['registros']}")
    print(f"Normales: {report['normales']}")
    print(f"Anómalos: {report['anomalias']}")
    print(f"Grupos K-Means: {report['kmeans_grupos']}")
    print(f"Atípicos Isolation Forest: {report['isolation_forest_atipicos']}")
    print("\nMétricas MLP:")
    print(json.dumps(report["mlp"], ensure_ascii=False, indent=2))
    print("\nArchivos generados en las carpetas data/, models/ y outputs/.")


if __name__ == "__main__":
    main()
