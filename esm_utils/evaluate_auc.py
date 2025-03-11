import argparse
import csv
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, matthews_corrcoef, average_precision_score

def parse_ground_truth(file_path):
    """
    Lee un archivo de ground truth (test.fasta o test.csv) y devuelve un diccionario:
    { key: [label1, label2, ..., labelN] }
    Donde cada posición vale 1 si el aminoácido está en mayúscula (parte del epítopo) o 0 si no.
    Se asume que el archivo tiene registros en dos líneas: 
      - Primera línea: encabezado que comienza con ">"
      - Segunda línea: la secuencia (puede venir entre comillas)
    """
    records = {}
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    i = 0
    while i < len(lines):
        if lines[i].startswith('>'):
            header = lines[i][1:].split()[0]
            seq = lines[i+1].replace('"','').strip()
            labels = [1 if c.isupper() else 0 for c in seq]
            records[header] = labels
            i += 2
        else:
            i += 1
    return records

def parse_predictions(file_path):
    """
    Lee el archivo predict.csv con la siguiente estructura:
      key, pos, a, pred, y/n
    Se asume que las columnas están separadas por comas.
    Retorna una lista de tuplas (key, pos, pred_score)
    """
    preds = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if len(row) < 5:
                continue
            key = row[0].strip()
            try:
                pos = int(row[1].strip())
                pred_score = float(row[3].strip())
            except ValueError:
                continue
            preds.append((key, pos, pred_score))
    return preds

def main():
    parser = argparse.ArgumentParser(description="Evalúa AUC, BAC, MCC y PR-AUC a partir de un archivo de predicciones y un ground truth.")
    parser.add_argument("--pred", required=True, help="Ruta al archivo predict.csv")
    parser.add_argument("--gt", required=True, help="Ruta al archivo ground truth (test.csv o test.fasta)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Umbral para obtener etiquetas de clasificación (default: 0.5)")
    args = parser.parse_args()

    ground_truth = parse_ground_truth(args.gt)
    predictions = parse_predictions(args.pred)

    y_true = []
    y_score = []
    missing = 0

    for key, pos, score in predictions:
        if key in ground_truth:
            labels = ground_truth[key]
            if 1 <= pos <= len(labels):
                y_true.append(labels[pos-1])
                y_score.append(score)
            else:
                missing += 1
        else:
            missing += 1

    if missing:
        print(f"Warning: {missing} predicciones no pudieron ser emparejadas con el ground truth.")

    if not y_true:
        print("Error: No se encontraron muestras emparejadas para evaluar.")
        return

    try:
        auc = roc_auc_score(y_true, y_score)
        # Para calcular BAC y MCC, se obtiene la clase predicha usando un umbral.
        y_pred = [1 if s >= args.threshold else 0 for s in y_score]
        bac = balanced_accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        pr_auc = average_precision_score(y_true, y_score)
        
        print(f"AUC: {auc}")
        print(f"BAC: {bac}")
        print(f"MCC: {mcc}")
        print(f"PR-AUC: {pr_auc}")
    except Exception as e:
        print(f"Error al calcular las métricas: {e}")

if __name__ == "__main__":
    main()