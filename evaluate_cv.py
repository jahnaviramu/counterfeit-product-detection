"""evaluate_cv.py
Evaluate a Keras CV model on a labeled folder structure.
Expects directory with subfolders for classes (e.g. authentic/, counterfeit/) containing test images.
Outputs classification report and per-image CSV.
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def load_image(path, img_size):
    from PIL import Image
    img = Image.open(path).convert('RGB')
    img = img.resize((img_size, img_size))
    arr = np.array(img).astype('float32')
    return arr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Folder with class subfolders')
    parser.add_argument('--model', default='data/cv_model.h5')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--out', default='cv_eval_results.csv')
    args = parser.parse_args()

    try:
        import tensorflow as tf
    except Exception as e:
        print('TensorFlow required to run evaluate_cv.py:', e)
        return
    model = tf.keras.models.load_model(args.model)

    # try to load saved class names next to model (e.g. model.h5.classes.joblib)
    classes = None
    try:
        import joblib
        classes_path = args.model + '.classes.joblib'
        if os.path.exists(classes_path):
            classes = joblib.load(classes_path)
    except Exception:
        classes = None

    # fallback: infer classes from folder names (sorted)
    if classes is None:
        classes = sorted([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])
    for cls in classes:
        cls_dir = os.path.join(args.data_dir, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append(os.path.join(cls_dir, fname))
                labels.append(cls)

    X = np.stack([load_image(p, args.img_size) for p in images], axis=0)
    # apply the same preprocessing used during training (EfficientNet preprocess)
    try:
        preprocess_fn = tf.keras.applications.efficientnet.preprocess_input
        X = preprocess_fn(X)
    except Exception:
        # if preprocess not available, scale to [0,1]
        X = X / 255.0
    preds = model.predict(X)
    preds_a = np.array(preds)
    y_pred_labels = []
    scores = []
    if preds_a.ndim == 2 and preds_a.shape[1] == 1:
        # binary sigmoid output, preds_a contains probabilities for class 1
        prob_pos = preds_a.ravel()
        threshold = 0.5
        y_preds = (prob_pos >= threshold).astype(int)
        # if classes available, assume classes[0]=negative, classes[1]=positive
        if len(classes) >= 2:
            y_pred_labels = [classes[int(y)] for y in y_preds]
        else:
            # fallback to numeric labels
            y_pred_labels = [str(int(y)) for y in y_preds]
        scores = prob_pos.tolist()
    else:
        # multi-class softmax
        idxs = np.argmax(preds_a, axis=1)
        y_pred_labels = [classes[i] if i < len(classes) else str(i) for i in idxs]
        # store max probability as score
        scores = np.max(preds_a, axis=1).tolist()

    print(classification_report(labels, y_pred_labels))
    print('Confusion matrix:\n', confusion_matrix(labels, y_pred_labels))

    out = []
    for i, p in enumerate(images):
        out.append({'image': p, 'label': labels[i], 'pred': y_pred_labels[i], 'score': scores[i] if i < len(scores) else None})
    pd.DataFrame(out).to_csv(args.out, index=False)
    print('Saved detailed CV results to', args.out)

if __name__ == '__main__':
    main()
