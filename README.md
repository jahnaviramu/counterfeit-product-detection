# Brand Auth Backend — model & run notes

This folder contains the Flask API and helper scripts for the Brand Auth project. This README focuses on the SBERT-based NLP components and quick steps to produce artifacts the backend auto-loads.

## Important file locations

- Trained SBERT sklearn classifier (optional): `data/text/sbert_clf.joblib`
- Preferred SBERT model name or fine-tuned model path: `data/text/sbert_model_name.txt` (if this file contains a model name, the backend will attempt to load a SentenceTransformer with that name); alternatively place a fine-tuned SentenceTransformer directory under `data/text/sbert_finetuned` and write that directory name into `sbert_model_name.txt`.
- Legacy TF-IDF artifacts: `data/text/nlp_model.joblib` and `data/text/tfidf_vectorizer.joblib` (used by `/api/nlp_check`).

## How the backend auto-loads models

On startup `app.py` will:

- Attempt to load `sbert_clf.joblib` if present.
- Read `sbert_model_name.txt` to determine which SentenceTransformer model to use. If missing, it defaults to `all-MiniLM-L6-v2`.
- Try to preload a `SentenceTransformer` instance into memory as `sb_embedder` for faster inference. If it can't (missing packages or heavy resources), the endpoints fall back to lazy-loading on request.

## Training SBERT classifier (quick)

1. Install heavy dependencies in your backend venv (this repo's `requirements.txt` should include `sentence-transformers`, `scikit-learn`, `joblib`, `pandas`). Example:

```bash
# Windows (cmd.exe)
cd brand-auth-backend
venv\Scripts\activate
pip install -r requirements.txt
pip install sentence-transformers
```

2. Prepare sentence-level CSV(s) with columns `text` and `label`. You can create sentence-level data by running `preprocess_reviews.py` which splits review text into sentences.

3. Train classifier (this script is in the repo):

```bash
python train_sentence_transformer_classifier.py --input "data/text/*_sentences.csv" --output data/text/sbert_clf.joblib --model-name all-MiniLM-L6-v2
```

This will embed sentences using the named SentenceTransformer (downloaded automatically) and train a scikit-learn classifier. The script supports upsampling and probability calibration options.

4. Optionally fine-tune SBERT (higher quality):

```bash
python train_sbert_finetune.py --input data/text/*_sentences.csv --output-dir data/text/sbert_finetuned
```

Then write the model path into `data/text/sbert_model_name.txt`:

```
echo sbert_finetuned > data/text/sbert_model_name.txt
```

5. Place the trained classifier `sbert_clf.joblib` into `data/text/` and (optionally) `sbert_model_name.txt`. Restart the backend — it will attempt to preload the embedder and classifier.

## Quick API health check

After starting the backend, call the new endpoint to check model/component health and run a small sample prediction:

```bash
curl http://localhost:5000/api/model_health
```

The JSON response lists which components are loaded and returns a small sample prediction if the SBERT classifier is available.

## Notes and troubleshooting

- If `sentence-transformers` or `torch` are not installed, model preloading will fail but endpoints may still work when lazy-loading (if installed). Install the packages in the backend venv to avoid runtime errors.
- Large models require memory; prefer `all-MiniLM-L6-v2` for small-footprint inference.
- For CV training, see `cv_train.py` (Keras/TF) — make sure TensorFlow is installed in the environment before running.

## CV training (detailed)

1. Prepare your images in this layout (example):

```
brand-auth-backend/data/images/authentic/
brand-auth-backend/data/images/counterfeit/
```

2. Create and activate your backend venv and install dependencies (Windows example):

```bash
cd brand-auth-backend
venv\Scripts\activate
pip install -r requirements.txt
pip install tensorflow
```

3. Train a model with the provided starter (`cv_train.py`):

```bash
python cv_train.py --data_dir data/images --output data/cv_model.h5 --img_size 224 --batch 32 --epochs 15
```

The script uses EfficientNetB0 with a small classification head. It saves the best model to `data/cv_model.h5`.

4. After training, place the produced `data/cv_model.h5` file in the backend `data` folder (the script already writes there if you used the example command). Restart the backend; `/api/cv_check` will now use the trained model.

## Multimodal (CV + NLP) usage

The backend exposes `/api/multimodal_check` which accepts:
- an image file in multipart/form-data under the field name `image`
- and/or a text field `description` (form field) or JSON with `description`/`text`.

Example with `curl` (Windows PowerShell / cmd with proper quoting):

```bash
curl -X POST -F "image=@C:\path\to\authentic.jpg" -F "description=Official product from Brand X" http://localhost:5000/api/multimodal_check
```

Response (JSON) contains three keys: `cv`, `nlp`, and `fused`. Each holds modality-specific `authentic_prob` (0-1) and the `fused` section holds the combined probability and a boolean `authentic` decision.

You can tune the fusion weights with environment variables in `.env`:

```
MULTIMODAL_CV_WEIGHT=0.6
MULTIMODAL_NLP_WEIGHT=0.4
CV_AUTH_THRESHOLD=0.5
CV_IMG_SIZE=224
```

## Quick troubleshooting checklist for CV
- If `/api/cv_check` returns an error about TensorFlow missing, install `tensorflow` in the backend venv.
- If predictions are poor: gather more counterfeit examples, ensure images used at inference match training framing (cropped vs scene), and add augmentation.

## Product details, OCR, barcode and influencer reporting

New endpoints were added to support richer product registration and moderation workflows similar to features found in modern product-check apps (e.g., Checko-style flows):

- `POST /api/register_detailed` (multipart/form-data)
	- fields: `name` (required), `brand` (required), `description`, `serial_number`, `batchNumber`, `manufactureDate`, `manufacturer`, `origin_country`, `seller_info` (JSON string), `geolocation` (JSON or `lat` and `lng`), `register_blockchain` ("1" to register on chain)
	- image files under field name `images` (can upload many)
	- The backend will save images under `data/images/products/<product_id>/`, compute perceptual hash (phash) when `imagehash` is installed, run OCR with `pytesseract` (if installed) to extract textual clues (serials/labels), and decode any barcodes/QR codes if `pyzbar` is available.
	- The product document is stored in MongoDB `products` collection with `images`, `metadata`, and `reports` fields.

- `GET /api/product/<product_id>` — returns the stored product document (metadata, images, OCR/barcode/hash results, reports).

- `POST /api/report_suspicious` (JSON) — submit suspicious reports from users or influencers.
	- body: `{ product_id, reporter_name, reporter_handle, reason, evidence_images (optional list of base64 images) }`
	- The backend stores the report in `reports` collection and adds a minimal summary to the product's `reports` array.

- `GET /api/influencer_activity` — aggregates `reports` by `reporter_handle`, returning counts and latest activity. Useful to surfacing high-value reporters.

Notes:
- The OCR, barcode decoding and phash steps are optional and require additional Python packages (`imagehash`, `pytesseract`, `pyzbar`) to be installed in your backend venv. The endpoints handle missing libraries gracefully but those features will be disabled until installed.
- Evidence images uploaded as base64 via `report_suspicious` are saved under `data/reports/` and referenced from the stored report.



If you'd like, I can add step-by-step example commands to train on a sample dataset or wire CI to produce artifacts automatically.
