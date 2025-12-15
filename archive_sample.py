#!/usr/bin/env python3
"""
Small helper to save a local image + metadata into brand-auth-backend/dataset/archive.
Usage:
  python archive_sample.py --image "/path/to/2 cc.png" --description "bad product" --label Authentic --cv_conf 1 --nlp_score 0.637 --product_url "http://example" --product_id pid123

This script does not require the Flask app to be running.
"""
import os, uuid, json, argparse, datetime
from PIL import Image

def ensure_dirs():
    base = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'archive')
    base = os.path.normpath(base)
    images = os.path.join(base, 'images')
    meta = os.path.join(base, 'meta')
    os.makedirs(images, exist_ok=True)
    os.makedirs(meta, exist_ok=True)
    return images, meta


def save_sample(image_path, metadata):
    images_dir, meta_dir = ensure_dirs()
    archive_id = str(uuid.uuid4())
    img_name = f"{archive_id}.jpg"
    img_dest = os.path.join(images_dir, img_name)
    try:
        im = Image.open(image_path)
        im.convert('RGB').save(img_dest, format='JPEG', quality=90)
    except Exception as e:
        print('Failed to copy/save image:', e)
        return None
    metadata_out = dict(metadata or {})
    metadata_out['archive_id'] = archive_id
    metadata_out['archive_image'] = os.path.relpath(img_dest, start=os.path.dirname(os.path.dirname(__file__)))
    metadata_out['saved_at'] = datetime.datetime.utcnow().isoformat()
    meta_path = os.path.join(meta_dir, archive_id + '.json')
    with open(meta_path, 'w', encoding='utf-8') as mf:
        json.dump(metadata_out, mf, ensure_ascii=False, indent=2)
    print('Saved image ->', img_dest)
    print('Saved metadata ->', meta_path)
    return {'archive_id': archive_id, 'image_path': img_dest, 'meta_path': meta_path}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True, help='Path to local image to archive')
    p.add_argument('--description', default='', help='Product description or review text')
    p.add_argument('--label', default='', help='Label (Authentic/Counterfeit)')
    p.add_argument('--cv_conf', type=float, default=None, help='CV confidence')
    p.add_argument('--nlp_score', type=float, default=None, help='NLP score')
    p.add_argument('--product_url', default=None)
    p.add_argument('--product_id', default=None)
    args = p.parse_args()

    meta = {'description': args.description, 'label': args.label}
    if args.cv_conf is not None:
        meta['cv_conf'] = float(args.cv_conf)
    if args.nlp_score is not None:
        meta['nlp_score'] = float(args.nlp_score)
    if args.product_url:
        meta['product_url'] = args.product_url
    if args.product_id:
        meta['product_id'] = args.product_id

    res = save_sample(args.image, meta)
    if not res:
        print('Archive save failed')
    else:
        print('Archive save successful:', res)
