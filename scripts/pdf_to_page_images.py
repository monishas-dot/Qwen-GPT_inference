"""
Convert every PDF in a directory into one PNG per page, so a downstream
classifier can pick out the SRF page from each document.

Output structure:
    <output_dir>/<pdf_stem>/<pdf_stem>_page01.png
    <output_dir>/<pdf_stem>/<pdf_stem>_page02.png
    ...

Usage:
    python pdf_to_page_images.py <input_pdf_dir> <output_dir> [--dpi 200]

Requires: pip install pymupdf
"""

import sys
import os
import argparse

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Missing dependency. Install it with:\n    pip install pymupdf")
    sys.exit(1)


def convert_pdf_to_images(pdf_path: str, out_dir: str, dpi: int = 200):
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_out_dir = os.path.join(out_dir, stem)
    os.makedirs(pdf_out_dir, exist_ok=True)

    zoom = dpi / 72  # PyMuPDF's default is 72 dpi
    matrix = fitz.Matrix(zoom, zoom)

    saved_paths = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"!! Failed to open {pdf_path}: {e}")
        return []

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=matrix)
        out_path = os.path.join(pdf_out_dir, f"{stem}_page{page_index + 1:02d}.png")
        pix.save(out_path)
        saved_paths.append(out_path)

    doc.close()
    return saved_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_pdf_dir", help="Directory containing the filtered PDFs")
    parser.add_argument("output_dir", help="Directory to write per-page PNGs into")
    parser.add_argument("--dpi", type=int, default=200, help="Render resolution (default 200)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pdf_files = [
        f for f in os.listdir(args.input_pdf_dir)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print(f"No PDF files found in {args.input_pdf_dir}")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDFs. Converting at {args.dpi} DPI...")

    total_pages = 0
    failed = []

    for i, fname in enumerate(sorted(pdf_files), 1):
        pdf_path = os.path.join(args.input_pdf_dir, fname)
        pages = convert_pdf_to_images(pdf_path, args.output_dir, dpi=args.dpi)
        if not pages:
            failed.append(fname)
        else:
            total_pages += len(pages)
        if i % 10 == 0 or i == len(pdf_files):
            print(f"  Processed {i}/{len(pdf_files)} PDFs...")

    print("\n--- Summary ---")
    print(f"PDFs processed: {len(pdf_files) - len(failed)}/{len(pdf_files)}")
    print(f"Total page images created: {total_pages}")
    if failed:
        print(f"Failed to convert ({len(failed)}):")
        for f in failed:
            print(f"  - {f}")
    print(f"\nOutput saved under: {args.output_dir}")


if __name__ == "__main__":
    main()