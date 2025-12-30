#!/usr/bin/env python3
"""
Convert Existing PNG Figures to PDFs for LaTeX

This script converts the large PNG figures (~9MB each) to optimized PDFs
that are much smaller and better suited for LaTeX documents.

Usage:
    cd aramco_security_opt
    python src/convert_figures_to_pdf.py
"""

from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORT_FIGURES_DIR = PROJECT_ROOT / "report" / "figures"



def convert_png_to_pdf_optimized():
    """
    Convert PNG figures to optimized PDFs using Pillow.
    
    This produces raster PDFs from the existing PNGs.
    For true vector PDFs, re-run the optimization with save_format='pdf'.
    """
    try:
        from PIL import Image
    except ImportError:
        print("Error: Pillow is required. Install with: pip install Pillow")
        return []
    
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    png_files = list(FIGURES_DIR.glob("*.png"))
    
    if not png_files:
        print("No PNG files found in figures directory.")
        print(f"  Searched in: {FIGURES_DIR}")
        return []
    
    print(f"Found {len(png_files)} PNG files to convert...")
    print(f"Output directories:")
    print(f"  - Results: {FIGURES_DIR}")
    print(f"  - Report:  {REPORT_FIGURES_DIR}")
    print("-" * 60)
    
    pdf_paths = []
    total_png_size = 0
    total_pdf_size = 0
    
    for png_path in sorted(png_files):
        pdf_filename = png_path.stem + ".pdf"
        pdf_path = FIGURES_DIR / pdf_filename
        report_pdf_path = REPORT_FIGURES_DIR / pdf_filename
        
        try:
            img = Image.open(png_path)
            
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            max_width = 2000
            if img.width > max_width:
                ratio = max_width / img.width
                new_size = (max_width, int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Save as PDF to results/figures
            img.save(pdf_path, 'PDF', resolution=150, optimize=True)
            
            # Also save to report/figures
            img.save(report_pdf_path, 'PDF', resolution=150, optimize=True)
            
            # Calculate sizes
            png_size = png_path.stat().st_size / (1024 * 1024)
            pdf_size = pdf_path.stat().st_size / (1024 * 1024)
            reduction = (1 - pdf_size / png_size) * 100
            
            total_png_size += png_size
            total_pdf_size += pdf_size
            
            print(f"✓ {png_path.name}")
            print(f"  PNG: {png_size:.2f} MB → PDF: {pdf_size:.2f} MB ({reduction:.1f}% smaller)")
            pdf_paths.append(str(pdf_path))
            
        except Exception as e:
            print(f"✗ Error converting {png_path.name}: {e}")
    
    print("-" * 60)
    total_reduction = (1 - total_pdf_size / total_png_size) * 100 if total_png_size > 0 else 0
    print(f"TOTAL: {total_png_size:.2f} MB → {total_pdf_size:.2f} MB ({total_reduction:.1f}% reduction)")
    print(f"\nPDF files saved to:")
    print(f"  - {FIGURES_DIR}")
    print(f"  - {REPORT_FIGURES_DIR}")
    
    return pdf_paths


if __name__ == "__main__":
    print("=" * 60)
    print("PNG to PDF Converter for LaTeX Figures")
    print("=" * 60)
    convert_png_to_pdf_optimized()
