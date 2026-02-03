
import argparse
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional, Union
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import warnings
warnings.filterwarnings('ignore')
class ModularHDR:

    def __init__(self, alignment_method: str = 'mtb', merging_method: str = 'debevec', 
                 tonemapping_method: str = 'reinhard'):
        self.alignment_method = alignment_method
        self.merging_method = merging_method
        self.tonemapping_method = tonemapping_method
        self.aligner = self._create_aligner()
        self.merger = self._create_merger()
        self.tonemapper = self._create_tonemapper()
    def _create_aligner(self):
        if self.alignment_method == 'mtb':
            return cv2.createAlignMTB()
        elif self.alignment_method == 'homography':
            return None  
        else:
            raise ValueError(f"Unknown alignment method: {self.alignment_method}")
    def _create_merger(self):
        if self.merging_method == 'robertson':
            return cv2.createMergeRobertson()
        elif self.merging_method == 'debevec':
            return cv2.createMergeDebevec()
        elif self.merging_method == 'mertens':
            return cv2.createMergeMertens()
        else:
            raise ValueError(f"Unknown merging method: {self.merging_method}")
    def _create_tonemapper(self):
        if self.tonemapping_method == 'mantiuk':
            return cv2.createTonemapMantiuk()
        elif self.tonemapping_method == 'reinhard':
            return cv2.createTonemapReinhard()
        elif self.tonemapping_method == 'drago':
            return cv2.createTonemapDrago()
        elif self.tonemapping_method == 'local':
            return LocalTonemapper()
        else:
            raise ValueError(f"Unknown tone mapping method: {self.tonemapping_method}")
    def _create_output_directories(self, base_output_dir: str = "hdr_outputs"):
        os.makedirs(base_output_dir, exist_ok=True)
        dirs = {
            'base': base_output_dir,
            'inputs': os.path.join(base_output_dir, 'input_images'),
            'aligned': os.path.join(base_output_dir, 'aligned_images'),
            'hdr': os.path.join(base_output_dir, 'hdr_images'),
            'results': os.path.join(base_output_dir, 'final_results'),
            'comparisons': os.path.join(base_output_dir, 'comparisons'),
            'logs': os.path.join(base_output_dir, 'logs')
        }
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        return dirs
    def _get_output_filename(self, dirs: dict, method_name: str, suffix: str = "", 
                           extension: str = "jpg") -> str:
        clean_name = method_name.replace(' ', '_').replace('/', '_')
        if not suffix:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = f"_{timestamp}"
        filename = f"{clean_name}{suffix}.{extension}"
        return os.path.join(dirs['results'], filename)
    def _save_intermediate_results(self, dirs: dict, images: List[np.ndarray], 
                                 exposures: Optional[List[float]] = None):
        for i, img in enumerate(images):
            input_path = os.path.join(dirs['inputs'], f"input_{i+1:02d}.jpg")
            cv2.imwrite(input_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if exposures:
            exp_path = os.path.join(dirs['logs'], "exposure_times.txt")
            with open(exp_path, 'w') as f:
                f.write(','.join(map(str, exposures)))
    def load_images(self, image_paths: List[str]) -> List[np.ndarray]:
        images = []
        for path in image_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image file not found: {path}")
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"Could not load image: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
        return images
    def align(self, images: List[np.ndarray]) -> List[np.ndarray]:
        if self.alignment_method == 'mtb':
            return self._align_mtb(images)
        elif self.alignment_method == 'homography':
            return self._align_homography(images)
        else:
            raise ValueError(f"Unknown alignment method: {self.alignment_method}")
    def _align_mtb(self, images: List[np.ndarray]) -> List[np.ndarray]:
        images_bgr = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images]
        aligned_images_bgr = [img.copy() for img in images_bgr]
        self.aligner.process(images_bgr, aligned_images_bgr)
        aligned_images = [cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) for img_bgr in aligned_images_bgr]
        return aligned_images
    def _align_homography(self, images: List[np.ndarray]) -> List[np.ndarray]:
        if len(images) < 2:
            return images
        reference = images[0]
        reference_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
        orb = cv2.ORB_create()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        kp_ref, des_ref = orb.detectAndCompute(reference_gray, None)
        aligned_images = [reference]
        for img in images[1:]:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            kp_img, des_img = orb.detectAndCompute(img_gray, None)
            if des_ref is not None and des_img is not None:
                matches = bf.match(des_ref, des_img)
                matches = sorted(matches, key=lambda x: x.distance)
                src_pts = np.float32([kp_ref[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_img[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
                if len(src_pts) >= 4:
                    homography, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                    if homography is not None:
                        h, w = reference.shape[:2]
                        aligned = cv2.warpPerspective(img, homography, (w, h))
                        aligned_images.append(aligned)
                    else:
                        aligned_images.append(img)
                else:
                    aligned_images.append(img)
            else:
                aligned_images.append(img)
        return aligned_images
    def merge(self, images: List[np.ndarray], exposures: Optional[List[float]] = None) -> np.ndarray:
        if self.merging_method in ['robertson', 'debevec'] and exposures is None:
            raise ValueError(f"Exposure times required for {self.merging_method} method")
        if self.merging_method == 'mertens':
            return self._merge_mertens(images)
        else:
            return self._merge_with_exposures(images, exposures)
    def _merge_mertens(self, images: List[np.ndarray]) -> np.ndarray:
        images_bgr = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images]
        hdr = self.merger.process(images_bgr)
        hdr_rgb = cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB)
        return hdr_rgb
    def _merge_with_exposures(self, images: List[np.ndarray], exposures: List[float]) -> np.ndarray:
        images_bgr = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images]
        exposures_array = np.array(exposures, dtype=np.float32)
        hdr = self.merger.process(images_bgr, exposures_array)
        hdr_rgb = cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB)
        return hdr_rgb
    def tonemap(self, hdr_image: np.ndarray) -> np.ndarray:
        if self.tonemapping_method == 'local':
            return self.tonemapper.process(hdr_image)
        else:
            return self._tonemap_opencv(hdr_image)
    def _tonemap_opencv(self, hdr_image: np.ndarray) -> np.ndarray:
        hdr_bgr = cv2.cvtColor(hdr_image, cv2.COLOR_RGB2BGR)
        ldr = self.tonemapper.process(hdr_bgr)
        ldr_rgb = cv2.cvtColor(ldr, cv2.COLOR_BGR2RGB)
        ldr_rgb = np.clip(ldr_rgb * 255, 0, 255).astype(np.uint8)
        return ldr_rgb
    def process(self, image_paths: List[str], exposures: Optional[List[float]] = None, 
                 output_dir: str = "hdr_outputs", save_intermediate: bool = False) -> Tuple[np.ndarray, dict]:
        print(f"Processing {len(image_paths)} images with methods:")
        print(f"  Alignment: {self.alignment_method}")
        print(f"  Merging: {self.merging_method}")
        print(f"  Tone mapping: {self.tonemapping_method}")
        dirs = self._create_output_directories(output_dir)
        print(f"Output directory: {dirs['base']}")
        print("Loading images...")
        images = self.load_images(image_paths)
        if save_intermediate:
            self._save_intermediate_results(dirs, images, exposures)
        print("Aligning images...")
        aligned_images = self.align(images)
        if save_intermediate:
            for i, img in enumerate(aligned_images):
                aligned_path = os.path.join(dirs['aligned'], f"aligned_{i+1:02d}.jpg")
                cv2.imwrite(aligned_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print("Merging into HDR...")
        hdr_image = self.merge(aligned_images, exposures)
        if save_intermediate:
            hdr_path = os.path.join(dirs["hdr"], "hdr_image.hdr")
            cv2.imwrite(hdr_path, cv2.cvtColor(hdr_image, cv2.COLOR_RGB2BGR))
        print("Applying tone mapping...")
        ldr_image = self.tonemap(hdr_image)
        method_name = f"{self.alignment_method}_{self.merging_method}_{self.tonemapping_method}"
        output_path = self._get_output_filename(dirs, method_name)
        cv2.imwrite(output_path, cv2.cvtColor(ldr_image, cv2.COLOR_RGB2BGR))
        print(f"Final result saved: {output_path}")
        print("Processing complete!")
        return ldr_image, dirs
class LocalTonemapper:
    def __init__(self, alpha: float = 0.18, phi: float = 8.0, epsilon: float = 0.05):
        self.alpha = alpha
        self.phi = phi
        self.epsilon = epsilon
    def process(self, hdr_image: np.ndarray) -> np.ndarray:
        img = hdr_image.astype(np.float32)
        luminance = self._rgb_to_luminance(img)
        tone_mapped_luminance = self._local_tone_mapping(luminance)
        tone_mapped_image = self._scale_colors(img, luminance, tone_mapped_luminance)
        tone_mapped_image = np.clip(tone_mapped_image * 255, 0, 255).astype(np.uint8)
        return tone_mapped_image
    def _rgb_to_luminance(self, img: np.ndarray) -> np.ndarray:
        return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    def _local_tone_mapping(self, luminance: np.ndarray) -> np.ndarray:
        log_luminance = np.log(luminance + 1e-6)
        key_value = np.exp(np.mean(log_luminance))
        normalized_luminance = self.alpha / key_value * luminance
        filtered_luminance = cv2.bilateralFilter(
            normalized_luminance.astype(np.float32), 
            d=9, 
            sigmaColor=75, 
            sigmaSpace=75
        )
        local_contrast = normalized_luminance / (filtered_luminance + 1e-6)
        tone_mapped = normalized_luminance / (1 + normalized_luminance)
        tone_mapped = tone_mapped * np.power(local_contrast, self.phi)
        return tone_mapped
    def _scale_colors(self, img: np.ndarray, luminance: np.ndarray, 
                     tone_mapped_luminance: np.ndarray) -> np.ndarray:
        luminance = np.maximum(luminance, 1e-6)
        scale = tone_mapped_luminance / luminance
        tone_mapped_image = img.copy()
        for i in range(3):
            tone_mapped_image[:, :, i] = img[:, :, i] * scale
        return tone_mapped_image
def parse_exposures(exposure_input: str) -> List[float]:
    if os.path.exists(exposure_input):
        with open(exposure_input, 'r') as f:
            content = f.read().strip()
        exposures = [float(x.strip()) for x in content.split(',')]
    else:
        exposures = [float(x.strip()) for x in exposure_input.split(',')]
    return exposures
def compare_results(results: dict, output_dir: str = "comparison", 
                     comparison_name: str = "comparison"):
    os.makedirs(output_dir, exist_ok=True)
    n_results = len(results)
    fig, axes = plt.subplots(1, n_results, figsize=(5 * n_results, 5))
    if n_results == 1:
        axes = [axes]
    for i, (method, image) in enumerate(results.items()):
        axes[i].imshow(image)
        axes[i].set_title(f"{method}")
        axes[i].axis('off')
    plt.tight_layout()
    comparison_plot_path = os.path.join(output_dir, f"{comparison_name}.png")
    plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved: {comparison_plot_path}")
    for method, image in results.items():
        method_dir = os.path.join(output_dir, "individual_results")
        os.makedirs(method_dir, exist_ok=True)
        result_path = os.path.join(method_dir, f"{method}.jpg")
        cv2.imwrite(result_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"Individual result saved: {result_path}")
    if len(results) > 1:
        print("\nQuantitative Comparison:")
        print("-" * 50)
        metrics_path = os.path.join(output_dir, f"{comparison_name}_metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write(f"HDR Pipeline Comparison Metrics\n")
            f.write(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            reference_method = list(results.keys())[0]
            reference_image = results[reference_method]
            f.write(f"Reference method: {reference_method}\n\n")
            for method, image in results.items():
                if method == reference_method:
                    continue
                ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_RGB2GRAY)
                img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                ssim_value = ssim(ref_gray, img_gray, data_range=255)
                psnr_value = psnr(ref_gray, img_gray, data_range=255)
                print(f"{method} vs {reference_method}:")
                print(f"  SSIM: {ssim_value:.4f}")
                print(f"  PSNR: {psnr_value:.2f} dB")
                print()
                f.write(f"{method} vs {reference_method}:\n")
                f.write(f"  SSIM: {ssim_value:.4f}\n")
                f.write(f"  PSNR: {psnr_value:.2f} dB\n\n")
        print(f"Metrics saved: {metrics_path}")
def main():
    parser = argparse.ArgumentParser(description="Modular HDR Imaging Pipeline with Organized Folders")
    parser.add_argument('--images', required=True, 
                       help='Comma-separated list of image files or directory path')
    parser.add_argument('--exposures', 
                       help='Comma-separated exposure times or file path containing exposures')
    parser.add_argument('--alignment', choices=['mtb', 'homography'], default='mtb',
                       help='Image alignment method')
    parser.add_argument('--merging', choices=['robertson', 'debevec', 'mertens'], default='debevec',
                       help='HDR merging method')
    parser.add_argument('--tonemapping', choices=['mantiuk', 'reinhard', 'drago', 'local'], 
                       default='reinhard', help='Tone mapping method')
    parser.add_argument('--output', default='output.jpg',
                       help='Output file path')
    parser.add_argument('--output-dir', default='hdr_outputs',
                       help='Base directory for organized outputs')
    parser.add_argument('--save-intermediate', action='store_true',
                       help='Save intermediate processing results')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple methods if multiple are specified')
    parser.add_argument('--alignment-methods', nargs='+',
                       help='Multiple alignment methods to compare')
    parser.add_argument('--merging-methods', nargs='+',
                       help='Multiple merging methods to compare')
    parser.add_argument('--tonemapping-methods', nargs='+',
                       help='Multiple tone mapping methods to compare')
    args = parser.parse_args()
    if os.path.isdir(args.images):
        image_files = sorted([str(f) for f in Path(args.images).glob('*') 
                             if f.suffix.lower() in ['.jpg', '.jpeg', '.tiff', '.tif', '.png']])
    else:
        image_files = [f.strip() for f in args.images.split(',')]
    if not image_files:
        print("Error: No valid image files found")
        sys.exit(1)
    print(f"Found {len(image_files)} images: {image_files}")
    exposures = None
    if args.exposures:
        exposures = parse_exposures(args.exposures)
        print(f"Exposure times: {exposures}")
    output_base_dir = args.output_dir
    if args.compare or any([args.alignment_methods, args.merging_methods, args.tonemapping_methods]):
        output_base_dir = f"{output_base_dir}_comparison"
    if args.compare or any([args.alignment_methods, args.merging_methods, args.tonemapping_methods]):
        results = {}
        alignment_methods = args.alignment_methods or [args.alignment]
        merging_methods = args.merging_methods or [args.merging]
        tonemapping_methods = args.tonemapping_methods or [args.tonemapping]
        for align_method in alignment_methods:
            for merge_method in merging_methods:
                for tone_method in tonemapping_methods:
                    method_name = f"{align_method}_{merge_method}_{tone_method}"
                    try:
                        pipeline = ModularHDR(align_method, merge_method, tone_method)
                        result, dirs = pipeline.process(image_files, exposures, output_base_dir, save_intermediate=False)
                        results[method_name] = result
                        print(f"Processed {method_name}")
                    except Exception as e:
                        print(f"Error processing {method_name}: {e}")
        if len(results) > 1:
            comparison_dir = os.path.join(output_base_dir, "comparisons")
            compare_results(results, comparison_dir, "method_comparison")
        if results:
            main_result = list(results.values())[0]
            main_output_path = os.path.join(output_base_dir, "final_results", args.output)
            cv2.imwrite(main_output_path, cv2.cvtColor(main_result, cv2.COLOR_RGB2BGR))
            print(f"Main output saved: {main_output_path}")
    else:
        try:
            pipeline = ModularHDR(args.alignment, args.merging, args.tonemapping)
            result, dirs = pipeline.process(image_files, exposures, output_base_dir, save_intermediate=args.save_intermediate)
            cv2.imwrite(args.output, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            print(f"Output also saved to: {args.output}")
        except Exception as e:
            print(f"Error processing images: {e}")
            sys.exit(1)
if __name__ == "__main__":
    main()
