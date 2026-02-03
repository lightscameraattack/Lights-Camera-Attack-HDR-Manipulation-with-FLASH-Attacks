import cv2
import numpy as np
import os
from typing import List, Optional, Tuple
from hdr_pipeline_organized import ModularHDR as BaseModularHDR, LocalTonemapper

class ModularHDR(BaseModularHDR):

    def process(self, image_paths: List[str], exposures: Optional[List[float]]=None, output_dir: str='hdr_outputs', save_intermediate: bool=False) -> Tuple[np.ndarray, dict]:
        print(f'Processing {len(image_paths)} images with methods:')
        print(f'  Alignment: {self.alignment_method}')
        print(f'  Merging: {self.merging_method}')
        print(f'  Tone mapping: {self.tonemapping_method}')
        if save_intermediate:
            dirs = self._create_output_directories(output_dir)
            print(f"Output directory: {dirs['base']}")
        else:
            dirs = {'base': output_dir}
            print(f'Processing in memory only')
        print('Loading images...')
        images = self.load_images(image_paths)
        if save_intermediate:
            self._save_intermediate_results(dirs, images, exposures)
        print('Aligning images...')
        aligned_images = self.align(images)
        if save_intermediate:
            for i, img in enumerate(aligned_images):
                aligned_path = os.path.join(dirs['aligned'], f'aligned_{i + 1:02d}.jpg')
                cv2.imwrite(aligned_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print('Merging into HDR...')
        if self.merging_method in ['robertson', 'debevec']:
            if exposures is None:
                raise ValueError(f'Exposure times required for {self.merging_method} method')
            if len(exposures) != len(aligned_images):
                raise ValueError(f'Number of exposure times ({len(exposures)}) must match number of images ({len(aligned_images)})')
        hdr_image = self.merge(aligned_images, exposures)
        if save_intermediate:
            hdr_path = os.path.join(dirs['hdr'], 'hdr_image.hdr')
            cv2.imwrite(hdr_path, cv2.cvtColor(hdr_image, cv2.COLOR_RGB2BGR))
        print('Applying tone mapping...')
        ldr_image = self.tonemap(hdr_image)
        if save_intermediate:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            method_name = f'{self.alignment_method}_{self.merging_method}_{self.tonemapping_method}'
            filename = f'{method_name}_{timestamp}.jpg'
            final_path = os.path.join(dirs['final'], filename)
            cv2.imwrite(final_path, ldr_image)
            print(f'Final result saved: {final_path}')
        else:
            print('Final result generated (not saved)')
        print('Processing complete!')
        return (ldr_image, dirs)

    def _create_output_directories(self, base_output_dir: str='hdr_outputs'):
        os.makedirs(base_output_dir, exist_ok=True)
        dirs = {'base': base_output_dir, 'inputs': os.path.join(base_output_dir, 'input_images'), 'aligned': os.path.join(base_output_dir, 'aligned_images'), 'hdr': os.path.join(base_output_dir, 'hdr_images'), 'final': os.path.join(base_output_dir, 'final_results'), 'comparisons': os.path.join(base_output_dir, 'comparisons'), 'logs': os.path.join(base_output_dir, 'logs')}
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        return dirs

    def _save_intermediate_results(self, dirs: dict, images: List[np.ndarray], exposures: Optional[List[float]]):
        for i, img in enumerate(images):
            input_path = os.path.join(dirs['inputs'], f'input_{i + 1:02d}.jpg')
            cv2.imwrite(input_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if exposures:
            exp_path = os.path.join(dirs['logs'], 'exposure_times.txt')
            with open(exp_path, 'w') as f:
                f.write(','.join(map(str, exposures)))

    def process_generated(self, generated_images: List[np.ndarray], exposures: List[float], output_dir: str='hdr_outputs', save_intermediate: bool=False) -> Tuple[np.ndarray, dict]:
        print(f'Processing {len(generated_images)} generated images with methods:')
        print(f'  Alignment: {self.alignment_method}')
        print(f'  Merging: {self.merging_method}')
        print(f'  Tone mapping: {self.tonemapping_method}')
        if save_intermediate:
            dirs = self._create_output_directories(output_dir)
            print(f"Output directory: {dirs['base']}")
        else:
            dirs = {'base': output_dir}
            print(f'Processing in memory only')
        if save_intermediate:
            for i, img in enumerate(generated_images):
                input_path = os.path.join(dirs['inputs'], f'generated_{i + 1:02d}.jpg')
                cv2.imwrite(input_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            exp_path = os.path.join(dirs['logs'], 'generated_exposure_times.txt')
            with open(exp_path, 'w') as f:
                f.write(','.join(map(str, exposures)))
        print('Aligning images...')
        aligned_images = self.align(generated_images)
        if save_intermediate:
            for i, img in enumerate(aligned_images):
                aligned_path = os.path.join(dirs['aligned'], f'aligned_{i + 1:02d}.jpg')
                cv2.imwrite(aligned_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print('Merging into HDR...')
        if self.merging_method in ['robertson', 'debevec']:
            if exposures is None:
                raise ValueError(f'Exposure times required for {self.merging_method} method')
            if len(exposures) != len(aligned_images):
                raise ValueError(f'Number of exposure times ({len(exposures)}) must match number of images ({len(aligned_images)})')
        hdr_image = self.merge(aligned_images, exposures)
        if save_intermediate:
            hdr_path = os.path.join(dirs['hdr'], 'hdr_image.hdr')
            cv2.imwrite(hdr_path, cv2.cvtColor(hdr_image, cv2.COLOR_RGB2BGR))
        print('Applying tone mapping...')
        ldr_image = self.tonemap(hdr_image)
        if save_intermediate:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            method_name = f'{self.alignment_method}_{self.merging_method}_{self.tonemapping_method}'
            filename = f'{method_name}_generated_{timestamp}.jpg'
            final_path = os.path.join(dirs['final'], filename)
            cv2.imwrite(final_path, ldr_image)
            print(f'Final result saved: {final_path}')
        else:
            print('Final result generated (not saved)')
        print('Processing complete!')
        return (ldr_image, dirs)