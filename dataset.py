import torch
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import json
import numpy as np
import pickle
from typing import Dict, List, Union

try:
    from pycocotools import mask as coco_mask
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    print("Warning: pycocotools not available. Install with: pip install pycocotools")

class ViTASimDataset(Dataset):
    def __init__(self, root_dir: str, annotation_file: str, split: str = 'train', 
                 fusion_file: str = None, target_size=(224, 224)):
        """
        Dataset for ViTASim amodal segmentation data
        
        Args:
            root_dir: Root directory containing the ViTASimData folder
            annotation_file: Path to the JSON annotation file (COCO format)
            split: 'train' or 'val'
            fusion_file: Path to the fusion pickle file (optional, for additional data)
            target_size: Target size for images (H, W)
        """
        self.root_dir = root_dir
        self.split = split
        self.target_size = target_size
        
        # Load COCO-format annotations
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create image id to filename mapping
        self.images = {img['id']: img for img in coco_data['images']}
        
        # Filter annotations by split if needed
        self.annotations = coco_data['annotations']
        
        # Load fusion data if provided
        self.fusion_data = None
        if fusion_file and os.path.exists(fusion_file):
            with open(fusion_file, 'rb') as f:
                self.fusion_data = pickle.load(f)
        
        print(f"Loaded {len(self.annotations)} annotations for {split} split")
        if self.fusion_data:
            print(f"Loaded fusion data with {len(self.fusion_data)} entries")
    
    def decode_rle_mask(self, rle_data: Dict) -> np.ndarray:
        """
        Decode RLE mask to binary numpy array
        
        Args:
            rle_data: Dictionary with 'size' and 'counts' keys
            
        Returns:
            Binary mask as numpy array
        """
        if PYCOCOTOOLS_AVAILABLE:
            return coco_mask.decode(rle_data)
        else:
            # Fallback implementation for RLE decoding
            return self._decode_rle_fallback(rle_data)
    
    def _decode_rle_fallback(self, rle_data: Dict) -> np.ndarray:
        """
        Fallback RLE decoder when pycocotools is not available
        """
        h, w = rle_data['size']
        counts = rle_data['counts']
        
        if isinstance(counts, str):
            # Compressed RLE string - simplified decoding
            mask = np.zeros(h * w, dtype=np.uint8)
            return mask.reshape(h, w)
        else:
            # Uncompressed RLE
            mask = np.zeros(h * w, dtype=np.uint8)
            idx = 0
            flag = 0
            for count in counts:
                mask[idx:idx+count] = flag
                idx += count
                flag = 1 - flag
            return mask.reshape(h, w)

    def crop_and_resize(self, image: Image.Image, visible_mask: np.ndarray, 
                       amodal_mask: np.ndarray, bbox: List[float]) -> tuple:
        """
        Crop image and masks according to bbox and resize to target size
        
        Args:
            image: PIL Image
            visible_mask: Numpy array for visible mask
            amodal_mask: Numpy array for amodal mask
            bbox: [x, y, width, height] in COCO format
            
        Returns:
            Tuple of (cropped_image, cropped_visible_mask, cropped_amodal_mask)
        """
        # Convert COCO bbox format [x, y, w, h] to [x1, y1, x2, y2]
        x, y, w, h = bbox
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        
        # Add padding for context (20% of bbox size)
        pad_factor = 0.2
        pad_w = int(w * pad_factor)
        pad_h = int(h * pad_factor)
        
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(image.size[0], x2 + pad_w)
        y2 = min(image.size[1], y2 + pad_h)
        
        # Crop image
        image_crop = image.crop((x1, y1, x2, y2))
        
        # Crop masks
        visible_mask_crop = visible_mask[y1:y2, x1:x2]
        amodal_mask_crop = amodal_mask[y1:y2, x1:x2]
        
        # Convert masks to PIL for resizing
        visible_mask_pil = Image.fromarray(visible_mask_crop.astype(np.uint8))
        amodal_mask_pil = Image.fromarray(amodal_mask_crop.astype(np.uint8))
        
        # Resize to target size
        image_resized = image_crop.resize(self.target_size, Image.BILINEAR)
        visible_mask_resized = visible_mask_pil.resize(self.target_size, Image.NEAREST)
        amodal_mask_resized = amodal_mask_pil.resize(self.target_size, Image.NEAREST)
        
        return image_resized, np.array(visible_mask_resized), np.array(amodal_mask_resized)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        annotation = self.annotations[idx]
        
        # Get image info
        image_id = annotation['image_id']
        image_info = self.images[image_id]
        
        # Load image - try multiple path structures
        # First try: root_dir/ViTASimData/split2014/filename (old structure)
        img_path = os.path.join(self.root_dir, 'ViTASimData', f"{self.split}2014", image_info['file_name'])
        if not os.path.exists(img_path):
            # Second try: root_dir/split2014/filename (direct structure, for ViTASimDataArol)
            img_path = os.path.join(self.root_dir, f"{self.split}2014", image_info['file_name'])
        if not os.path.exists(img_path):
            # Third try: root_dir/train2014 or val2014 based on filename prefix
            # This handles COCOA-style structure where both train and val are in same folder
            filename_split = 'train' if 'train' in image_info['file_name'].lower() else 'val'
            img_path = os.path.join(self.root_dir, f"{filename_split}2014", image_info['file_name'])
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        image = Image.open(img_path).convert('RGB')
        
        # Decode masks from RLE format
        amodal_mask = self.decode_rle_mask(annotation['segmentation'])
        visible_mask = self.decode_rle_mask(annotation['visible_mask'])
        
        # Get bbox and crop/resize
        bbox = annotation['bbox']
        image_crop, visible_mask_crop, amodal_mask_crop = self.crop_and_resize(
            image, visible_mask, amodal_mask, bbox
        )
        
        # Convert to tensors
        # Image: normalize to [0, 1] and apply ImageNet normalization
        image_tensor = transforms.ToTensor()(image_crop)
        # Apply ImageNet normalization
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image_tensor = normalize(image_tensor)
        
        # Masks: convert to float tensors - ALREADY in [0, 1] from decode_rle_mask
        # DO NOT divide by 255 since decode returns binary {0, 1} values
        visible_mask_tensor = torch.from_numpy(visible_mask_crop).float().unsqueeze(0)
        amodal_mask_tensor = torch.from_numpy(amodal_mask_crop).float().unsqueeze(0)
        
        return {
            'image': image_tensor,
            'visible_mask': visible_mask_tensor,
            'amodal_mask': amodal_mask_tensor,
            'bbox': torch.tensor(bbox, dtype=torch.float32),
            'category_id': annotation['category_id'],
            'image_id': image_id,
            'annotation_id': annotation['id'],
            'area': annotation['area'],
            'iscrowd': annotation.get('iscrowd', 0),
            'occlude_rate': annotation.get('occlude_rate', 0.0),
            'filename': image_info['file_name'],
            'dataset': 'vitasim'  # Add dataset identifier for consistency
        }

    def __len__(self) -> int:
        return len(self.annotations)


class COCOADataset(Dataset):
    def __init__(self, root_dir: str, annotation_file: str, split: str = 'train', 
                 target_size=(224, 224)):
        """
        Dataset for COCOA amodal segmentation data
        
        Args:
            root_dir: Root directory containing the COCOA dataset
            annotation_file: Path to the JSON annotation file (COCO format with amodal annotations)
            split: 'train' or 'val'
            target_size: Target size for images (H, W)
        """
        self.root_dir = root_dir
        self.split = split
        self.target_size = target_size
        
        # Load COCO-format annotations
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create image id to filename mapping
        self.images = {img['id']: img for img in coco_data['images']}
        
        # Filter annotations by split if needed
        self.annotations = coco_data['annotations']
        
        print(f"Loaded {len(self.annotations)} COCOA annotations for {split} split")
    
    def decode_segmentation(self, segmentation_data) -> np.ndarray:
        """
        Decode segmentation mask from various formats (RLE, polygon, etc.)
        
        Args:
            segmentation_data: Segmentation data (can be RLE or polygon)
            
        Returns:
            Binary mask as numpy array
        """
        if isinstance(segmentation_data, dict):
            # RLE format
            if PYCOCOTOOLS_AVAILABLE:
                return coco_mask.decode(segmentation_data)
            else:
                return self._decode_rle_fallback(segmentation_data)
        elif isinstance(segmentation_data, list):
            # Polygon format
            # For now, we'll create an empty mask - you may need to implement polygon to mask conversion
            return np.zeros((self.target_size[0], self.target_size[1]), dtype=np.uint8)
        else:
            # Unknown format
            return np.zeros((self.target_size[0], self.target_size[1]), dtype=np.uint8)
    
    def _decode_rle_fallback(self, rle_data: Dict) -> np.ndarray:
        """
        Fallback RLE decoder when pycocotools is not available
        """
        h, w = rle_data['size']
        counts = rle_data['counts']
        
        if isinstance(counts, str):
            # Compressed RLE string - simplified decoding
            mask = np.zeros(h * w, dtype=np.uint8)
            return mask.reshape(h, w)
        else:
            # Uncompressed RLE
            mask = np.zeros(h * w, dtype=np.uint8)
            idx = 0
            flag = 0
            for count in counts:
                mask[idx:idx+count] = flag
                idx += count
                flag = 1 - flag
            return mask.reshape(h, w)

    def crop_and_resize(self, image: Image.Image, visible_mask: np.ndarray, 
                       amodal_mask: np.ndarray, bbox: List[float]) -> tuple:
        """
        Crop image and masks according to bbox and resize to target size
        """
        # Convert COCO bbox format [x, y, w, h] to [x1, y1, x2, y2]
        x, y, w, h = bbox
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        
        # Add padding for context (20% of bbox size)
        pad_factor = 0.2
        pad_w = int(w * pad_factor)
        pad_h = int(h * pad_factor)
        
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(image.size[0], x2 + pad_w)
        y2 = min(image.size[1], y2 + pad_h)
        
        # Crop image
        image_crop = image.crop((x1, y1, x2, y2))
        
        # Crop masks
        visible_mask_crop = visible_mask[y1:y2, x1:x2]
        amodal_mask_crop = amodal_mask[y1:y2, x1:x2]
        
        # Convert masks to PIL for resizing
        visible_mask_pil = Image.fromarray(visible_mask_crop.astype(np.uint8))
        amodal_mask_pil = Image.fromarray(amodal_mask_crop.astype(np.uint8))
        
        # Resize to target size
        image_resized = image_crop.resize(self.target_size, Image.BILINEAR)
        visible_mask_resized = visible_mask_pil.resize(self.target_size, Image.NEAREST)
        amodal_mask_resized = amodal_mask_pil.resize(self.target_size, Image.NEAREST)
        
        return image_resized, np.array(visible_mask_resized), np.array(amodal_mask_resized)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        annotation = self.annotations[idx]
        
        # Get image info
        image_id = annotation['image_id']
        image_info = self.images[image_id]
        
        # Load image - COCOA uses train2014/val2014 directories
        img_filename = image_info['file_name']
        img_path = os.path.join(self.root_dir, f'{self.split}2014', img_filename)
        
        # Fallback paths if the above doesn't work
        if not os.path.exists(img_path):
            img_path = os.path.join(self.root_dir, 'images', self.split, img_filename)
        if not os.path.exists(img_path):
            img_path = os.path.join(self.root_dir, self.split, img_filename)
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        image = Image.open(img_path).convert('RGB')
        
        # Decode masks
        # COCOA should have both 'segmentation' (amodal) and 'visible_mask' fields
        amodal_mask = self.decode_segmentation(annotation['segmentation'])
        
        # Handle visible mask - it might be in different fields
        if 'visible_mask' in annotation:
            visible_mask = self.decode_segmentation(annotation['visible_mask'])
        elif 'inmodal_seg' in annotation:
            visible_mask = self.decode_segmentation(annotation['inmodal_seg'])
        else:
            # Fallback: use amodal mask as visible mask
            visible_mask = amodal_mask.copy()
        
        # Get bbox and crop/resize
        bbox = annotation['bbox']
        image_crop, visible_mask_crop, amodal_mask_crop = self.crop_and_resize(
            image, visible_mask, amodal_mask, bbox
        )
        
        # Convert to tensors
        # Image: normalize to [0, 1] and apply ImageNet normalization
        image_tensor = transforms.ToTensor()(image_crop)
        # Apply ImageNet normalization
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image_tensor = normalize(image_tensor)
        
        # Masks: convert to float tensors - ALREADY in [0, 1] from decode_segmentation
        # DO NOT divide by 255 since decode returns binary {0, 1} values
        visible_mask_tensor = torch.from_numpy(visible_mask_crop).float().unsqueeze(0)
        amodal_mask_tensor = torch.from_numpy(amodal_mask_crop).float().unsqueeze(0)
        
        return {
            'image': image_tensor,
            'visible_mask': visible_mask_tensor,
            'amodal_mask': amodal_mask_tensor,
            'bbox': torch.tensor(bbox, dtype=torch.float32),
            'category_id': annotation['category_id'],
            'image_id': image_id,
            'annotation_id': annotation['id'],
            'area': annotation['area'],
            'iscrowd': annotation.get('iscrowd', 0),
            'occlude_rate': annotation.get('occlude_rate', 0.0),  # Add occlude_rate field for consistency
            'filename': image_info['file_name'],
            'dataset': 'cocoa'
        }

    def __len__(self) -> int:
        return len(self.annotations)


class KINSDataset(Dataset):
    """
    Dataset for KINS (KITTI INStance) amodal segmentation data
    
    KINS uses the same COCO annotation format but with different structure.
    Follows the same preprocessing as C2F-Seg for fair comparison.
    """
    def __init__(self, root_dir: str, annotation_file: str, split: str = 'train',
                 fusion_file: str = None, target_size=(224, 224), enlarge_coef=2.0):
        """
        Args:
            root_dir: Root directory containing KINS dataset
            annotation_file: Path to the JSON annotation file (e.g., update_train_2020.json)
            split: 'train' or 'test'
            fusion_file: Optional fusion pickle file for C2F-Seg predictions
            target_size: Target size for images (H, W)
            enlarge_coef: Bbox enlargement coefficient for cropping (default: 2.0)
        """
        self.root_dir = root_dir
        self.split = split
        self.target_size = target_size
        self.enlarge_coef = enlarge_coef
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            kins_data = json.load(f)
        
        # Create image dictionaries
        self.images = {img['id']: img for img in kins_data['images']}
        
        # Filter out invalid annotations (bbox with width or height <= 0)
        valid_annotations = []
        invalid_count = 0
        for ann in kins_data['annotations']:
            bbox = ann.get('i_bbox', [0, 0, 0, 0])
            # Check if bbox is valid (width and height > 0)
            if bbox[2] > 0 and bbox[3] > 0:
                valid_annotations.append(ann)
            else:
                invalid_count += 1
        
        self.annotations = valid_annotations
        
        if invalid_count > 0:
            print(f"  ⚠️  Filtered out {invalid_count} annotations with invalid bboxes")
        
        # Create annotation lookup by image_id for efficiency
        self.anns_by_image = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.anns_by_image:
                self.anns_by_image[img_id] = []
            self.anns_by_image[img_id].append(ann)
        
        # Image directory
        self.img_dir = os.path.join(root_dir, f"{split}ing", "image_2")
        
        # Load fusion data if provided
        self.fusion_data = None
        if fusion_file and os.path.exists(fusion_file):
            with open(fusion_file, 'rb') as f:
                self.fusion_data = pickle.load(f)
        
        print(f"Loaded {len(self.annotations)} KINS annotations for {split} split")
        print(f"  Images directory: {self.img_dir}")
    
    def decode_segmentation(self, segmentation, height, width):
        """
        Decode polygon or RLE segmentation to binary mask
        
        Args:
            segmentation: List of polygons or RLE dict
            height: Image height
            width: Image width
            
        Returns:
            Binary mask as numpy array (H, W) with values in {0, 1}
        """
        if not PYCOCOTOOLS_AVAILABLE:
            raise ImportError("pycocotools is required for KINS dataset. Install with: pip install pycocotools")
        
        if isinstance(segmentation, list):
            # Polygon format
            rles = coco_mask.frPyObjects(segmentation, height, width)
            rle = coco_mask.merge(rles)
        elif isinstance(segmentation, dict):
            # RLE format
            if isinstance(segmentation['counts'], list):
                rle = coco_mask.frPyObjects(segmentation, height, width)
            else:
                rle = segmentation
        else:
            raise ValueError(f"Unknown segmentation format: {type(segmentation)}")
        
        return coco_mask.decode(rle).astype(np.float32)
    
    def crop_and_resize(self, image: Image.Image, visible_mask: np.ndarray,
                       amodal_mask: np.ndarray, bbox: List[float]) -> tuple:
        """
        Crop image and masks according to bbox with padding and resize to target size
        Similar to C2F-Seg preprocessing
        
        Args:
            image: PIL Image
            visible_mask: Visible mask (inmodal) as numpy array
            amodal_mask: Amodal mask as numpy array
            bbox: Bounding box [x, y, width, height] in COCO format
            
        Returns:
            Tuple of (image_crop, visible_crop, amodal_crop)
        """
        W, H = image.size
        x, y, w, h = bbox
        
        # Handle invalid bboxes (width or height <= 0)
        if w <= 0 or h <= 0:
            # Return default crops of minimum size
            x1, y1, x2, y2 = 0, 0, min(10, W), min(10, H)
            image_crop = image.crop((x1, y1, x2, y2))
            visible_crop = visible_mask[y1:y2, x1:x2]
            amodal_crop = amodal_mask[y1:y2, x1:x2]
        else:
            # Calculate padded bbox using enlarge_coef
            pad_w = int(w * (self.enlarge_coef - 1) / 2)
            pad_h = int(h * (self.enlarge_coef - 1) / 2)
            
            x1 = max(0, int(x) - pad_w)
            y1 = max(0, int(y) - pad_h)
            x2 = min(W, int(x + w) + pad_w)
            y2 = min(H, int(y + h) + pad_h)
            
            # Ensure valid crop coordinates
            x1 = min(x1, W - 1)
            y1 = min(y1, H - 1)
            x2 = max(x1 + 1, min(x2, W))  # Ensure x2 > x1
            y2 = max(y1 + 1, min(y2, H))  # Ensure y2 > y1
            
            # Crop image and masks
            image_crop = image.crop((x1, y1, x2, y2))
            visible_crop = visible_mask[y1:y2, x1:x2]
            amodal_crop = amodal_mask[y1:y2, x1:x2]
        
        # Resize to target size
        image_crop = image_crop.resize(self.target_size, Image.BILINEAR)
        visible_crop = cv2.resize(visible_crop, self.target_size, interpolation=cv2.INTER_LINEAR)
        amodal_crop = cv2.resize(amodal_crop, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        return image_crop, visible_crop, amodal_crop
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample from the dataset
        
        Returns:
            Dictionary with keys: image, visible_mask, amodal_mask, bbox, etc.
        """
        annotation = self.annotations[idx]
        image_id = annotation['image_id']
        image_info = self.images[image_id]
        
        # Load image
        img_path = os.path.join(self.img_dir, image_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        H, W = image_info['height'], image_info['width']
        
        # Decode masks
        amodal_mask = self.decode_segmentation(annotation['a_segm'], H, W)
        visible_mask = self.decode_segmentation(annotation['i_segm'], H, W)
        
        # Get bbox (inmodal bbox) - already validated during init
        bbox = annotation['i_bbox']  # [x, y, width, height]
        
        # Crop and resize
        image_crop, visible_crop, amodal_crop = self.crop_and_resize(
            image, visible_mask, amodal_mask, bbox
        )
        
        # Convert to tensors
        image_tensor = transforms.ToTensor()(image_crop)
        # Apply ImageNet normalization
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image_tensor = normalize(image_tensor)
        
        # Masks: convert to float tensors
        visible_mask_tensor = torch.from_numpy(visible_crop).float().unsqueeze(0)
        amodal_mask_tensor = torch.from_numpy(amodal_crop).float().unsqueeze(0)
        
        return {
            'image': image_tensor,
            'visible_mask': visible_mask_tensor,
            'amodal_mask': amodal_mask_tensor,
            'bbox': torch.tensor(bbox, dtype=torch.float32),
            'category_id': annotation['category_id'],
            'image_id': image_id,
            'annotation_id': annotation['id'],
            'area': annotation.get('area', 0.0),
            'iscrowd': annotation.get('iscrowd', 0),
            'occlude_rate': annotation.get('occlude_rate', 0.0),
            'filename': image_info['file_name'],
            'dataset': 'kins'
        }
    
    def __len__(self) -> int:
        return len(self.annotations)


class D2SADataset(Dataset):
    def __init__(self, root_dir: str, annotation_file: str, split: str = 'train', 
                 target_size=(224, 224)):
        """
        Dataset for D2SA (D2S Amodal) amodal segmentation data
        
        D2SA is in COCO format with amodal annotations.
        
        Args:
            root_dir: Root directory containing the D2SA dataset
            annotation_file: Path to the JSON annotation file (COCO format with amodal annotations)
            split: 'train' or 'val'
            target_size: Target size for images (H, W)
        """
        self.root_dir = root_dir
        self.split = split
        self.target_size = target_size
        
        # Load COCO-format annotations
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create image id to filename mapping
        self.images = {img['id']: img for img in coco_data['images']}
        
        # Filter annotations by split if needed
        self.annotations = coco_data['annotations']
        
        print(f"Loaded {len(self.annotations)} D2SA annotations for {split} split")
    
    def decode_segmentation(self, segmentation_data) -> np.ndarray:
        """
        Decode segmentation mask from various formats (RLE, polygon, etc.)
        
        Args:
            segmentation_data: Segmentation data (can be RLE or polygon)
            
        Returns:
            Binary mask as numpy array
        """
        if isinstance(segmentation_data, dict):
            # RLE format
            if PYCOCOTOOLS_AVAILABLE:
                return coco_mask.decode(segmentation_data)
            else:
                return self._decode_rle_fallback(segmentation_data)
        elif isinstance(segmentation_data, list):
            # Polygon format - need image size to convert
            # For now, return empty mask - will be handled in __getitem__
            return None
        else:
            # Unknown format
            return np.zeros((self.target_size[0], self.target_size[1]), dtype=np.uint8)
    
    def _decode_rle_fallback(self, rle_data: Dict) -> np.ndarray:
        """
        Fallback RLE decoder when pycocotools is not available
        """
        h, w = rle_data['size']
        counts = rle_data['counts']
        
        if isinstance(counts, str):
            # Compressed RLE string - simplified decoding
            mask = np.zeros(h * w, dtype=np.uint8)
            return mask.reshape(h, w)
        else:
            # Uncompressed RLE
            mask = np.zeros(h * w, dtype=np.uint8)
            idx = 0
            flag = 0
            for count in counts:
                mask[idx:idx+count] = flag
                idx += count
                flag = 1 - flag
            return mask.reshape(h, w)
    
    def decode_polygon(self, segmentation, height, width):
        """
        Decode polygon segmentation to binary mask
        
        Args:
            segmentation: List of polygons
            height: Image height
            width: Image width
            
        Returns:
            Binary mask as numpy array (H, W)
        """
        if not PYCOCOTOOLS_AVAILABLE:
            # Fallback: create mask from polygon using cv2
            mask = np.zeros((height, width), dtype=np.uint8)
            for poly in segmentation:
                poly_points = np.array(poly).reshape((-1, 2)).astype(np.int32)
                cv2.fillPoly(mask, [poly_points], 1)
            return mask
        else:
            # Use pycocotools for accurate conversion
            rles = coco_mask.frPyObjects(segmentation, height, width)
            rle = coco_mask.merge(rles)
            return coco_mask.decode(rle)

    def crop_and_resize(self, image: Image.Image, visible_mask: np.ndarray, 
                       amodal_mask: np.ndarray, bbox: List[float]) -> tuple:
        """
        Crop image and masks according to bbox and resize to target size
        """
        # Convert COCO bbox format [x, y, w, h] to [x1, y1, x2, y2]
        x, y, w, h = bbox
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        
        # Add padding for context (20% of bbox size)
        pad_factor = 0.2
        pad_w = int(w * pad_factor)
        pad_h = int(h * pad_factor)
        
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(image.size[0], x2 + pad_w)
        y2 = min(image.size[1], y2 + pad_h)
        
        # Crop image
        image_crop = image.crop((x1, y1, x2, y2))
        
        # Crop masks
        visible_mask_crop = visible_mask[y1:y2, x1:x2]
        amodal_mask_crop = amodal_mask[y1:y2, x1:x2]
        
        # Convert masks to PIL for resizing
        visible_mask_pil = Image.fromarray(visible_mask_crop.astype(np.uint8))
        amodal_mask_pil = Image.fromarray(amodal_mask_crop.astype(np.uint8))
        
        # Resize to target size
        image_resized = image_crop.resize(self.target_size, Image.BILINEAR)
        visible_mask_resized = visible_mask_pil.resize(self.target_size, Image.NEAREST)
        amodal_mask_resized = amodal_mask_pil.resize(self.target_size, Image.NEAREST)
        
        return image_resized, np.array(visible_mask_resized), np.array(amodal_mask_resized)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        annotation = self.annotations[idx]
        
        # Get image info
        image_id = annotation['image_id']
        image_info = self.images[image_id]
        
        # Load image - D2SA should have images in root_dir/images/ or root_dir/<split>/
        img_filename = image_info['file_name']
        
        # Try multiple path structures
        possible_paths = [
            os.path.join(self.root_dir, 'images', self.split, img_filename),
            os.path.join(self.root_dir, 'images', img_filename),
            os.path.join(self.root_dir, self.split, img_filename),
            os.path.join(self.root_dir, img_filename),
        ]
        
        img_path = None
        for path in possible_paths:
            if os.path.exists(path):
                img_path = path
                break
        
        if img_path is None:
            raise FileNotFoundError(f"Image not found. Tried paths: {possible_paths}")
        
        image = Image.open(img_path).convert('RGB')
        H, W = image_info['height'], image_info['width']
        
        # Decode amodal mask
        amodal_seg = annotation['segmentation']
        if isinstance(amodal_seg, dict):
            # RLE format
            amodal_mask = self.decode_segmentation(amodal_seg)
        else:
            # Polygon format
            amodal_mask = self.decode_polygon(amodal_seg, H, W)
        
        # Decode visible mask
        # D2SA should have visible_mask or inmodal_seg field
        if 'visible_mask' in annotation:
            visible_seg = annotation['visible_mask']
            if isinstance(visible_seg, dict):
                visible_mask = self.decode_segmentation(visible_seg)
            else:
                visible_mask = self.decode_polygon(visible_seg, H, W)
        elif 'inmodal_seg' in annotation:
            inmodal_seg = annotation['inmodal_seg']
            if isinstance(inmodal_seg, dict):
                visible_mask = self.decode_segmentation(inmodal_seg)
            else:
                visible_mask = self.decode_polygon(inmodal_seg, H, W)
        else:
            # Fallback: use amodal mask as visible mask
            visible_mask = amodal_mask.copy()
        
        # Get bbox and crop/resize
        bbox = annotation['bbox']
        image_crop, visible_mask_crop, amodal_mask_crop = self.crop_and_resize(
            image, visible_mask, amodal_mask, bbox
        )
        
        # Convert to tensors
        # Image: normalize to [0, 1] and apply ImageNet normalization
        image_tensor = transforms.ToTensor()(image_crop)
        # Apply ImageNet normalization
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image_tensor = normalize(image_tensor)
        
        # Masks: convert to float tensors
        visible_mask_tensor = torch.from_numpy(visible_mask_crop).float().unsqueeze(0)
        amodal_mask_tensor = torch.from_numpy(amodal_mask_crop).float().unsqueeze(0)
        
        return {
            'image': image_tensor,
            'visible_mask': visible_mask_tensor,
            'amodal_mask': amodal_mask_tensor,
            'bbox': torch.tensor(bbox, dtype=torch.float32),
            'category_id': annotation['category_id'],
            'image_id': image_id,
            'annotation_id': annotation['id'],
            'area': annotation['area'],
            'iscrowd': annotation.get('iscrowd', 0),
            'occlude_rate': annotation.get('occlude_rate', 0.0),
            'filename': image_info['file_name'],
            'dataset': 'd2sa'
        }

    def __len__(self) -> int:
        return len(self.annotations)


class CombinedDataset(Dataset):
    """
    Combined dataset that can merge ViTASim, COCOA, KINS, and D2SA datasets
    """
    def __init__(self, datasets: List[Dataset]):
        """
        Args:
            datasets: List of dataset objects to combine
        """
        self.datasets = datasets
        self.dataset_sizes = [len(d) for d in datasets]
        self.cumulative_sizes = np.cumsum([0] + self.dataset_sizes)
        
        print(f"Combined dataset with {len(self)} total samples:")
        for i, size in enumerate(self.dataset_sizes):
            print(f"  Dataset {i}: {size} samples")
    
    def __getitem__(self, idx: int):
        # Find which dataset this index belongs to
        dataset_idx = np.searchsorted(self.cumulative_sizes[1:], idx, side='right')
        # Get the local index within that dataset
        local_idx = idx - self.cumulative_sizes[dataset_idx]
        return self.datasets[dataset_idx][local_idx]
    
    def __len__(self) -> int:
        return sum(self.dataset_sizes)


# Legacy class name for backward compatibility
class AmodalSegmentationDataset(ViTASimDataset):
    """Backward compatibility alias"""
    pass