"""2F pressure-array sensor dataset with text modality support."""

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import json
import logging
from torch.utils.data import Dataset
from modal_config import MODALITY_CONFIG
from ImageBind.models.multimodal_preprocessors import SimpleTokenizer

class SomaData2F(Dataset):
    """2F pressure-array sensor dataset."""

    def __init__(self, root_dir, label_file="标签.xlsx", text_annotations_file="text_annotations.json",
                 bpe_path=None, window_size=3, strict_validation=True):
        """
        Args:
            root_dir: processed data root directory
            label_file: label file path
            text_annotations_file: text annotation file path
            bpe_path: BPE file path (None = use default)
            window_size: temporal window size in frames
            strict_validation: whether to enforce strict validation
        """
        self.root_dir = root_dir
        self.window_size = window_size
        self.sensor_type = "2F"
        self.enabled_modalities = MODALITY_CONFIG['MODALITY_ENABLED']
        self.strict_validation = strict_validation

        print(f"SomaData2F init:")
        print(f"   Root: {root_dir}")
        print(f"   Sensor: {self.sensor_type}")
        print(f"   Modalities: {[m for m, e in self.enabled_modalities.items() if e]}")
        print(f"   Window: {window_size} frames")
        print(f"   Strict: {strict_validation}")
        if bpe_path:
            print(f"   BPE path: {bpe_path}")

        self.transforms = self._initialize_transforms()

        try:
            if bpe_path is None:
                bpe_path = "ImageBind/bpe/bpe_simple_vocab_16e6.txt.gz"
                print(f"   Using default BPE path: {bpe_path}")
            self.tokenizer = SimpleTokenizer(bpe_path=bpe_path)
            print("   SimpleTokenizer initialized")
        except Exception as e:
            error_msg = f"SimpleTokenizer init failed: {str(e)}"
            logging.error(error_msg)
            if self.strict_validation:
                raise RuntimeError(error_msg)
            else:
                print(f"   {error_msg}")

        self.label_df = self._load_labels(os.path.join(root_dir, label_file))

        # Text annotations are loaded from JSON, not from a Text/ folder
        self.text_annotations_file = os.path.join(root_dir, text_annotations_file)
        if not os.path.exists(self.text_annotations_file):
            error_msg = f"Text annotation file not found: {self.text_annotations_file}"
            logging.error(error_msg)
            if self.strict_validation:
                raise FileNotFoundError(error_msg)
            else:
                print(f"   {error_msg}")
        else:
            try:
                with open(self.text_annotations_file, 'r', encoding='utf-8') as f:
                    self.text_annotations = json.load(f)
                print(f"   Loaded {len(self.text_annotations['sample_annotations'])} text annotations")
            except Exception as e:
                error_msg = f"Failed to load text annotations: {str(e)}"
                logging.error(error_msg)
                if self.strict_validation:
                    raise RuntimeError(error_msg)
                else:
                    print(f"   {error_msg}")
                    self.text_annotations = {'sample_annotations': {}}

        # Text modality does not require a folder; only check JSON
        self.samples = self._collect_samples_strict()

        print(f"Init complete:")
        print(f"   Valid samples: {len(self.samples)}")

    def _initialize_transforms(self):
        transforms_dict = {}

        for modality, enabled in self.enabled_modalities.items():
            if not enabled:
                continue

            if modality == 'depth':
                transforms_dict['depth'] = self._depth_transform()
            elif modality == 'rgb':
                transforms_dict['rgb'] = self._rgb_transform()
            elif modality == 'photo':
                transforms_dict['photo'] = self._photo_transform()
            elif modality == 'imu':
                transforms_dict['imu'] = self._imu_transform()
            elif modality == 'tactile':
                transforms_dict['tactile'] = self._tactile_transform()

        return transforms_dict

    def _depth_transform(self):
        params = MODALITY_CONFIG['MODALITY_PARAMS']['vision']
        return transforms.Compose([
            transforms.Lambda(lambda img: img.crop(params['crop_region'])),
            transforms.Resize(params['target_size'], interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def _rgb_transform(self):
        params = MODALITY_CONFIG['MODALITY_PARAMS']['vision']
        return transforms.Compose([
            transforms.Lambda(lambda img: img.crop(params['crop_region'])),
            transforms.Resize(params['target_size'], interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                               std=(0.26862954, 0.26130258, 0.27577711))
        ])

    def _photo_transform(self):
        params = MODALITY_CONFIG['MODALITY_PARAMS']['vision']
        return transforms.Compose([
            transforms.Resize(params['target_size'], interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                               std=(0.26862954, 0.26130258, 0.27577711))
        ])

    def _imu_transform(self):
        return lambda x: x

    def _tactile_transform(self):
        """Tactile data preprocessing for 2F sensor (2 fingers)."""
        params = MODALITY_CONFIG['MODALITY_PARAMS']['tactile']
        def reshape_and_normalize(x):
            # 2F: (3, 2, 8, 14) -> (3, 1, 16, 14) -> (3, 3, 224, 224)
            x = x.view(3, 1, 16, 14)  # merge 2 sensors
            x = x.repeat(1, 3, 1, 1)  # expand to 3 channels
            x = torch.nn.functional.interpolate(x.float(), size=(224, 224), mode='bilinear', align_corners=False)
            if params['preprocessing'] == 'normalize':
                return (x - 0.5) / 0.5
            else:
                return x
        return reshape_and_normalize

    def _validate_file_exists(self, file_path, error_msg_prefix=""):
        if not os.path.exists(file_path):
            error_msg = f"{error_msg_prefix} file not found: {file_path}"
            logging.error(error_msg)
            if self.strict_validation:
                raise FileNotFoundError(error_msg)
            else:
                print(f"   WARNING: {error_msg}")
                return False
        return True

    def _load_labels(self, label_path):
        self._validate_file_exists(label_path, "Label file")

        df = pd.read_excel(label_path)
        df.columns = [col.strip() for col in df.columns]

        # The training data uses Chinese column names. The generated English
        # label sheets are accepted as an optional compatibility format.
        df = df.rename(columns={
            '物品名称': 'item_name',
            '实验次数': 'trial',
            '样本序号': 'sample_idx',
            '标签': 'label',
        })

        required_columns = ['item_name', 'trial', 'sample_idx', 'label']
        for col in required_columns:
            if col not in df.columns:
                error_msg = f"Label file missing column: '{col}'"
                logging.error(error_msg)
                if self.strict_validation:
                    raise ValueError(error_msg)
                else:
                    print(f"   WARNING: {error_msg}")

        return df

    def _get_label(self, class_name, trial, sample_idx):
        match = self.label_df[
            (self.label_df['item_name'] == class_name) &
            (self.label_df['trial'] == int(trial)) &
            (self.label_df['sample_idx'] == int(sample_idx))
        ]
        if match.empty:
            error_msg = f"Label not found: {class_name}/{trial}/{sample_idx}"
            logging.warning(error_msg)
            return None
        return match.iloc[0]['label']

    def _get_modality_folder(self, modality):
        """Map modality name to folder name. Text is not loaded from a folder."""
        mapping = {
            'depth': 'Depth',
            'rgb': 'RGB',
            'photo': 'Photo',
            'imu': 'IMU',
            'tactile': 'Tactile'
        }
        return mapping.get(modality, modality.capitalize())

    def _validate_modality_data(self, tensor, modality_name, expected_dims=None):
        if not isinstance(tensor, torch.Tensor):
            error_msg = f"Modality '{modality_name}' is not a Tensor, type: {type(tensor)}"
            logging.error(error_msg)
            if self.strict_validation:
                raise TypeError(error_msg)
            else:
                print(f"   WARNING: {error_msg}")
                return False

        if torch.isnan(tensor).any():
            error_msg = f"Modality '{modality_name}' contains NaN"
            logging.error(error_msg)
            if self.strict_validation:
                raise ValueError(error_msg)
            else:
                print(f"   WARNING: {error_msg}")
                return False

        if expected_dims is not None:
            if tensor.dim() != len(expected_dims):
                error_msg = f"Modality '{modality_name}' dim mismatch: expected {len(expected_dims)}D, got {tensor.dim()}D"
                logging.error(error_msg)
                if self.strict_validation:
                    raise ValueError(error_msg)
                else:
                    print(f"   WARNING: {error_msg}")
                    return False

            for i, (expected, actual) in enumerate(zip(expected_dims, tensor.shape)):
                if expected is not None and actual != expected:
                    error_msg = f"Modality '{modality_name}' dim[{i}] mismatch: expected {expected}, got {actual}"
                    logging.error(error_msg)
                    if self.strict_validation:
                        raise ValueError(error_msg)
                    else:
                        print(f"   WARNING: {error_msg}")
                        return False

        return True

    def _load_text(self, class_name, trial, sample_idx):
        """
        Load text description from JSON (not from folder).
        Uses the first qa_variant, if present, for deterministic inference.
        """
        key = f"{class_name}/{trial}/{sample_idx}"

        try:
            if 'sample_annotations' in self.text_annotations and key in self.text_annotations['sample_annotations']:
                ann = self.text_annotations['sample_annotations'][key]

                if 'qa_variants' in ann and ann['qa_variants']:
                    # Evaluation and inference must be deterministic across workers.
                    text_desc = ann['qa_variants'][0]
                else:
                    text_desc = ann.get('english_description', ann.get('chinese_description', ''))

                if not text_desc or not isinstance(text_desc, str) or len(text_desc.strip()) == 0:
                    raise ValueError(f"Invalid text description for key: '{key}'")
                return text_desc
            else:
                if self.strict_validation:
                    raise ValueError(f"Text annotation not found for key: '{key}'")
                else:
                    print(f"   WARNING: Text annotation not found for key: '{key}', using default")
                    return self._generate_default_text(class_name, trial, sample_idx)
        except Exception as e:
            error_msg = f"Failed to load text [{key}]: {str(e)}"
            logging.warning(error_msg)
            if self.strict_validation:
                raise RuntimeError(error_msg)
            else:
                print(f"   WARNING: {error_msg}")
                return self._generate_default_text(class_name, trial, sample_idx)

    def _generate_default_text(self, class_name, trial, sample_idx):
        """Generate a default text description when annotations are missing."""
        item_name = class_name.split('_')[-1] if '_' in class_name else class_name
        return f"A {item_name} object in experiment {trial}, sample {sample_idx}."

    def _tokenize_text(self, text_desc, max_length=77):
        try:
            if not text_desc or not isinstance(text_desc, str) or len(text_desc.strip()) == 0:
                text_desc = "An object"

            tokens = self.tokenizer(text_desc, context_length=max_length)

            if tokens.dim() == 0 or tokens.size(0) == 0:
                raise ValueError("Empty token sequence")

            return tokens

        except Exception as e:
            error_msg = f"Text tokenization failed: {str(e)}"
            logging.error(error_msg)
            if self.strict_validation:
                raise RuntimeError(error_msg)
            else:
                print(f"   WARNING: {error_msg}")
                return self.tokenizer("An object", context_length=max_length)

    def _load_depth(self, path):
        if not self.enabled_modalities.get('depth', False):
            return None

        frames = []
        for i in range(self.window_size):
            img_path = os.path.join(path, f"depth_{i}.png")
            self._validate_file_exists(img_path, "Depth")

            img = Image.open(img_path)
            if img.mode in ('I', 'I;16'):
                # Match the exact preprocessing used by Stage 2/3 training.
                arr = np.array(img, dtype=np.float32)
                max_value = float(arr.max())
                if max_value > 0:
                    arr /= max_value
            else:
                # 8-bit or other: backward compatible
                arr = np.array(img, dtype=np.float32) / 255.0
            depth_img = Image.fromarray(arr, mode='F')
            frame = self.transforms['depth'](depth_img)

            if not self._validate_modality_data(frame, 'depth_frame', expected_dims=[1, 224, 224]):
                raise ValueError(f"Invalid depth frame {i}")

            frames.append(frame)

        return torch.stack(frames) if frames else None

    def _load_rgb(self, path):
        if not self.enabled_modalities.get('rgb', False):
            return None

        frames = []
        for i in range(self.window_size):
            img_path = os.path.join(path, f"rgb_{i}.jpg")
            self._validate_file_exists(img_path, "RGB")

            img = Image.open(img_path).convert('RGB')
            frame = self.transforms['rgb'](img)

            if not self._validate_modality_data(frame, 'rgb_frame', expected_dims=[3, 224, 224]):
                raise ValueError(f"Invalid RGB frame {i}")

            frames.append(frame)

        return torch.stack(frames) if frames else None

    def _load_photo(self, path):
        if not self.enabled_modalities.get('photo', False):
            return None

        frames = []
        for i in range(self.window_size):
            img_path = os.path.join(path, f"frame_{i}.jpg")
            self._validate_file_exists(img_path, "Photo")

            img = Image.open(img_path).convert('RGB')
            frame = self.transforms['photo'](img)

            if not self._validate_modality_data(frame, 'photo_frame', expected_dims=[3, 224, 224]):
                raise ValueError(f"Invalid photo frame {i}")

            frames.append(frame)

        return torch.stack(frames) if frames else None

    def _load_imu(self, base_path):
        if not self.enabled_modalities.get('imu', False):
            return None

        data = []
        for sensor_id in [1, 2]:
            csv_path = os.path.join(base_path, f"imu{sensor_id}.csv")
            self._validate_file_exists(csv_path, f"IMU sensor {sensor_id}")

            sensor_data = []
            with open(csv_path, 'r') as f:
                lines = f.readlines()
                original_lines = lines[:2000]
                if len(original_lines) < 2000:
                    num_copies = (2000 + len(original_lines) - 1) // len(original_lines)
                    expanded_lines = (original_lines * num_copies)[:2000]
                    lines = expanded_lines
                else:
                    lines = original_lines[:2000]

                for i, line in enumerate(lines[:2000]):
                    values = line.strip().split(',')[1:7]
                    if len(values) < 6:
                        error_msg = f"IMU line {i} format error: expected 6 values, got {len(values)}"
                        if self.strict_validation:
                            raise ValueError(error_msg)
                        else:
                            print(f"   WARNING: {error_msg}")
                            values = ['0'] * 6

                    try:
                        sensor_data.append(np.array(values[:6], dtype=np.float32))
                    except Exception as e:
                        error_msg = f"IMU line {i} conversion failed: {str(e)}"
                        if self.strict_validation:
                            raise ValueError(error_msg)
                        else:
                            print(f"   WARNING: {error_msg}")
                            sensor_data.append(np.zeros(6, dtype=np.float32))

            if len(sensor_data) < 2000:
                sensor_data += [np.zeros(6, dtype=np.float32)] * (2000 - len(sensor_data))
            data.append(np.stack(sensor_data))

        imu_6axis = (data[0] + data[1]) / 2
        tensor = torch.tensor(imu_6axis.T, dtype=torch.float32)

        if not self._validate_modality_data(tensor, 'imu', expected_dims=[6, 2000]):
            raise ValueError("Invalid IMU data")

        return tensor

    def _load_tactile(self, base_path):
        if not self.enabled_modalities.get('tactile', False):
            return None

        csv_path = os.path.join(base_path, "sensor_data.csv")
        self._validate_file_exists(csv_path, "Tactile")

        data = []
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            if len(lines) < self.window_size:
                error_msg = f"Tactile data insufficient frames: expected {self.window_size}, got {len(lines)}"
                if self.strict_validation:
                    raise ValueError(error_msg)
                else:
                    print(f"   WARNING: {error_msg}")

            for i in range(min(self.window_size, len(lines))):
                line = lines[i].strip().split(',')
                if len(line) < 224:
                    error_msg = f"Tactile line {i} format error: expected 224 values, got {len(line)}"
                    if self.strict_validation:
                        raise ValueError(error_msg)
                    else:
                        print(f"   WARNING: {error_msg}")
                        continue

                try:
                    sensor1 = np.array(line[:112], dtype=np.float32).reshape(8, 14)
                    sensor2 = np.array(line[112:224], dtype=np.float32).reshape(8, 14)
                    data.append(np.stack([sensor1, sensor2], axis=0))
                except Exception as e:
                    error_msg = f"Tactile line {i} processing failed: {str(e)}"
                    if self.strict_validation:
                        raise ValueError(error_msg)
                    else:
                        print(f"   WARNING: {error_msg}")
                        continue

        if len(data) < self.window_size:
            error_msg = f"Tactile valid frames insufficient: expected {self.window_size}, got {len(data)}"
            if self.strict_validation:
                raise ValueError(error_msg)
            else:
                print(f"   WARNING: {error_msg}")
                return None

        tactile_tensor = torch.tensor(np.stack(data[:self.window_size]), dtype=torch.float32)
        transformed = self.transforms['tactile'](tactile_tensor)

        if not self._validate_modality_data(transformed, 'tactile', expected_dims=[3, 3, 224, 224]):
            raise ValueError("Invalid tactile data")

        return transformed

    def _collect_samples_strict(self):
        """Collect valid samples. Text modality checks JSON, not folder."""
        samples = []
        total_skipped = 0
        skip_reasons = {}

        print("Strictly validating all samples...")

        for class_name in sorted(os.listdir(self.root_dir)):
            class_path = os.path.join(self.root_dir, class_name)
            if not (os.path.isdir(class_path) and class_name[0].isdigit()):
                continue

            for trial_folder in sorted(os.listdir(class_path)):
                trial_path = os.path.join(class_path, trial_folder)
                if not (os.path.isdir(trial_path) and trial_folder.isdigit()):
                    continue

                for sample_folder in sorted(os.listdir(trial_path)):
                    sample_path = os.path.join(trial_path, sample_folder)
                    if not (os.path.isdir(sample_path) and sample_folder.isdigit()):
                        continue

                    valid = True
                    missing_modalities = []
                    reason = ""

                    try:
                        # Non-text modalities: check folders
                        for modality, enabled in self.enabled_modalities.items():
                            if not enabled or modality == 'text':
                                continue

                            mod_path = os.path.join(sample_path, self._get_modality_folder(modality))
                            if not os.path.isdir(mod_path):
                                valid = False
                                missing_modalities.append(f"{modality} folder")
                                reason = f"Missing folder: {mod_path}"
                                break

                            if modality == 'tactile':
                                csv_path = os.path.join(mod_path, "sensor_data.csv")
                                if not os.path.exists(csv_path):
                                    valid = False
                                    missing_modalities.append("sensor_data.csv")
                                    reason = f"Missing file: {csv_path}"
                                    break

                            elif modality == 'imu':
                                for sensor_id in range(1, 3):
                                    csv_path = os.path.join(mod_path, f"imu{sensor_id}.csv")
                                    if not os.path.exists(csv_path):
                                        valid = False
                                        missing_modalities.append(f"imu{sensor_id}.csv")
                                        reason = f"Missing file: {csv_path}"
                                        break
                            if not valid:
                                break

                        # Text modality: check JSON annotation
                        if valid and self.enabled_modalities.get('text', False):
                            key = f"{class_name}/{trial_folder}/{sample_folder}"
                            if 'sample_annotations' not in self.text_annotations or key not in self.text_annotations['sample_annotations']:
                                valid = False
                                reason = f"Missing text annotation for key: {key}"

                        if not valid:
                            raise FileNotFoundError(reason)

                        label = self._get_label(class_name, trial_folder, sample_folder)
                        if label is None:
                            raise ValueError("Label not found")

                        samples.append({
                            'class': class_name,
                            'trial': trial_folder,
                            'sample': sample_folder,
                            'path': sample_path,
                            'label': label
                        })

                    except Exception as e:
                        total_skipped += 1
                        reason = str(e)
                        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

                        if self.strict_validation:
                            print(f"   Strict validation failed: {reason}")
                            raise

        if skip_reasons:
            print("Skip reasons:")
            for reason, count in skip_reasons.items():
                print(f"   {reason}: {count} samples")

        print(f"Sample collection complete:")
        print(f"   Valid: {len(samples)}")
        print(f"   Skipped: {total_skipped}")

        if len(samples) == 0:
            error_msg = "No valid samples passed validation"
            logging.error(error_msg)
            raise RuntimeError(error_msg)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx >= len(self.samples) or idx < 0:
            error_msg = f"Invalid index {idx}, dataset size {len(self.samples)}"
            if self.strict_validation:
                raise IndexError(error_msg)
            else:
                idx = idx % len(self.samples)

        sample = self.samples[idx]
        sample_path = sample['path']
        class_name = sample['class']
        trial = sample['trial']
        sample_idx = sample['sample']

        # Stable sample ID with a sensor-type prefix for evaluation records.
        # Format: "{sensor_type}/{class}/{trial}/{sample_idx}"
        # The prefix avoids collisions when 2F and 3F share the same directory structure.
        sample_id = f"2F/{class_name}/{trial}/{sample_idx}"

        try:
            data = {}

            for modality, enabled in self.enabled_modalities.items():
                if not enabled or modality == 'text':
                    continue

                if modality == 'depth':
                    depth_path = os.path.join(sample_path, 'Depth')
                    data['depth'] = self._load_depth(depth_path)
                elif modality == 'rgb':
                    rgb_path = os.path.join(sample_path, 'RGB')
                    data['rgb'] = self._load_rgb(rgb_path)
                elif modality == 'photo':
                    photo_path = os.path.join(sample_path, 'Photo')
                    data['photo'] = self._load_photo(photo_path)
                elif modality == 'imu':
                    imu_path = os.path.join(sample_path, 'IMU')
                    data['imu'] = self._load_imu(imu_path)
                elif modality == 'tactile':
                    tactile_path = os.path.join(sample_path, 'Tactile')
                    data['tactile'] = self._load_tactile(tactile_path)

            text_desc = ""
            if self.enabled_modalities.get('text', False):
                text_desc = self._load_text(class_name, trial, sample_idx)
                max_length = MODALITY_CONFIG['MODALITY_PARAMS']['text'].get('max_length', 77)
                text_tokens = self._tokenize_text(text_desc, max_length)
                data['text'] = text_tokens
                self._validate_modality_data(text_tokens, 'text', expected_dims=[max_length])

            label = torch.tensor(sample['label'], dtype=torch.long)

            # Keep the stable ID alongside tensors so evaluation outputs are traceable.
            data['sample_id'] = sample_id

            for modality, tensor in data.items():
                if tensor is None:
                    error_msg = f"Modality '{modality}' is None"
                    if self.strict_validation:
                        raise ValueError(error_msg)
                    else:
                        print(f"   WARNING: {error_msg}")
                        continue

            return data, label, text_desc

        except Exception as e:
            error_msg = f"Sample {idx} (path: {sample_path}) load failed: {str(e)}"
            logging.error(error_msg, exc_info=True)
            if self.strict_validation:
                raise RuntimeError(error_msg)
            else:
                print(f"   WARNING: {error_msg}")
                placeholder_data = {}
                for modality, enabled in self.enabled_modalities.items():
                    if enabled:
                        if modality == 'text':
                            placeholder_data[modality] = torch.zeros(77, dtype=torch.long)
                        elif modality == 'imu':
                            placeholder_data[modality] = torch.zeros(6, 2000)
                        else:
                            placeholder_data[modality] = torch.zeros(3, 3, 224, 224)
                return placeholder_data, torch.tensor(-1, dtype=torch.long), "fallback text"

    def get_modality_list(self):
        """Return list of enabled modalities."""
        return [m for m, e in self.enabled_modalities.items() if e]

    def validate_all_samples(self):
        """Pre-validate all samples by attempting to load each one."""
        print("Pre-validating all samples...")
        invalid_samples = []

        for idx in range(len(self.samples)):
            try:
                self[idx]
                print(f"   Sample {idx}: OK")
            except Exception as e:
                invalid_samples.append((idx, str(e)))
                print(f"   Sample {idx}: FAILED - {str(e)}")

        print(f"\nValidation result:")
        print(f"   Valid: {len(self.samples) - len(invalid_samples)}")
        print(f"   Invalid: {len(invalid_samples)}")

        if invalid_samples:
            print("\nInvalid samples:")
            for idx, error in invalid_samples[:5]:
                print(f"   Sample {idx}: {error}")
                if len(invalid_samples) > 5:
                    print(f"   ... ({len(invalid_samples)} total invalid samples)")

        return invalid_samples
