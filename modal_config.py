"""Public inference-time modality and preprocessing configuration."""

MODALITY_CONFIG = {
    'MODALITY_ENABLED': {
        'rgb': True,
        'photo': True,
        'depth': True,
        'tactile': True,
        'imu': True,
        'text': True
    },

    # Retained for compatibility with applications that display or select
    # enabled modalities. These values are not consumed by the release model.
    'MODALITY_WEIGHTS': {
        'rgb': 1.0 / 6.0,
        'photo': 1.0 / 6.0,
        'depth': 1.0 / 6.0,
        'tactile': 1.0 / 6.0,
        'imu': 1.0 / 6.0,
        'text': 1.0 / 6.0,
    },

    'MODALITY_PARAMS': {
        'tactile': {
            'sensor_type': 'Unified',  # '2F', '3F', or 'Unified' (2F+3F mixed)
            'preprocessing': 'normalize'
        },
        'imu': {
            'sampling_rate': 100,
            'channels': 6
        },
        'vision': {
            'crop_region': (160, 0, 480, 480),
            'target_size': (224, 224)
        },
        'text': {
            'max_length': 77,
            'use_english': True
        }
    },

    # Minimal compatibility metadata; no training objective is implemented here.
    'TASK_CONFIG': {
        'current_task': 'modality_alignment',
        'use_llama': False,
    },
}


def get_active_modalities():
    """Return enabled modalities with normalized compatibility weights."""
    active = {
        modality: MODALITY_CONFIG['MODALITY_WEIGHTS'].get(modality, 1.0)
        for modality, enabled in MODALITY_CONFIG['MODALITY_ENABLED'].items()
        if enabled
    }
    total = sum(active.values())
    return {modality: weight / total for modality, weight in active.items()} if total else {}


def is_task_enabled(task_name):
    """Compatibility helper for applications using the historical config API."""
    return MODALITY_CONFIG['TASK_CONFIG']['current_task'] == task_name


def should_use_llama():
    """Return whether an external LLaMA component is requested (disabled here)."""
    return MODALITY_CONFIG['TASK_CONFIG']['use_llama']


def get_sensor_type():
    """Return the configured tactile dataset adapter."""
    return MODALITY_CONFIG['MODALITY_PARAMS']['tactile']['sensor_type']
