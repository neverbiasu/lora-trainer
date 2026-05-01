"""Error definitions and friendly messages"""


class LoRATrainerError(Exception):
    """Base error class"""

    error_code: str = "E000"

    def __init__(self, message: str, suggestions: list = None):
        self.message = message
        self.suggestions = suggestions or []
        super().__init__(self.format_error())

    def format_error(self) -> str:
        """Format error message"""
        msg = f"❌ [{self.error_code}] {self.message}"
        if self.suggestions:
            msg += "\n\n💡 Suggestions:"
            for suggestion in self.suggestions:
                msg += f"\n  - {suggestion}"
        return msg


# Dataset-related errors
class DatasetNotFoundError(LoRATrainerError):
    """Dataset not found"""

    error_code = "E001"


class MissingCaptionError(LoRATrainerError):
    """Missing caption file"""

    error_code = "E002"


class DatasetTooSmallError(LoRATrainerError):
    """Dataset too small"""

    error_code = "E003"


# VRAM-related errors
class CUDANotAvailableError(LoRATrainerError):
    """CUDA not available"""

    error_code = "E010"


class OutOfMemoryError(LoRATrainerError):
    """Out of memory"""

    error_code = "E011"


class InsufficientVRAMError(LoRATrainerError):
    """Insufficient VRAM (estimation phase)"""

    error_code = "E012"


# Model-related errors
class ModelNotFoundError(LoRATrainerError):
    """Model not found"""

    error_code = "E020"


class ModelVersionIncompatibleError(LoRATrainerError):
    """Model version incompatible"""

    error_code = "E021"


# Training-related errors
class NaNLossError(LoRATrainerError):
    """NaN loss"""

    error_code = "E030"


class InfiniteLossError(LoRATrainerError):
    """Infinite loss"""

    error_code = "E031"


class IneffectiveTrainingError(LoRATrainerError):
    """Training did not pass effectiveness gate checks"""

    error_code = "E032"


# Config-related errors
class InvalidConfigError(LoRATrainerError):
    """Invalid config format"""

    error_code = "E040"


class MissingRequiredFieldError(LoRATrainerError):
    """Missing required field"""

    error_code = "E041"


class ConfigConflictError(LoRATrainerError):
    """Config conflict"""

    error_code = "E042"


# Checkpoint-related errors
class CheckpointNotFoundError(LoRATrainerError):
    """Checkpoint not found"""

    error_code = "E050"


class CheckpointCorruptedError(LoRATrainerError):
    """Checkpoint corrupted"""

    error_code = "E051"
