from ant.calibration.calibrator import (
    IdentityCalibrator,
    MLPCalibrator,
    TemperatureCalibrator,
    VectorScalingCalibrator,
    build_calibrator,
)

CALIBRATOR_REGISTRY = {
    "identity": IdentityCalibrator,
    "temperature": TemperatureCalibrator,
    "vector_scaling": VectorScalingCalibrator,
    "mlp": MLPCalibrator,
}

__all__ = [
    "CALIBRATOR_REGISTRY",
    "IdentityCalibrator",
    "MLPCalibrator",
    "TemperatureCalibrator",
    "VectorScalingCalibrator",
    "build_calibrator",
]
