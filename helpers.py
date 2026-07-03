
success_buffer_value = -100.0

def is_successful_ending_value_with_clamp(ending_value: float) -> tuple[bool, float]:
    """Return whether a strategy should count as successful (slight buffer) and a clamped ending value."""

    success = is_successful_ending_value(ending_value)
    clamped_value = max(0.0, float(ending_value))
    return success, clamped_value

def is_successful_ending_value(ending_value: float) -> bool:
    """Return whether a strategy should count as successful (slight buffer)"""

    return ending_value > success_buffer_value