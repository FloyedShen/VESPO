import math
import re
from typing import Optional, Tuple, Union, Dict, Any

# Register the custom GeoGuessr reward manager
# This import triggers the @register("geoguessr") decorator, making the reward manager
# available when verl loads this module via custom_reward_function.path
try:
    from .reward_manager import GeoGuessrRewardManager  # noqa: F401
except ImportError:
    # Fallback for direct execution or when module structure is different
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from geoguessr.reward_manager import GeoGuessrRewardManager  # noqa: F401
    except ImportError:
        import warnings
        warnings.warn(
            "Failed to import GeoGuessrRewardManager. "
            "The 'geoguessr' reward manager will not be available. "
            "Use reward_model.reward_manager=prime or reward_model.reward_manager=batch instead.",
            ImportWarning
        )


# ============================================================================
# Coordinate Extraction Functions (following math_reward.py style)
# ============================================================================

def last_boxed_only_string(string: str) -> Optional[str]:
    """
    Extract the last \boxed{} content from a string.

    Adapted from verl.utils.reward_score.math_reward to handle coordinate format.

    Args:
        string: Input string containing \boxed{latitude, longitude}

    Returns:
        The last boxed string including the \boxed{} wrapper, or None if not found
    """
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx: right_brace_idx + 1]

    return retval


def remove_boxed(s: str) -> str:
    """
    Remove the \boxed{} wrapper from a string.

    Adapted from verl.utils.reward_score.math_reward.

    Args:
        s: String with \boxed{} wrapper

    Returns:
        Content inside the \boxed{}
    """
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left): -1]


def parse_coordinates_from_boxed(boxed_string: str) -> Optional[Tuple[float, float]]:
    """
    Parse coordinates from the content inside \boxed{}.

    Expected format: latitude, longitude (e.g., "40.7128, -74.0060")

    Args:
        boxed_string: String containing coordinates

    Returns:
        Tuple of (latitude, longitude) or None if parsing fails
    """
    try:
        # Remove whitespace and split by comma
        coords_str = boxed_string.strip()

        # Try to parse as "lat, lon" format
        parts = coords_str.split(',')
        if len(parts) == 2:
            lat = float(parts[0].strip())
            lon = float(parts[1].strip())

            # Validate coordinate ranges
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return (lat, lon)

        # Try regex patterns as fallback
        # Pattern 1: Two numbers separated by comma/whitespace
        pattern1 = r'(-?\d+\.?\d*)\s*,?\s*(-?\d+\.?\d*)'
        match = re.search(pattern1, coords_str)
        if match:
            lat = float(match.group(1))
            lon = float(match.group(2))
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return (lat, lon)

    except (ValueError, AttributeError, IndexError):
        pass

    return None


import re
import signal
from typing import Optional, Tuple


def parse_coordinates_fallback(text: str) -> Optional[Tuple[float, float]]:
    """
    Fallback coordinate parser for various formats when \boxed{} is not found.
    Searches from the END of text as final answers are usually at the bottom.

    Supports formats:
    - "latitude: 40.7128, longitude: -74.0060"
    - "lat: 40.7128, lon: -74.0060"
    - "(40.7128, -74.0060)"
    - "40.7128°N, 74.0060°W"
    - Plain numbers: "40.7128, -74.0060"

    Args:
        text: Model output text

    Returns:
        Tuple of (latitude, longitude) or None if not found
    """

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Regex timeout")

    # 设置3秒超时
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(3)

    try:
        # 只处理文本的最后200字符
        text = text[-200:] if len(text) > 200 else text
        text_lower = text.lower()

        # Pattern 1: "latitude: XX, longitude: YY" or "lat: XX, lon: YY"
        # 使用 findall 找到所有匹配，取最后一个
        pattern1 = r'lat(?:itude)?[:\s]+(-?\d+\.?\d*)[,\s]+lon(?:gitude)?[:\s]+(-?\d+\.?\d*)'
        matches = re.findall(pattern1, text_lower)
        if matches:
            lat, lon = float(matches[-1][0]), float(matches[-1][1])
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return (lat, lon)

        # Pattern 2: "(XX, YY)" - parentheses format
        # 找到所有匹配，取最后一个
        pattern2 = r'\((-?\d+\.?\d*)[,\s]+(-?\d+\.?\d*)\)'
        matches = re.findall(pattern2, text)
        if matches:
            # 从后往前检查，找到第一个有效的
            for match in reversed(matches):
                lat, lon = float(match[0]), float(match[1])
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return (lat, lon)

        # Pattern 3: "XX°N/S, YY°E/W"
        # 找到所有匹配，取最后一个
        pattern3 = r'(-?\d+\.?\d*)[°\s]*([ns])[,\s]+(-?\d+\.?\d*)[°\s]*([ew])'
        matches = re.findall(pattern3, text_lower)
        if matches:
            # 从后往前检查
            for match in reversed(matches):
                lat = float(match[0])
                if match[1] == 's':
                    lat = -lat
                lon = float(match[2])
                if match[3] == 'w':
                    lon = -lon
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return (lat, lon)

        # Pattern 4: Any two decimal numbers (从后往前查找)
        # 限制只查找最后10对数字
        pattern4 = r'(-?\d{1,3}\.?\d{0,8})[,\s]+(-?\d{1,3}\.?\d{0,8})'
        matches = re.findall(pattern4, text)

        if matches:
            # 从最后往前检查，最多检查10对
            for match in reversed(matches[-10:]):
                try:
                    lat, lon = float(match[0]), float(match[1])
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        return (lat, lon)
                except ValueError:
                    continue

        return None

    except TimeoutError:
        print(f"[WARNING] Coordinate parsing timeout (text length: {len(text)})")
        return None
    except Exception as e:
        print(f"[ERROR] Coordinate parsing failed: {type(e).__name__}: {e}")
        return None
    finally:
        signal.alarm(0)

# ============================================================================
# Distance and Score Calculation
# ============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth
    using the Haversine formula.

    Args:
        lat1, lon1: Latitude and longitude of first point (in degrees)
        lat2, lon2: Latitude and longitude of second point (in degrees)

    Returns:
        Distance in kilometers
    """
    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    # Radius of Earth in kilometers
    r = 6371.0

    return c * r


def calculate_linear_score(distance_km: float, max_distance: float = 20000.0) -> float:
    """
    Calculate linear decay score based on distance.

    Args:
        distance_km: Distance in kilometers
        max_distance: Maximum distance for scoring (default: 20000 km, ~half Earth circumference)

    Returns:
        Score in [0, 1] range
    """
    return max(0.0, 1.0 - distance_km / max_distance)


def calculate_geoguessr_score(distance_km: float) -> float:
    """
    Calculate score using the official GeoGuessr competitive scoring system.

    Official GeoGuessr uses a 5000-point maximum score with distance-based decay.
    The formula approximates the real competitive scoring:
        score_raw = 5000 * 2^(-(distance_km / scale)^power)

    This implementation uses empirically-derived parameters that closely match
    the actual GeoGuessr scoring curve used in competitive play:
    - Scale: ~2500 km (controls the distance at which score halves)
    - Power: 3 (controls the steepness of decay curve)

    Args:
        distance_km: Distance in kilometers

    Returns:
        Score in [0, 1] range (normalized from 0-5000 points)
    """
    # Official GeoGuessr parameters (empirically derived)
    decay_coefficient = 1492.7  # Distance at which score decays significantly
    power = 3.0  # Steepness of the decay curve

    # Calculate raw score using power-law exponential decay
    # Formula: 5000 * 2^(-(distance / scale)^power)
    if distance_km == 0:
        score = 1.0
    else:
        score = math.exp(-distance_km / decay_coefficient)

    return max(0.0, min(1.0, score))


def calculate_accuracy_metrics(distance_km: float) -> Dict[str, float]:
    """
    Calculate accuracy metrics at different distance thresholds.

    Common GeoGuessr accuracy thresholds:
    - 1 km: Very precise (city block level)
    - 25 km: Regional accuracy (within city/county)
    - 200 km: Country-level accuracy (small/medium countries)
    - 750 km: Large region accuracy (state/province level)
    - 2500 km: Continental accuracy

    Args:
        distance_km: Distance in kilometers

    Returns:
        Dictionary with accuracy at each threshold (1.0 if within threshold, 0.0 otherwise)
    """
    thresholds = [1, 25, 200, 750, 2500]
    return {f"acc@{t}km": 1.0 if distance_km <= t else 0.0 for t in thresholds}


# ============================================================================
# Main Reward Function (verl interface)
# ============================================================================

def geoguessr_reward_function(
        data_source: str,
        solution_str: str,
        ground_truth: str,
        extra_info: Optional[dict] = None
) -> Union[float, Dict[str, Any]]:
    """
    Calculate reward for GeoGuessr task following verl framework interface.

    This function matches the interface of verl.utils.reward_score.math_reward
    and can be used as a drop-in replacement for GeoGuessr tasks.

    Args:
        data_source: Source identifier (not used, kept for interface compatibility)
        solution_str: Model's output string containing predicted coordinates in \boxed{lat, lon}
        ground_truth: Ground truth coordinates as string "lat,lon" or dict/tuple
        extra_info: Optional dict with configuration:
            - reward_type: str, one of ["linear", "official"] (default: "official")
            - max_distance: float, max distance for linear scoring (default: 20000.0)
            - verbose: bool, whether to print debug info (default: False)
            - return_dict: bool, whether to return detailed dict (default: True)

    Returns:
        If return_dict=True (default): Dictionary with keys:
            - score: float in [0, 1] range (based on reward_type)
            - distance@km: float, distance in kilometers
            - acc@1km, acc@25km, acc@200km, acc@750km, acc@2500km: float, accuracy at thresholds
            - geoguessr@point: float, official GeoGuessr points (0-5000)
            - parse_success: bool, whether coordinate parsing succeeded
        If return_dict=False: float score in [0, 1] range (backward compatible)

        Returns 0.0 or failure dict if parsing fails or coordinates are invalid
    """
    try:
        # Parse configuration
        if extra_info is None:
            extra_info = {}

        reward_type = extra_info.get("reward_type", "official")
        max_distance = extra_info.get("max_distance", 20000.0)
        verbose = extra_info.get("verbose", False)
        return_dict = extra_info.get("return_dict", True)

        # Extract predicted coordinates
        # First try to find \boxed{} format (preferred)
        last_boxed_string = last_boxed_only_string(solution_str)
        predicted_coords = None

        if last_boxed_string is not None:
            coords_content = remove_boxed(last_boxed_string)
            predicted_coords = parse_coordinates_from_boxed(coords_content)

        # Fallback to other formats if \boxed{} not found or parsing failed
        if predicted_coords is None:
            predicted_coords = parse_coordinates_fallback(solution_str)

        if predicted_coords is None:
            if verbose:
                print(f"[GeoGuessr Reward] Failed to parse coordinates from: {solution_str[:100]}")

            if return_dict:
                return {
                    "score": 0.0,
                    "distance@km": 20000.0,
                    "acc@1km": 0.0,
                    "acc@25km": 0.0,
                    "acc@200km": 0.0,
                    "acc@750km": 0.0,
                    "acc@2500km": 0.0,
                    "geoguessr@point": 0.0,
                    "parse_success": False
                }
            else:
                return 0.0

        pred_lat, pred_lon = predicted_coords

        # Parse ground truth
        gt_coords = None
        if isinstance(ground_truth, str):
            # Parse string format: "lat,lon"
            parts = ground_truth.split(',')
            if len(parts) == 2:
                gt_lat = float(parts[0].strip())
                gt_lon = float(parts[1].strip())
                gt_coords = (gt_lat, gt_lon)
        elif isinstance(ground_truth, (tuple, list)) and len(ground_truth) == 2:
            gt_coords = (float(ground_truth[0]), float(ground_truth[1]))
        elif isinstance(ground_truth, dict):
            gt_coords = (float(ground_truth['lat']), float(ground_truth['lon']))

        if gt_coords is None:
            if verbose:
                print(f"[GeoGuessr Reward] Invalid ground truth format: {ground_truth}")

            if return_dict:
                return {
                    "score": 0.0,
                    "distance@km": 20000.0,
                    "acc@1km": 0.0,
                    "acc@25km": 0.0,
                    "acc@200km": 0.0,
                    "acc@750km": 0.0,
                    "acc@2500km": 0.0,
                    "geoguessr@point": 0.0,
                    "parse_success": False
                }
            else:
                return 0.0

        gt_lat, gt_lon = gt_coords

        # Calculate distance
        distance_km = haversine_distance(pred_lat, pred_lon, gt_lat, gt_lon)

        # Calculate official GeoGuessr score (always needed for geoguessr@point)
        official_score = calculate_geoguessr_score(distance_km)
        geoguessr_points = official_score * 5000.0

        # Calculate reward based on type
        if reward_type == "official":
            reward = official_score
        elif reward_type == "linear":
            reward = calculate_linear_score(distance_km, max_distance=max_distance)
        else:
            if verbose:
                print(f"[GeoGuessr Reward] Unknown reward_type: {reward_type}, using official")
            reward = official_score

        if verbose:
            print(f"[GeoGuessr Reward] Predicted: {predicted_coords}, GT: {gt_coords}, "
                  f"Distance: {distance_km:.2f} km, Reward: {reward:.4f}, "
                  f"GeoGuessr Points: {geoguessr_points:.0f}")

        # Return results
        if return_dict:
            result = {
                "score": float(reward),
                "distance@km": float(distance_km),
                "geoguessr@point": float(geoguessr_points),
                "parse_success": True
            }
            # Add accuracy metrics
            result.update(calculate_accuracy_metrics(distance_km))
            return result
        else:
            return float(reward)

    except Exception as e:
        if extra_info and extra_info.get("verbose", False):
            print(f"[GeoGuessr Reward] Exception: {e}")
            print(f"  Solution: {solution_str[:100]}")
            print(f"  Ground truth: {ground_truth}")

        if extra_info and extra_info.get("return_dict", True):
            return {
                "score": 0.0,
                "distance@km": 20000.0,
                "acc@1km": 0.0,
                "acc@25km": 0.0,
                "acc@200km": 0.0,
                "acc@750km": 0.0,
                "acc@2500km": 0.0,
                "geoguessr@point": 0.0,
                "parse_success": False
            }
        else:
            return 0.0


# ============================================================================
# Convenience wrapper functions
# ============================================================================

def geoguessr_reward_linear(data_source: str, solution_str: str,
                            ground_truth: str, extra_info: Optional[dict] = None) -> Union[float, Dict[str, Any]]:
    """
    Linear decay reward.

    Returns dict with 'score' key using linear decay, plus all other metrics.
    """
    info = extra_info or {}
    info["reward_type"] = "linear"
    return geoguessr_reward_function(data_source, solution_str, ground_truth, info)


def geoguessr_reward_official(data_source: str, solution_str: str,
                              ground_truth: str, extra_info: Optional[dict] = None) -> Union[float, Dict[str, Any]]:
    """
    Official GeoGuessr competitive scoring (5000-point scale, normalized to [0,1]).

    Returns dict with 'score' key using official scoring, plus all other metrics.
    """
    info = extra_info or {}
    info["reward_type"] = "official"
    return geoguessr_reward_function(data_source, solution_str, ground_truth, info)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example 1: Official scoring with dict return
    solution1 = "Based on the architecture and vegetation, this is New York City. \\boxed{40.7128, -74.0060}"
    ground_truth1 = "40.7128, -74.02"  # Eiffel Tower

    result1 = geoguessr_reward_official("test", solution1, ground_truth1, {"verbose": True})
    print("\nExample 1 - Official scoring:")
    print(result1)

    # Example 2: Linear scoring with dict return
    result2 = geoguessr_reward_linear("test", solution1, ground_truth1, {"verbose": True})
    print("\nExample 2 - Linear scoring:")
    print(result2)

    # Example 3: Backward compatible (float return)
    result3 = geoguessr_reward_official("test", solution1, ground_truth1,
                                        {"verbose": True, "return_dict": False})
    print("\nExample 3 - Backward compatible float:")
    print(f"Score: {result3}")

    # Example 4: Parse failure
    solution4 = "I cannot determine the location."
    result4 = geoguessr_reward_official("test", solution4, ground_truth1, {"verbose": True})
    print("\nExample 4 - Parse failure:")
    print(result4)