"""
Physics simulation engine for bouncing dot animations.

Provides shared physics logic for single and multiple dot scenarios,
including wall collisions, dot-to-dot collisions (with spatial hashing
optimization), and bounce event tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from .config import AnimationConfig
from .constants import (
    MIN_SIMULATION_TIME_BEFORE_STOP,
    STOPPING_FRAME_PADDING,
    STOPPING_VELOCITY_THRESHOLD,
)


class DotState(TypedDict):
    """
    Type definition for a dot's simulation state.

    Attributes:
        pos: Current position (3D numpy array)
        vel: Current velocity (3D numpy array)
        damping: Damping factor for collisions
        radius: Dot radius
        color: Dot color (string)
        positions: List of recorded positions over time
    """

    pos: NDArray[np.floating]
    vel: NDArray[np.floating]
    damping: float
    radius: float
    color: str
    positions: list[NDArray[np.floating]]


@dataclass
class BounceEvent:
    """
    Records a bounce event for audio generation.

    Attributes:
        time: Simulation time when bounce occurred (seconds)
        speed: Speed at moment of bounce (for volume calculation)
        bounce_type: Type of collision ('wall' or 'dot')
        dot_indices: Indices of dots involved (single for wall, pair for dot collision)
    """

    time: float
    speed: float
    bounce_type: str = "wall"
    dot_indices: tuple[int, ...] = field(default_factory=lambda: (0,))


class SpatialHashGrid:
    """
    Grid-based spatial hashing for O(n) average collision detection.

    Divides the simulation space into cells. Each cell tracks which dots
    are in it, allowing collision checks to only consider nearby dots
    instead of all pairs.

    Attributes:
        cell_size: Size of each grid cell (should be >= 2 * max_dot_radius)
        bounds: Simulation boundaries (min_x, min_y, max_x, max_y)
    """

    def __init__(self, cell_size: float, bounds: tuple[float, float]) -> None:
        """
        Initialize the spatial hash grid.

        Args:
            cell_size: Size of each cell (should be >= 2 * max_dot_radius for correctness)
            bounds: Tuple of (min_x, min_y) defining the simulation area
        """
        self.cell_size = cell_size
        self.bounds = bounds
        self._grid: dict[tuple[int, int], list[int]] = {}

    def _hash(self, pos: NDArray[np.floating]) -> tuple[int, int]:
        """Convert a position to grid cell coordinates."""
        return (
            int((pos[0] - self.bounds[0]) / self.cell_size),
            int((pos[1] - self.bounds[1]) / self.cell_size),
        )

    def clear(self) -> None:
        """Clear all dots from the grid."""
        self._grid.clear()

    def insert(self, dot_index: int, pos: NDArray[np.floating]) -> None:
        """
        Insert a dot into the grid at its current position.

        Args:
            dot_index: Index of the dot in the dot_states list
            pos: Current position of the dot
        """
        cell = self._hash(pos)
        if cell not in self._grid:
            self._grid[cell] = []
        self._grid[cell].append(dot_index)

    def get_nearby(self, pos: NDArray[np.floating]) -> list[int]:
        """
        Get indices of all dots in the current cell and neighboring cells.

        Args:
            pos: Position to search around

        Returns:
            List of dot indices that could potentially collide with this position
        """
        cx, cy = self._hash(pos)
        nearby: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cell = (cx + dx, cy + dy)
                if cell in self._grid:
                    nearby.extend(self._grid[cell])
        return nearby


class PhysicsSimulation:
    """
    Physics simulation engine for bouncing dots.

    Handles gravity, wall collisions, and dot-to-dot collisions with
    configurable parameters. Uses spatial hashing for efficient multi-dot
    collision detection.

    Example:
        config = AnimationConfig()
        dot_states = [
            {"pos": np.array([0, 0, 0]), "vel": np.array([1, -5, 0]),
             "damping": 0.95, "radius": 0.2, "color": "RED", "positions": []},
        ]
        sim = PhysicsSimulation(config, dot_states)
        positions_per_dot, bounce_events = sim.simulate()
    """

    def __init__(
        self,
        config: AnimationConfig,
        dot_states: list[DotState],
        debug: bool = False,
    ) -> None:
        """
        Initialize the physics simulation.

        Args:
            config: AnimationConfig instance with physics parameters
            dot_states: List of dot state dictionaries (will be modified in place)
            debug: Enable debug output
        """
        self.config = config
        self.dot_states = dot_states
        self.debug = debug

        # Pre-compute collision boundary
        self.circle_radius = config.CIRCLE_RADIUS

        # Initialize spatial hash grid for multi-dot collision detection
        # Cell size = 2.5 * max_radius for good balance of accuracy and performance
        max_radius = max(state["radius"] for state in dot_states) if dot_states else 0.2
        cell_size = 2.5 * max_radius
        bounds = (-self.circle_radius, -self.circle_radius)
        self._spatial_grid = SpatialHashGrid(cell_size, bounds)

        # Track bounce events for audio
        self._bounce_events: list[BounceEvent] = []
        self._bounce_count = 0

    def simulate(self) -> tuple[list[list[NDArray[np.floating]]], list[BounceEvent]]:
        """
        Run the full physics simulation.

        Returns:
            Tuple of:
                - List of position lists, one per dot (each position is a 3D numpy array)
                - List of BounceEvent objects for audio generation
        """
        cfg = self.config
        dt = cfg.SIMULATION_DT
        max_time = cfg.MAX_SIMULATION_TIME
        gravity = cfg.GRAVITY
        num_dots = len(self.dot_states)

        # Initialize positions lists
        for state in self.dot_states:
            state["positions"] = [state["pos"].copy()]

        current_time: float = 0.0

        if self.debug:
            print(f"Starting simulation with {num_dots} dot(s)...")
            print(f"Circle radius: {self.circle_radius}")

        # Main simulation loop
        while current_time < max_time:
            # Update each dot's physics
            for i, state in enumerate(self.dot_states):
                self._update_dot_physics(state, gravity, dt, i, current_time)

            # Check dot-to-dot collisions (only if multiple dots)
            if num_dots > 1:
                self._handle_dot_collisions(current_time)

            # Store positions for animation
            for state in self.dot_states:
                state["positions"].append(state["pos"].copy())

            current_time += dt

            # Check stopping condition
            if self._check_stopping_condition(current_time):
                # Append padding frames
                for state in self.dot_states:
                    state["positions"].extend(
                        [state["pos"].copy() for _ in range(STOPPING_FRAME_PADDING)]
                    )
                break

        if self.debug:
            print("\nSimulation complete:")
            print(f"  Total positions per dot: {len(self.dot_states[0]['positions'])}")
            print(f"  Total bounces: {self._bounce_count}")
            print(f"  Simulation duration: {current_time:.2f}s")
            print()

        # Extract positions lists
        positions_per_dot = [state["positions"] for state in self.dot_states]
        return positions_per_dot, self._bounce_events

    def _update_dot_physics(
        self,
        state: DotState,
        gravity: NDArray[np.floating],
        dt: float,
        dot_index: int,
        current_time: float,
    ) -> None:
        """
        Update a single dot's position and velocity, handling wall collisions.

        Args:
            state: The dot's state dictionary (modified in place)
            gravity: Gravity vector
            dt: Time step
            dot_index: Index of this dot (for bounce events)
            current_time: Current simulation time
        """
        # Apply gravity
        state["vel"] = state["vel"] + gravity * dt
        new_pos = state["pos"] + state["vel"] * dt

        # Check wall collision
        dot_collision_distance = self.circle_radius - state["radius"]
        distance_from_center = np.linalg.norm(new_pos[:2])

        if distance_from_center > dot_collision_distance:
            # Record bounce event
            self._bounce_count += 1
            speed = float(np.linalg.norm(state["vel"]))
            self._bounce_events.append(
                BounceEvent(
                    time=current_time,
                    speed=speed,
                    bounce_type="wall",
                    dot_indices=(dot_index,),
                )
            )

            if self.debug:
                print(f"Wall bounce #{self._bounce_count} at t={current_time:.2f}s")

            # Collision response - reflect velocity off boundary
            direction_2d = new_pos[:2] / distance_from_center
            new_pos[:2] = direction_2d * dot_collision_distance
            normal_2d = -direction_2d
            vel_normal = np.dot(state["vel"][:2], normal_2d)

            if vel_normal < 0:
                state["vel"][:2] -= 2 * vel_normal * normal_2d
                state["vel"] *= state["damping"]

        state["pos"] = new_pos

    def _handle_dot_collisions(self, current_time: float) -> None:
        """
        Check and resolve dot-to-dot collisions using spatial hashing.

        Uses spatial hash grid to reduce collision checks from O(n^2) to O(n) average.

        Args:
            current_time: Current simulation time (for bounce events)
        """
        num_dots = len(self.dot_states)
        if num_dots < 2:
            return

        # Rebuild spatial hash grid with current positions
        self._spatial_grid.clear()
        for i, state in enumerate(self.dot_states):
            self._spatial_grid.insert(i, state["pos"])

        # Track checked pairs to avoid duplicate checks
        checked_pairs: set[tuple[int, int]] = set()

        for i, state_i in enumerate(self.dot_states):
            nearby = self._spatial_grid.get_nearby(state_i["pos"])

            for j in nearby:
                if i >= j:
                    continue  # Skip self and already-checked pairs

                pair = (i, j)
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)

                state_j = self.dot_states[j]
                self._resolve_dot_collision(state_i, state_j, i, j, current_time)

    def _resolve_dot_collision(
        self,
        state_i: DotState,
        state_j: DotState,
        i: int,
        j: int,
        current_time: float,
    ) -> None:
        """
        Check and resolve collision between two dots.

        Args:
            state_i: First dot's state
            state_j: Second dot's state
            i: Index of first dot
            j: Index of second dot
            current_time: Current simulation time
        """
        diff = state_i["pos"] - state_j["pos"]
        dist = float(np.linalg.norm(diff[:2]))
        min_dist = state_i["radius"] + state_j["radius"]

        if dist < min_dist and dist > 0:
            # Collision detected
            self._bounce_count += 1

            # Calculate relative speed for audio
            relative_vel = state_i["vel"] - state_j["vel"]
            speed = float(np.linalg.norm(relative_vel))
            self._bounce_events.append(
                BounceEvent(
                    time=current_time,
                    speed=speed,
                    bounce_type="dot",
                    dot_indices=(i, j),
                )
            )

            if self.debug:
                print(
                    f"Dot collision #{self._bounce_count} between dot {i} and {j} at t={current_time:.2f}s"
                )

            # Normal vector from j to i
            normal = diff[:2] / dist

            # Separate the dots
            overlap = min_dist - dist
            state_i["pos"][:2] += normal * (overlap / 2)
            state_j["pos"][:2] -= normal * (overlap / 2)

            # Calculate relative velocity along normal
            vel_i_normal = float(np.dot(state_i["vel"][:2], normal))
            vel_j_normal = float(np.dot(state_j["vel"][:2], normal))

            # Only resolve if dots are approaching each other
            if vel_i_normal - vel_j_normal < 0:
                avg_damping = (state_i["damping"] + state_j["damping"]) / 2

                state_i["vel"][:2] += (vel_j_normal - vel_i_normal) * normal * avg_damping
                state_j["vel"][:2] += (vel_i_normal - vel_j_normal) * normal * avg_damping

    def _check_stopping_condition(self, current_time: float) -> bool:
        """
        Check if all dots have effectively stopped.

        A dot is considered stopped when:
        1. Its speed is below the threshold
        2. It is resting on the circle boundary (not mid-air)

        Args:
            current_time: Current simulation time

        Returns:
            True if simulation should stop, False otherwise
        """
        # Too early to stop
        if current_time < MIN_SIMULATION_TIME_BEFORE_STOP:
            return False
        
        is_single_dot = len(self.dot_states) == 1

        for state in self.dot_states:
            speed = float(np.linalg.norm(state["vel"]))
            
            # Return if speed is above threshold
            if speed >= STOPPING_VELOCITY_THRESHOLD:
                return False
            
            # check on boundary, if more than one dot, skip this check
            if is_single_dot:
                # Must be on the boundary (not floating mid-air)
                dot_collision_distance = self.circle_radius - state["radius"]
                distance_from_center = float(np.linalg.norm(state["pos"][:2]))

                # Allow small tolerance for "on boundary" check
                boundary_tolerance = state["radius"] * 0.5
                # distance from center is less than this means dot is not touching boundary
                is_not_on_boundary = distance_from_center < (dot_collision_distance - boundary_tolerance)

                if is_not_on_boundary:
                    return False

        return True


def create_dot_states_from_config(
    dots_config: list[dict],
) -> list[DotState]:
    """
    Create DotState list from configuration dictionaries.

    Args:
        dots_config: List of dot configuration dicts with keys:
            - color (str)
            - radius (float)
            - start_pos (list or np.ndarray)
            - initial_velocity (list or np.ndarray)
            - damping (float)

    Returns:
        List of DotState TypedDicts ready for simulation
    """
    dot_states: list[DotState] = []

    for dot_config in dots_config:
        pos = dot_config["start_pos"]
        vel = dot_config["initial_velocity"]

        # Ensure numpy arrays
        if not isinstance(pos, np.ndarray):
            pos = np.array(pos, dtype=np.float64)
        if not isinstance(vel, np.ndarray):
            vel = np.array(vel, dtype=np.float64)

        dot_states.append(
            {
                "pos": pos.copy(),
                "vel": vel.copy(),
                "damping": float(dot_config["damping"]),
                "radius": float(dot_config["radius"]),
                "color": str(dot_config["color"]),
                "positions": [],
            }
        )

    return dot_states
