#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import select
import sys
import time
from queue import Queue
from typing import Any

from lerobot.processor import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_keyboard import (
    KeyboardEndEffectorTeleopConfig,
    KeyboardRoverTeleopConfig,
    KeyboardTeleopConfig,
)

PYNPUT_AVAILABLE = True
try:
    if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
        logging.info("No DISPLAY set. Skipping pynput import.")
        raise ImportError("pynput blocked intentionally due to no display.")

    from pynput import keyboard
except ImportError:
    keyboard = None
    PYNPUT_AVAILABLE = False
except Exception as e:
    keyboard = None
    PYNPUT_AVAILABLE = False
    logging.info(f"Could not import pynput: {e}")


class KeyboardTeleop(Teleoperator):
    """
    Teleop class to use keyboard inputs for control.
    """

    config_class = KeyboardTeleopConfig
    name = "keyboard"

    def __init__(self, config: KeyboardTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.event_queue = Queue()
        self.current_pressed = {}
        self.listener = None
        self.logs = {}
        self.stdin_mode = False
        self._stdin_fd = None
        self._stdin_old_settings = None
        self._stdin_owns_fd = False

    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (len(self.arm),),
            "names": {"motors": list(self.arm.motors)},
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        pynput_ok = PYNPUT_AVAILABLE and isinstance(self.listener, keyboard.Listener) and self.listener.is_alive()
        return pynput_ok or self.stdin_mode

    @property
    def is_calibrated(self) -> bool:
        pass

    @check_if_already_connected
    def connect(self) -> None:
        self.listener = None
        self.stdin_mode = False

        if PYNPUT_AVAILABLE:
            logging.info("pynput is available - enabling local keyboard listener.")
            self.listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self.listener.start()
        else:
            logging.info("pynput not available.")

        # Always try stdin fallback in terminal sessions.
        # This makes keyboard control work in environments where pynput imports
        # but does not actually receive events (remote shell/VM/IDE terminal).
        if sys.stdin.isatty():
            try:
                import termios
                import tty

                self._stdin_fd = sys.stdin.fileno()
                self._stdin_old_settings = termios.tcgetattr(self._stdin_fd)
                tty.setcbreak(self._stdin_fd)
                self.stdin_mode = True
                logging.info("stdin fallback enabled.")
            except Exception as e:
                logging.info(f"stdin fallback init failed: {e}")
                self.stdin_mode = False
        else:
            # Some IDE terminals expose non-tty stdin to Python.
            # Fallback to controlling terminal directly.
            try:
                import termios
                import tty

                self._stdin_fd = os.open("/dev/tty", os.O_RDONLY | os.O_NONBLOCK)
                self._stdin_old_settings = termios.tcgetattr(self._stdin_fd)
                tty.setcbreak(self._stdin_fd)
                self._stdin_owns_fd = True
                self.stdin_mode = True
                logging.info("/dev/tty fallback enabled.")
            except Exception as e:
                logging.info(f"/dev/tty fallback init failed: {e}")
                self.stdin_mode = False

    def calibrate(self) -> None:
        pass

    def _on_press(self, key):
        # Keep chars as strings ("s", "q", "r"), but keep special keys
        # (arrows/shift/ctrl/esc) as pynput key objects so keyboard_ee can
        # map them to delta actions.
        if hasattr(key, "char") and key.char is not None:
            self.event_queue.put((key.char, True))
        else:
            self.event_queue.put((key, True))

    def _on_release(self, key):
        if hasattr(key, "char") and key.char is not None:
            self.event_queue.put((key.char, False))
        else:
            self.event_queue.put((key, False))
        if key == keyboard.Key.esc:
            logging.info("ESC pressed, disconnecting.")
            self.disconnect()

    def _drain_pressed_keys(self):
        # stdin fallback: consume typed chars as one-tick events
        if self.stdin_mode and self._stdin_fd is not None:
            while True:
                ready, _, _ = select.select([self._stdin_fd], [], [], 0)
                if not ready:
                    break
                ch = os.read(self._stdin_fd, 1)
                if not ch:
                    break
                try:
                    key_char = ch.decode("utf-8", errors="ignore")
                except Exception:
                    key_char = ""
                if key_char:
                    self.current_pressed[key_char] = True

        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            self.current_pressed[key_char] = is_pressed

    def configure(self):
        pass

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        before_read_t = time.perf_counter()

        self._drain_pressed_keys()

        # Generate action based on current key states
        action = {key for key, val in self.current_pressed.items() if val}
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return dict.fromkeys(action, None)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.listener is not None:
            self.listener.stop()
        if self.stdin_mode and self._stdin_fd is not None and self._stdin_old_settings is not None:
            try:
                import termios

                termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._stdin_old_settings)
            except Exception:
                pass
        if self._stdin_owns_fd and self._stdin_fd is not None:
            try:
                os.close(self._stdin_fd)
            except Exception:
                pass
        self._stdin_owns_fd = False
        self.stdin_mode = False


class KeyboardEndEffectorTeleop(KeyboardTeleop):
    """
    Teleop class to use keyboard inputs for end effector control.
    Designed to be used with the `So100FollowerEndEffector` robot.
    """

    config_class = KeyboardEndEffectorTeleopConfig
    name = "keyboard_ee"

    def __init__(self, config: KeyboardEndEffectorTeleopConfig):
        super().__init__(config)
        self.config = config
        self.misc_keys_queue = Queue()
        # Track last computed action so get_teleop_events can infer intervention
        # even when key state cache is cleared in get_action().
        self._last_action: dict[str, float] = {
            "delta_x": 0.0,
            "delta_y": 0.0,
            "delta_z": 0.0,
            "gripper": 1.0,
        }

    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (4,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
            }
        else:
            return {
                "dtype": "float32",
                "shape": (3,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
            }

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        self._drain_pressed_keys()
        delta_x = 0.0
        delta_y = 0.0
        delta_z = 0.0
        gripper_action = 1.0

        # Generate action based on current key states
        for key, val in self.current_pressed.items():
            if key == keyboard.Key.up or key == "i":
                delta_y = -int(val)
            elif key == keyboard.Key.down or key == "k":
                delta_y = int(val)
            elif key == keyboard.Key.left or key == "j":
                delta_x = int(val)
            elif key == keyboard.Key.right or key == "l":
                delta_x = -int(val)
            elif key == keyboard.Key.shift or key == "u":
                delta_z = -int(val)
            elif key == keyboard.Key.shift_r or key == "o":
                delta_z = int(val)
            elif key == keyboard.Key.ctrl_r or key == "m":
                # Gripper actions are expected to be between 0 (close), 1 (stay), 2 (open)
                gripper_action = int(val) + 1
            elif key == keyboard.Key.ctrl_l or key == "n":
                gripper_action = int(val) - 1
            elif val:
                # If the key is pressed, add it to the misc_keys_queue
                # this will record key presses that are not part of the delta_x, delta_y, delta_z
                # this is useful for retrieving other events like interventions for RL, episode success, etc.
                self.misc_keys_queue.put(key)

        action_dict = {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "delta_z": delta_z,
        }

        if self.config.use_gripper:
            action_dict["gripper"] = gripper_action

        # In stdin fallback mode, chars are one-shot pulses.
        if self.stdin_mode:
            self.current_pressed.clear()

        self._last_action = dict(action_dict)
        return action_dict

    def get_teleop_events(self) -> dict[str, Any]:
        """
        Get extra control events from the keyboard such as intervention status,
        episode termination, success indicators, etc.

        Keyboard mappings:
        - Any movement keys pressed = intervention active
        - 's' key = success (terminate episode successfully)
        - 'r' key = rerecord episode (terminate and rerecord)
        - 'q' key = quit episode (terminate without success)

        Returns:
            Dictionary containing:
                - is_intervention: bool - Whether human is currently intervening
                - terminate_episode: bool - Whether to terminate the current episode
                - success: bool - Whether the episode was successful
                - rerecord_episode: bool - Whether to rerecord the episode
        """
        if not self.is_connected:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        # Infer intervention from the last produced teleop action to avoid races with
        # key cache clearing in get_action().
        is_intervention = (
            abs(float(self._last_action.get("delta_x", 0.0))) > 1e-6
            or abs(float(self._last_action.get("delta_y", 0.0))) > 1e-6
            or abs(float(self._last_action.get("delta_z", 0.0))) > 1e-6
            or (self.config.use_gripper and abs(float(self._last_action.get("gripper", 1.0)) - 1.0) > 1e-6)
        )

        # Check for episode control commands from misc_keys_queue
        terminate_episode = False
        success = False
        rerecord_episode = False

        # Process any pending misc keys
        while not self.misc_keys_queue.empty():
            key = self.misc_keys_queue.get_nowait()
            if key == "s":
                success = True
            elif key == "r":
                terminate_episode = True
                rerecord_episode = True
            elif key == "q":
                terminate_episode = True
                success = False

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }


class KeyboardRoverTeleop(KeyboardTeleop):
    """
    Keyboard teleoperator for mobile robots like EarthRover Mini Plus.

    Provides intuitive WASD-style controls for driving a mobile robot:
    - Linear movement (forward/backward)
    - Angular movement (turning/rotation)
    - Speed adjustment
    - Emergency stop

    Keyboard Controls:
        Movement:
            - W: Move forward
            - S: Move backward
            - A: Turn left (with forward motion)
            - D: Turn right (with forward motion)
            - Q: Rotate left in place
            - E: Rotate right in place
            - X: Emergency stop

        Speed Control:
            - +/=: Increase speed
            - -: Decrease speed

        System:
            - ESC: Disconnect teleoperator

    Attributes:
        config: Teleoperator configuration
        current_linear_speed: Current linear velocity magnitude
        current_angular_speed: Current angular velocity magnitude

    Example:
        ```python
        from lerobot.teleoperators.keyboard import KeyboardRoverTeleop, KeyboardRoverTeleopConfig

        teleop = KeyboardRoverTeleop(
            KeyboardRoverTeleopConfig(linear_speed=1.0, angular_speed=1.0, speed_increment=0.1)
        )
        teleop.connect()

        while teleop.is_connected:
            action = teleop.get_action()
            robot.send_action(action)
        ```
    """

    config_class = KeyboardRoverTeleopConfig
    name = "keyboard_rover"

    def __init__(self, config: KeyboardRoverTeleopConfig):
        super().__init__(config)
        # Add rover-specific speed settings
        self.current_linear_speed = config.linear_speed
        self.current_angular_speed = config.angular_speed

    @property
    def action_features(self) -> dict:
        """Return action format for rover (linear and angular velocities)."""
        return {
            "linear.vel": float,
            "angular.vel": float,
        }

    @property
    def is_calibrated(self) -> bool:
        """Rover teleop doesn't require calibration."""
        return True

    def _drain_pressed_keys(self):
        """Update current_pressed state from event queue without clearing held keys"""
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            if is_pressed:
                self.current_pressed[key_char] = True
            else:
                # Only remove key if it's being released
                self.current_pressed.pop(key_char, None)

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """
        Get the current action based on pressed keys.

        Returns:
            RobotAction with 'linear.vel' and 'angular.vel' keys
        """
        before_read_t = time.perf_counter()

        self._drain_pressed_keys()

        linear_velocity = 0.0
        angular_velocity = 0.0

        # Check which keys are currently pressed (not released)
        active_keys = {key for key, is_pressed in self.current_pressed.items() if is_pressed}

        # Linear movement (W/S) - these take priority
        if "w" in active_keys:
            linear_velocity = self.current_linear_speed
        elif "s" in active_keys:
            linear_velocity = -self.current_linear_speed

        # Turning (A/D/Q/E)
        if "d" in active_keys:
            angular_velocity = -self.current_angular_speed
            if linear_velocity == 0:  # If not moving forward/back, add slight forward motion
                linear_velocity = self.current_linear_speed * self.config.turn_assist_ratio
        elif "a" in active_keys:
            angular_velocity = self.current_angular_speed
            if linear_velocity == 0:  # If not moving forward/back, add slight forward motion
                linear_velocity = self.current_linear_speed * self.config.turn_assist_ratio
        elif "q" in active_keys:
            angular_velocity = self.current_angular_speed
            linear_velocity = 0  # Rotate in place
        elif "e" in active_keys:
            angular_velocity = -self.current_angular_speed
            linear_velocity = 0  # Rotate in place

        # Stop (X) - overrides everything
        if "x" in active_keys:
            linear_velocity = 0
            angular_velocity = 0

        # Speed adjustment
        if "+" in active_keys or "=" in active_keys:
            self.current_linear_speed += self.config.speed_increment
            self.current_angular_speed += self.config.speed_increment * self.config.angular_speed_ratio
            logging.info(
                f"Speed increased: linear={self.current_linear_speed:.2f}, angular={self.current_angular_speed:.2f}"
            )
        if "-" in active_keys:
            self.current_linear_speed = max(
                self.config.min_linear_speed, self.current_linear_speed - self.config.speed_increment
            )
            self.current_angular_speed = max(
                self.config.min_angular_speed,
                self.current_angular_speed - self.config.speed_increment * self.config.angular_speed_ratio,
            )
            logging.info(
                f"Speed decreased: linear={self.current_linear_speed:.2f}, angular={self.current_angular_speed:.2f}"
            )

        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return {
            "linear.vel": linear_velocity,
            "angular.vel": angular_velocity,
        }
