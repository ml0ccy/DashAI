import time
import numpy as np
import cv2
import mss
import pyautogui
import gymnasium as gym
from gymnasium import spaces

class GeometryDashLiveEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        capture,
        render_mode=None,
        template_threshold=0.88,
        confirm_frames=3,
        death_cooldown=2.0,
        post_restart_mute=0.5,
        time_reward_base=0.02,
        time_reward_scale=0.04,
        time_reward_cap=0.2,
        time_reward_warmup=0.5,
        death_penalty_min=8.0,
        death_penalty_slope=2.0,
        death_penalty_cap=25.0
    ):
        super().__init__()
        self.capture_cfg = capture
        self.render_mode = render_mode

        self.attempt_templ = cv2.imread("templates/attempt.png", cv2.IMREAD_GRAYSCALE)
        if self.attempt_templ is None:
            print("⚠️ Шаблон 'attempt.png' не найден.")

        self.obs_h, self.obs_w = 84, 84
        self.n_channels = 1
        self.observation_shape = (self.obs_h, self.obs_w, self.n_channels)
        self.observation_space = spaces.Box(low=0, high=255, shape=self.observation_shape, dtype=np.uint8)
        self.action_space = spaces.Discrete(2)

        self.sct = mss.mss()
        self.monitor = {
            "left": capture.left,
            "top": capture.top,
            "width": capture.width,
            "height": capture.height,
        }

        self._last_frame_bgr = None
        self._last_obs = None

        self.dt = 1.0 / max(1, capture.target_fps)
        self._last_step_time = 0.0

        self.steps = 0

        self.template_threshold = float(template_threshold)
        self.confirm_frames = int(confirm_frames)
        self._dead_confirm_counter = 0

        self._gray_resized = np.empty((self.obs_h, self.obs_w), dtype=np.uint8)

        now = time.time()
        self._last_reward_time = now
        self._episode_start_time = now

        self.death_cooldown = float(death_cooldown)
        self.post_restart_mute = float(post_restart_mute)
        self._last_death_time = -1e9
        self._post_restart_mute_until = -1e9
        self._death_gate_until = -1e9

        self.time_reward_base = float(time_reward_base)
        self.time_reward_scale = float(time_reward_scale)
        self.time_reward_cap = float(time_reward_cap)
        self.time_reward_warmup = float(time_reward_warmup)

        self.death_penalty_min = float(death_penalty_min)
        self.death_penalty_slope = float(death_penalty_slope)
        self.death_penalty_cap = float(death_penalty_cap)

    def _grab_frame(self):
        img = np.array(self.sct.grab(self.monitor))
        frame_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return frame_bgr

    def _to_gray_and_resize(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        cv2.resize(gray, (self.obs_w, self.obs_h), dst=self._gray_resized, interpolation=cv2.INTER_AREA)
        obs = self._gray_resized[:, :, None].astype(np.uint8)
        return gray, obs

    def _detect_attempt_by_template_gray(self, gray) -> bool:
        if self.attempt_templ is None:
            self._dead_confirm_counter = 0
            return False
        res = cv2.matchTemplate(gray, self.attempt_templ, cv2.TM_CCOEFF_NORMED)
        hit = np.any(res >= self.template_threshold)
        if hit:
            self._dead_confirm_counter += 1
        else:
            self._dead_confirm_counter = 0
        return self._dead_confirm_counter >= self.confirm_frames

    def _compute_time_reward_increasing(self):
        now = time.time()
        elapsed = now - self._last_reward_time
        self._last_reward_time = now
        seconds = max(elapsed, 0.0)

        t_since_start = max(0.0, now - self._episode_start_time)
        if self.time_reward_warmup > 1e-6:
            growth = 1.0 - np.exp(-t_since_start / self.time_reward_warmup)
        else:
            growth = 1.0
        rate = self.time_reward_base + self.time_reward_scale * growth
        rate = min(rate, self.time_reward_cap)
        return seconds * rate

    def _compute_smart_death_penalty(self, t_episode: float) -> float:
        t = max(0.0, float(t_episode))
        penalty_mag = self.death_penalty_min + self.death_penalty_slope * np.sqrt(t)
        penalty_mag = min(penalty_mag, self.death_penalty_cap)
        return -float(penalty_mag)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self._dead_confirm_counter = 0
        now = time.time()
        self._last_reward_time = now
        self._episode_start_time = now

        frame = self._grab_frame()
        _, obs = self._to_gray_and_resize(frame)
        self._last_frame_bgr = frame
        self._last_obs = obs
        info = {}
        return obs, info

    def step(self, action):
        now = time.time()
        delay = self.dt - (now - self._last_step_time)
        if delay > 0:
            time.sleep(delay)
        self._last_step_time = time.time()

        if action == 1:
            pyautogui.keyDown("space")
            pyautogui.keyUp("space")

        frame = self._grab_frame()
        gray, obs = self._to_gray_and_resize(frame)
        self._last_frame_bgr = frame
        self._last_obs = obs

        reward = self._compute_time_reward_increasing()

        t = time.time()
        gate_active = t < self._death_gate_until
        in_post_mute = t < self._post_restart_mute_until
        forbid_detect = gate_active or in_post_mute

        if forbid_detect:
            self._dead_confirm_counter = 0
            is_dead = False
        else:
            is_dead = self._detect_attempt_by_template_gray(gray)

        terminated = bool(is_dead)
        truncated = False

        if terminated:
            t_episode = t - self._episode_start_time
            penalty = self._compute_smart_death_penalty(t_episode)
            reward += penalty
            print(f"💀 Смерть на шаге {self.steps} (t={t_episode:.2f}s) → penalty={penalty:.2f}, reward={reward:.2f}")

            self._last_death_time = t
            self._death_gate_until = t + self.death_cooldown
            self._post_restart_mute_until = t + self.post_restart_mute
            self._dead_confirm_counter = 0
            pyautogui.press('space')
        else:
            # печать подавлена как во втором файле
            pass

        info = {
            "dead": is_dead,
            "steps": self.steps,
            "raw_reward": reward,
            "episode_time": time.time() - self._episode_start_time,
            "death_gate_left": max(0.0, (self._death_gate_until - time.time())),
            "post_mute_left": max(0.0, (self._post_restart_mute_until - time.time()))
        }

        self.steps += 1
        return obs, float(reward), terminated, truncated, info

    def render(self, mode="human"):
        if mode == "rgb_array" and self._last_frame_bgr is not None:
            return cv2.cvtColor(self._last_frame_bgr, cv2.COLOR_BGR2RGB)
        return None

    def close(self):
        import cv2 as _cv2
        try:
            _cv2.destroyAllWindows()
        except Exception:
            pass
        self.sct.close()
