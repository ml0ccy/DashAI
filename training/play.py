import numpy as np
import keyboard
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack


def play_model(env_class, capture_cfg, model_path="models/gd_live_ppo.zip"):
    """
    Запуск обученной модели в режиме игры (инференс).

    env_class — класс окружения:
        - envs.gd_env_soft.GeometryDashLiveEnv (рекомендуется)
        - envs.gd_env_strict.GeometryDashLiveEnv (исторический вариант)
    """
    # Создаём окружение в тех же обёртках, что и при обучении
    def make_env():
        return env_class(capture=capture_cfg, render_mode=None)

    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4, channels_order="first")

    # Инициализируем среду и загружаем модель
    obs, info = env.reset()
    model = PPO.load(model_path, device="cpu")

    try:
        while True:
            # Позволяем выйти из цикла по ESC
            if keyboard.is_pressed("esc"):
                print("ESC pressed, exiting play loop.")
                break

            # Предсказание действия и шаг среды
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            # Если эпизод закончился (детект смерти), делаем reset
            if np.any(terminated) or np.any(truncated):
                print("Episode ended (death detected). Reset when ready.")
                obs, info = env.reset()
    finally:
        env.close()
