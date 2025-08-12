import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from training.callbacks import AutoSaveCallback
from training.window import focus_gd_window


def train_model(env_class, capture_cfg, total_timesteps=300_000,
                show_window=False, model_path="models/gd_live_ppo.zip"):
    """
    Универсальная функция тренировки модели PPO для Geometry Dash.
    env_class — класс окружения (envs.gd_env_strict.GeometryDashLiveEnv или envs.gd_env_soft.GeometryDashLiveEnv)
    """
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

    def make_env():
        return env_class(capture=capture_cfg, render_mode=None)

    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4, channels_order="first")

    # Параметры сохранены как в исходных кодах
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        device="cpu",
        n_steps=512,
        batch_size=128,
        n_epochs=12,
        gae_lambda=0.95,
        gamma=0.999,
        ent_coef=0.02,
        learning_rate=1e-3,
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.03,
    )

    focus_gd_window()
    callback = AutoSaveCallback(save_freq=2500, save_path=model_path)

    model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=1)
    model.save(model_path)
    print(f"✅ Финальное сохранение: {model_path}")
    env.close()
