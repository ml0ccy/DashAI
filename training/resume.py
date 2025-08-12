import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from training.callbacks import AutoSaveCallback
from training.window import focus_gd_window


def resume_training(env_class,
                    capture_cfg,
                    model_path="models/gd_live_ppo.zip",
                    total_timesteps=300_000,
                    save_freq=2500,
                    lr=3e-4,
                    n_steps=512,
                    batch_size=128,
                    n_epochs=8,
                    ent_coef=0.008,
                    gamma=0.999,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    target_kl=0.03):
    """
    Возобновление обучения PPO модели для Geometry Dash.

    env_class — класс окружения (обычно envs.gd_env_soft.GeometryDashLiveEnv)
    """
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

    # Создаём окружение в точно такой же обёртке, как при тренировке
    def make_env():
        return env_class(capture=capture_cfg, render_mode=None)

    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4, channels_order="first")

    # Загружаем существующую модель или создаём новую
    if os.path.isfile(model_path):
        print(f"🔄 Загружаю модель из чекпоинта: {model_path}")
        model = PPO.load(model_path, device="cpu")
        model.set_env(env)
        # Можно поменять LR прямо на лету
        model.learning_rate = lr
    else:
        print("ℹ️ Чекпоинт не найден — стартуем новую модель.")
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            device="cpu",
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gae_lambda=gae_lambda,
            gamma=gamma,
            ent_coef=ent_coef,
            learning_rate=lr,
            clip_range=clip_range,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
        )

    focus_gd_window()
    callback = AutoSaveCallback(save_freq=save_freq, save_path=model_path)

    # reset_num_timesteps=False — важно для корректного продолжения
    model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=1, reset_num_timesteps=False)
    model.save(model_path)
    print(f"✅ Сохранено после дообучения: {model_path}")
    env.close()
